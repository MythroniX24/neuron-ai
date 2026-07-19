"""
Anchor Attention and Persistent Memory Bank for Continuum SLM.

Implements Sections 7 and 12 of the architecture:
- Anchor Attention: bounded-size real softmax attention with local window + anchor tokens
- Persistent Memory Bank: content-addressed long-term memory with gated slot updates
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from continuum.model.layers import RMSNorm


# ============================================================================
# Anchor Attention (Section 7)
# ============================================================================

class AnchorAttention(nn.Module):
    """
    Bounded-size real softmax attention.

    Attends over a FIXED-SIZE set: w local window tokens + m anchor tokens.
    Anchors = static learned registers + top-k PMB readouts.
    Uses ALiBi positional bias (window only) and grouped-query attention.

    Args:
        d_model: Model dimension (192 for Nano)
        n_heads: Number of query heads (4)
        n_kv_heads: Number of key/value heads for GQA (2)
        window_size: Local window size w (48 for Nano)
        n_anchors: Total anchor count m (8 for Nano)
        n_static_anchors: Number of static learned registers (4)
        dropout: Attention dropout rate
    """

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        window_size: int = 48,
        n_anchors: int = 8,
        n_static_anchors: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.n_anchors = n_anchors
        self.n_static_anchors = n_static_anchors
        self.n_pmb_anchors = n_anchors - n_static_anchors

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        assert self.head_dim * n_heads == d_model

        # ⚡ Phase 5: Fused QKV projection — 1 matmul instead of 3
        q_dim = n_heads * self.head_dim
        kv_dim = n_kv_heads * self.head_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.W_qkv = nn.Linear(d_model, q_dim + 2 * kv_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Static anchor registers: learnable parameters
        # These are in d_model space and get projected by W_k/W_v at each step
        self.static_anchors = nn.Parameter(
            torch.randn(n_static_anchors, d_model) * 0.02
        )

        # Track W_qkv format to avoid hasattr checks in hot path (graph-break friendly)
        self._w_qkv_format = "linear"

        # ALiBi slopes: one per head (for window portion only)
        self._init_alibi_slopes()

        # Precompute full ALiBi bias for the window (never changes)
        # Shape: [1, 1, n_heads, window_size] — broadcasts correctly with scores [B, L, n_heads, T]
        distances = torch.arange(self.window_size).float()
        alibi_full = -self.alibi_slopes.view(self.n_heads, 1) * distances.view(1, -1)
        self.register_buffer("alibi_bias_full", alibi_full.view(1, 1, self.n_heads, self.window_size))

        # Pre-norm
        self.norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # ⚡ Phase 1 optimization: cache static anchor K/V (same for all tokens in a forward pass)
        # Regular attributes (not buffers) — these are transient per-forward-pass caches
        self._cached_static_k = None
        self._cached_static_v = None

    def _get_kv_weights(self):
        """Extract K and V weight matrices from fused W_qkv.
        
        Handles regular nn.Linear, QuantizedLinear (INT8), and QuantizedLinearINT4.
        For quantized layers, dequantizes weights back to float on-the-fly.
        Uses a cached format flag to avoid hasattr checks in the hot path.
        """
        fmt = getattr(self, "_w_qkv_format", "linear")
        if fmt == "int8":
            weight = self.W_qkv.weight_int8.float() * self.W_qkv.scale.float()
        elif fmt in ("int4", "quantized"):
            weight = self.W_qkv._dequantize()
        else:
            # Default: regular nn.Linear. If a quantizer forgot to update the flag,
            # fall back gracefully to the first available weight representation.
            try:
                weight = self.W_qkv.weight
            except AttributeError:
                if hasattr(self.W_qkv, "weight_int8"):
                    weight = self.W_qkv.weight_int8.float() * self.W_qkv.scale.float()
                elif hasattr(self.W_qkv, "_dequantize"):
                    weight = self.W_qkv._dequantize()
                else:
                    raise AttributeError(f"W_qkv has no recognized weight format: {type(self.W_qkv)}")
        return (
            weight[self.q_dim:self.q_dim + self.kv_dim],
            weight[self.q_dim + self.kv_dim:self.q_dim + 2 * self.kv_dim],
        )

    def refresh_static_cache(self):
        """Precompute static anchor K/V once per forward pass (not per token)."""
        W_k_w, W_v_w = self._get_kv_weights()
        sk = F.linear(self.static_anchors, W_k_w)
        sv = F.linear(self.static_anchors, W_v_w)
        self._cached_static_k = sk.view(self.n_static_anchors, self.n_kv_heads, self.head_dim).unsqueeze(0)
        self._cached_static_v = sv.view(self.n_static_anchors, self.n_kv_heads, self.head_dim).unsqueeze(0)

    def _init_alibi_slopes(self):
        """Initialize ALiBi slopes geometrically spaced per head."""
        # Standard ALiBi: slopes are powers of 2^(-8/n_heads * i)
        slopes = []
        for i in range(1, self.n_heads + 1):
            slope = 2 ** (-8.0 * i / self.n_heads)
            slopes.append(slope)
        self.register_buffer("alibi_slopes", torch.tensor(slopes))

    def _repeat_kv_for_gqa(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match Q head count for GQA.
        kv: [B, seq_len, n_kv_heads, head_dim]
        Returns: [B, seq_len, n_heads, head_dim]
        """
        n_groups = self.n_heads // self.n_kv_heads
        # ⚡ Phase 4: Use repeat_interleave instead of expand+reshape (avoids intermediate 5D tensor)
        return kv.repeat_interleave(n_groups, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        window_k: torch.Tensor,
        window_v: torch.Tensor,
        pmb_readouts: Optional[torch.Tensor] = None,
        causal_mask: bool = True,
    ) -> torch.Tensor:
        """
        Single-step or sequence forward pass.

        Args:
            x: Input hidden state [B, d_model] or [B, seq_len, d_model]
            window_k: Cached window keys [B, window_size, n_kv_heads, head_dim]
            window_v: Cached window values [B, window_size, n_kv_heads, head_dim]
            pmb_readouts: PMB readout vectors [B, n_pmb_anchors, d_model] or None
            causal_mask: Whether to apply causal masking to window

        Returns:
            output: same shape as x
        """
        # Pre-norm + residual
        residual = x
        x_norm = self.norm(x)

        original_shape = x_norm.shape
        if x_norm.dim() == 2:
            x_norm = x_norm.unsqueeze(1)  # [B, 1, d_model]
            residual = residual.unsqueeze(1)  # Match dims for broadcast-safe addition
            expand_dim = True
        else:
            expand_dim = False

        B, L, _ = x_norm.shape

        # ⚡ Phase 5: Fused QKV — single matmul, then split
        qkv = self.W_qkv(x_norm)  # [B, L, q_dim + 2*kv_dim]
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_kv_heads, self.head_dim)
        v = v.view(B, L, self.n_kv_heads, self.head_dim)

        # Build key/value set: anchors (static + PMB) + window
        # ⚡ Phase 1: Use precomputed static K/V cache (refreshed once per forward pass)
        if self._cached_static_k is not None:
            static_k = self._cached_static_k.expand(B, -1, -1, -1)
            static_v = self._cached_static_v.expand(B, -1, -1, -1)
        else:
            W_k_w, W_v_w = self._get_kv_weights()
            static_k = F.linear(self.static_anchors, W_k_w)
            static_k = static_k.view(self.n_static_anchors, self.n_kv_heads, self.head_dim).unsqueeze(0).expand(B, -1, -1, -1)
            static_v = F.linear(self.static_anchors, W_v_w)
            static_v = static_v.view(self.n_static_anchors, self.n_kv_heads, self.head_dim).unsqueeze(0).expand(B, -1, -1, -1)

        # PMB readouts: project through K/V weights
        if pmb_readouts is not None:
            W_k_w, W_v_w = self._get_kv_weights()
            pmb_k = F.linear(pmb_readouts, W_k_w).view(B, self.n_pmb_anchors, self.n_kv_heads, self.head_dim)
            pmb_v = F.linear(pmb_readouts, W_v_w).view(B, self.n_pmb_anchors, self.n_kv_heads, self.head_dim)
            all_k = torch.cat([static_k, pmb_k, window_k], dim=1)
            all_v = torch.cat([static_v, pmb_v, window_v], dim=1)
        else:
            all_k = torch.cat([static_k, window_k], dim=1)
            all_v = torch.cat([static_v, window_v], dim=1)

        total_kv_len = all_k.shape[1]
        anchor_count = self.n_static_anchors + (self.n_pmb_anchors if pmb_readouts is not None else 0)

        # Repeat K/V for GQA: [B, T, n_kv_heads, hd] → [B, T, n_heads, hd]
        all_k = self._repeat_kv_for_gqa(all_k)
        all_v = self._repeat_kv_for_gqa(all_v)

        # ⚡ Phase 5: SDPA — fused QK^T + softmax + @V in one optimized kernel
        # ALiBi bias: zeros for anchors + learned slopes for window
        anchor_bias = torch.zeros(1, 1, self.n_heads, anchor_count, device=x.device, dtype=q.dtype)
        full_bias = torch.cat([anchor_bias, self.alibi_bias_full.to(device=x.device, dtype=q.dtype)], dim=-1)
        # SDPA expects: [B, n_heads, L, T] or broadcast [1, n_heads, 1, T]
        alibi_attn_mask = full_bias.view(1, self.n_heads, 1, total_kv_len)

        if causal_mask and L > 1:
            # Combine ALiBi bias + causal mask into one float mask
            causal_float = torch.zeros(L, total_kv_len, device=x.device, dtype=q.dtype)
            causal_float[:, anchor_count:] = torch.triu(
                torch.ones(L, self.window_size, device=x.device, dtype=q.dtype) * float('-inf'),
                diagonal=1
            )
            alibi_attn_mask = alibi_attn_mask + causal_float  # broadcast [1, n_heads, 1, T] + [L, T] → [1, n_heads, L, T]

        # SDPA: [B, n_heads, L, hd] output
        # ⚡ Ensure all inputs are same dtype (SDPA requires matching dtypes)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # [B, n_heads, L, hd]
            all_k.transpose(1, 2),  # [B, n_heads, T, hd]
            all_v.transpose(1, 2),  # [B, n_heads, T, hd]
            attn_mask=alibi_attn_mask.to(dtype=q.dtype),
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)

        # Output projection
        output = self.W_o(output)

        # Residual connection
        output = output + residual

        # Restore original shape
        if expand_dim:
            output = output.squeeze(1)

        return output

    def update_window_cache(
        self,
        x: torch.Tensor,
        window_k: torch.Tensor,
        window_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the sliding window K/V cache with a new token.

        Args:
            x: New token hidden state [B, d_model]
            window_k: Current window keys [B, window_size, n_kv_heads, head_dim]
            window_v: Current window values [B, window_size, n_kv_heads, head_dim]

        Returns:
            new_window_k, new_window_v: Updated caches (shifted left + new token appended)
        """
        x_norm = self.norm(x)
        B = x.shape[0]

        # ⚡ Phase 5: Fused QKV — single matmul, then split
        qkv = self.W_qkv(x_norm)
        _, k_flat, v_flat = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        new_k = k_flat.view(B, 1, self.n_kv_heads, self.head_dim)
        new_v = v_flat.view(B, 1, self.n_kv_heads, self.head_dim)

        # Shift window left, append new token
        new_window_k = torch.cat([window_k[:, 1:, :, :], new_k], dim=1)
        new_window_v = torch.cat([window_v[:, 1:, :, :], new_v], dim=1)

        return new_window_k, new_window_v

    def init_window_cache(self, batch_size: int, device: str = "cpu",
                          dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a fresh zero-initialized window cache."""
        shape = (batch_size, self.window_size, self.n_kv_heads, self.head_dim)
        return (torch.zeros(*shape, device=device, dtype=dtype),
                torch.zeros(*shape, device=device, dtype=dtype))


# ============================================================================
# Persistent Memory Bank (Section 12)
# ============================================================================

class PersistentMemoryBank(nn.Module):
    """
    Content-addressed long-term memory with gated slot updates.

    A single shared bank of S slots, each a d_mem-dimensional vector.
    Written once per K-token chunk via content-similarity-based gated update.
    Read by Anchor Attention layers as part of their anchor set.

    Args:
        n_slots: Number of memory slots S (16 for Nano)
        d_mem: Memory slot dimension (d_model = 192 for Nano)
        n_readout: Number of top-k slots to read (n_pmb_anchors = 4 for Nano)
    """

    def __init__(self, n_slots: int = 16, d_mem: int = 192, n_readout: int = 4):
        super().__init__()
        self.n_slots = n_slots
        self.d_mem = d_mem
        self.n_readout = n_readout

        # Memory slots: learnable initial content
        self.slots = nn.Parameter(torch.randn(n_slots, d_mem) * 0.02)

        # Update gate: [slot_vector; chunk_summary] -> single scalar gate
        self.W_update = nn.Linear(2 * d_mem, 1)

        # Write similarity projection (optional: learn to weight dimensions)
        self.write_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_mem)))

        # Initialize
        nn.init.normal_(self.W_update.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_update.bias)

    def read(self, query: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Read top-k most similar slots given a query.

        Args:
            query: [B, d_mem] — typically the pre-projection hidden state
                   or pooled Q vectors
            k: Number of slots to return (default: self.n_readout)

        Returns:
            readouts: [B, k, d_mem] — top-k slot vectors
        """
        if k is None:
            k = self.n_readout

        B = query.shape[0]

        # Scaled dot-product similarity
        similarity = self.write_scale * torch.matmul(
            query, self.slots.T
        )  # [B, n_slots]

        # Top-k selection (differentiable soft top-k)
        # For inference: hard top-k. For training: this is called during attention,
        # and gradients flow through the selected slots via the attention mechanism.
        _, top_indices = torch.topk(similarity, k, dim=-1)  # [B, k]

        # Gather top-k slots
        readouts = self.slots[top_indices]  # [B, k, d_mem]

        return readouts

    def write(self, chunk_summary: torch.Tensor) -> None:
        """
        Write a chunk summary into the memory bank via content-addressed gated update.
        ⚡ Phase 1: Vectorized — all slots updated in parallel (no Python loop).

        Args:
            chunk_summary: [B, d_mem] — pooled representation of last K tokens
        """
        B = chunk_summary.shape[0]

        # Compute similarity between chunk summary and all slots
        similarity = self.write_scale * torch.matmul(
            chunk_summary, self.slots.T
        )  # [B, n_slots]

        # Softmax addressing weights
        addressing = F.softmax(similarity, dim=-1)  # [B, n_slots]

        # ⚡ Vectorized: batch all slots into [B, n_slots, d_mem]
        slots_expanded = self.slots.unsqueeze(0).expand(B, -1, -1)  # [B, n_slots, d_mem]

        # Update gate input: [slot; chunk_summary] for all slots
        chunk_expanded = chunk_summary.unsqueeze(1).expand(-1, self.n_slots, -1)  # [B, n_slots, d_mem]
        gate_input = torch.cat([slots_expanded, chunk_expanded], dim=-1)  # [B, n_slots, 2*d_mem]

        # All update gates simultaneously: [B, n_slots, 1]
        update_gates = torch.sigmoid(self.W_update(gate_input))  # [B, n_slots, 1]

        # Effective update: addressing_weight * update_gate
        effective_updates = addressing.unsqueeze(-1) * update_gates  # [B, n_slots, 1]

        # New slots: (1 - effective) * old + effective * chunk_summary
        # Average across batch dimension
        retain = (1 - effective_updates).mean(0)  # [n_slots, 1]
        update = (effective_updates * chunk_expanded).mean(0)  # [n_slots, d_mem]

        new_slots = retain * self.slots + update

        # In-place update all slots at once
        self.slots.data.copy_(new_slots.data)

    def reset(self):
        """Reset all slots to their initial learned values."""
        nn.init.normal_(self.slots, mean=0.0, std=0.02)

    def serialize(self) -> torch.Tensor:
        """Return slots as a flat tensor for save/load."""
        return self.slots.data.clone()

    def deserialize(self, saved_slots: torch.Tensor):
        """Load slots from a saved tensor."""
        self.slots.data.copy_(saved_slots)



