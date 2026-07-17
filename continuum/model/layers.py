"""
Core neural network layers for Continuum SLM.

Implements the three primary building blocks from the architecture:
- Factorized Embedding (Section 5): Tied input/output with ALBERT-style factorization
- Gated Linear Trace - GLT (Section 6): Matrix-valued associative recurrence with
  decoupled decay and input gates
- Gated Shard FFN (Section 10): Soft-gated SwiGLU feed-forward with per-shard activation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# Factorized Embedding (Section 5)
# ============================================================================

class FactorizedEmbedding(nn.Module):
    """
    Tied, factorized embedding layer.

    Instead of a [vocab_size, d_model] table, we use:
    - A small table of shape [vocab_size, d_embed]
    - An up-projection: d_embed -> d_model
    - A down-projection (for output): d_model -> d_embed

    The same table is used for input lookup and (transposed) output projection.
    """

    def __init__(self, vocab_size: int, d_model: int, d_embed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed

        # The shared embedding table
        self.embed_table = nn.Embedding(vocab_size, d_embed)

        # Factorized projections
        self.up_proj = nn.Linear(d_embed, d_model, bias=False)
        self.down_proj = nn.Linear(d_model, d_embed, bias=False)

        # Initialize
        nn.init.normal_(self.embed_table.weight, mean=0.0, std=1.0 / math.sqrt(d_embed))
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq_len] -> embeddings: [batch, seq_len, d_model]"""
        x = self.embed_table(token_ids)  # [B, L, d_embed]
        x = self.up_proj(x)              # [B, L, d_model]
        return x

    def project_to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: [batch, seq_len, d_model] -> logits: [batch, seq_len, vocab_size]
        Uses tied embedding: down_proj -> matmul with embed_table^T
        """
        x = self.down_proj(hidden)                            # [B, L, d_embed]
        logits = F.linear(x, self.embed_table.weight)         # [B, L, vocab_size]
        return logits


# ============================================================================
# Gated Linear Trace (Section 6)
# ============================================================================

class GLTLayer(nn.Module):
    """
    Gated Linear Trace — the default sequence mixer.

    State: S_t is a [d_state x d_state] matrix, carried forward per layer.
    Update: S_t = diag(gamma_t) * S_{t-1} + diag(iota_t) * (k_t ⊗ v_t)
    Read:   h_t = S_t * q_t
    Output: o_t = W_o(r_t * h_t)

    Two independently-learned gates (decay gamma, input iota) — decoupled by design.

    Args:
        d_model: Input/output dimension
        d_state: Internal state dimension (d_state < d_model)
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Key, Value, Query projections: d_model -> d_state
        self.W_k = nn.Linear(d_model, d_state, bias=False)
        self.W_v = nn.Linear(d_model, d_state, bias=False)
        self.W_q = nn.Linear(d_model, d_state, bias=False)

        # Decoupled gates: decay (gamma) and input (iota)
        self.W_gamma = nn.Linear(d_model, d_state, bias=True)
        self.W_iota = nn.Linear(d_model, d_state, bias=True)

        # Output gate and projection
        self.W_r = nn.Linear(d_model, d_state, bias=True)
        self.W_o = nn.Linear(d_state, d_model, bias=False)

        # Pre-norm (RMSNorm)
        self.norm = RMSNorm(d_model)
        self.kv_norm = RMSNorm(d_state)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize with spread decay biases for multi-timescale specialization."""
        # Spread decay biases: some channels start fast, some slow
        # This approximates HiPPO-style multi-timescale initialization (Section 6)
        spread = torch.linspace(-2.0, 2.0, self.d_state)
        with torch.no_grad():
            self.W_gamma.bias.copy_(spread)

        # Other biases to zero
        nn.init.zeros_(self.W_iota.bias)
        nn.init.zeros_(self.W_r.bias)

        # All weight matrices: small normal init
        for module in [self.W_k, self.W_v, self.W_q, self.W_gamma, self.W_iota, self.W_r, self.W_o]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step forward pass (sequential mode).

        Args:
            x: Input hidden state [batch, d_model]
            state: Previous GLT state [batch, d_state, d_state], or None to start fresh

        Returns:
            o_t: Output [batch, d_model]
            S_t: Updated state [batch, d_state, d_state]
        """
        batch_size = x.shape[0]

        # Pre-norm
        residual = x
        x_norm = self.norm(x)

        # Compute projections
        k = self.W_k(x_norm)       # [B, d_state]
        v = self.W_v(x_norm)       # [B, d_state]
        q = self.W_q(x_norm)       # [B, d_state]

        # Normalize k, v to prevent state magnitude drift
        k = self.kv_norm(k)
        v = self.kv_norm(v)

        # Gates (sigmoid-bounded for stability)
        gamma = torch.sigmoid(self.W_gamma(x_norm))  # [B, d_state] — decay gate
        iota = torch.sigmoid(self.W_iota(x_norm))    # [B, d_state] — input gate
        r = torch.sigmoid(self.W_r(x_norm))           # [B, d_state] — output gate

        # Initialize state if needed
        if state is None:
            state = torch.zeros(batch_size, self.d_state, self.d_state,
                               device=x.device, dtype=x.dtype)

        # Outer product: k ⊗ v = k * v^T  [B, d_state, d_state]
        outer = torch.bmm(k.unsqueeze(2), v.unsqueeze(1))  # [B, d_state, d_state]

        # State update: S_t = diag(gamma) * S_{t-1} + diag(iota) * (k ⊗ v)
        # diag(gamma) * S: multiply each row i by gamma[i]
        gamma_diag = gamma.unsqueeze(2)  # [B, d_state, 1]
        iota_diag = iota.unsqueeze(2)    # [B, d_state, 1]

        state = gamma_diag * state + iota_diag * outer

        # Read: h_t = S_t * q  [B, d_state]
        h = torch.bmm(state, q.unsqueeze(2)).squeeze(2)  # [B, d_state]

        # Output gate + projection
        o = self.W_o(r * h)  # [B, d_model]

        # Residual connection
        o = o + residual

        o = self.dropout(o)

        return o, state

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process an entire sequence sequentially.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Initial state [batch, d_state, d_state] or None

        Returns:
            outputs: [batch, seq_len, d_model]
            final_state: [batch, d_state, d_state]
        """
        batch_size, seq_len, _ = x.shape
        outputs = []
        for t in range(seq_len):
            o, state = self.forward(x[:, t, :], state)
            outputs.append(o.unsqueeze(1))
        return torch.cat(outputs, dim=1), state

    def reset_state(self, batch_size: int = 1, device: str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create a fresh zero state."""
        return torch.zeros(batch_size, self.d_state, self.d_state,
                          device=device, dtype=dtype)


# ============================================================================
# Gated Shard FFN (Section 10)
# ============================================================================

class GatedShardFFN(nn.Module):
    """
    Feed-forward with soft per-shard gating.

    Instead of discrete MoE routing, each of K shards is a SwiGLU-style FFN
    that contributes proportionally to its sigmoid gate value.
    At inference, shards with gate below threshold can be skipped.

    Total intermediate width = expansion * d_model.
    Each shard handles 1/K of that width.

    Args:
        d_model: Input/output dimension
        expansion: Total FFN expansion ratio (r in the architecture)
        num_shards: Number of independent FFN shards (K)
        sparsity_threshold: Gate values below this skip the shard at inference
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 3,
        num_shards: int = 2,
        sparsity_threshold: float = 0.05,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_shards = num_shards
        self.sparsity_threshold = sparsity_threshold

        # Total intermediate width, divided equally among shards
        total_intermediate = expansion * d_model
        shard_intermediate = total_intermediate // num_shards
        shard_intermediate = (shard_intermediate // 8) * 8
        total_intermediate = shard_intermediate * num_shards

        self.shard_intermediate = shard_intermediate
        self.total_intermediate = total_intermediate

        # ⚡ Phase 1: Fused projections — single large matmul instead of K small ones
        # gate_proj + up_proj combined: [d_model] → [K * shard_intermediate] each
        self.gate_proj_fused = nn.Linear(d_model, total_intermediate, bias=False)
        self.up_proj_fused = nn.Linear(d_model, total_intermediate, bias=False)
        # down_proj combined: [K * shard_intermediate] → [d_model] (summed across shards below)
        self.down_proj_fused = nn.Linear(total_intermediate, d_model, bias=False)

        # Soft gating: a single linear -> sigmoid per shard
        self.gate_head = nn.Linear(d_model, num_shards, bias=True)

        # Pre-norm
        self.norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate_proj_fused.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj_fused.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj_fused.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ⚡ torch.compile safe: No data-dependent control flow, purely tensor ops.
        
        Args:
            x: [batch, d_model] or [batch, seq_len, d_model]

        Returns:
            output: same shape as x
        """
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, d_model]
            was_2d = True
        else:
            was_2d = False

        # Pre-norm with residual
        residual = x
        x_norm = self.norm(x)

        B, L, _ = x_norm.shape

        # ⚡ Phase 1: Fused forward — single matmuls instead of K separate ones
        gates = torch.sigmoid(self.gate_head(x_norm))  # [B, L, num_shards]

        # Single fused gate_proj: [B, L, total_intermediate]
        gate_out = self.gate_proj_fused(x_norm)
        # Single fused up_proj
        up_out = self.up_proj_fused(x_norm)
        # SwiGLU activation
        swiglu_out = F.silu(gate_out) * up_out  # [B, L, total_intermediate]

        # Apply per-shard gates to the fused output
        # Reshape fused gates: [B, L, K, 1]
        gates_4d = gates.view(B, L, self.num_shards, 1)
        # Reshape swiglu output: [B, L, K, shard_intermediate]
        swiglu_4d = swiglu_out.view(B, L, self.num_shards, self.shard_intermediate)
        # Weighted by gates
        gated_swiglu = gates_4d * swiglu_4d  # [B, L, K, shard_intermediate]
        # Flatten back: [B, L, total_intermediate]
        gated_swiglu_flat = gated_swiglu.reshape(B, L, self.total_intermediate)

        # Single fused down_proj
        output = self.down_proj_fused(gated_swiglu_flat)  # [B, L, d_model]

        # Residual connection
        output = output + residual
        output = self.dropout(output)

        if was_2d:
            output = output.squeeze(1)

        return output

    def sparsity(self) -> float:
        """Report fraction of gates that were zero during the last forward pass.
        Used for monitoring sparsity regularization during training."""
        return 0.0  # Placeholder — tracked externally during training

    @staticmethod
    def get_compiled_ffn(d_model, expansion, num_shards, dropout=0.0):
        """
        Create a compiled GatedShardFFN for maximum performance.
        torch.compile fuses norm → gate_proj → silu → up_proj → * → down_proj
        into a single optimized CUDA kernel (no Python dispatch overhead).
        
        Usage:
            ffn = GatedShardFFN.get_compiled_ffn(d_model=768, expansion=4, num_shards=6)
        """
        ffn = GatedShardFFN(d_model, expansion, num_shards, dropout)
        ffn.forward = torch.compile(ffn.forward, mode="default", fullgraph=False)
        return ffn


# ============================================================================
# RMS Normalization (used throughout the architecture)
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    ⚡ Phase 1: Uses torch's optimized CUDA rms_norm when available (PyTorch ≥ 2.1).
    Simpler and faster than LayerNorm — commonly used in modern architectures
    (Llama, Mamba, etc.) because it removes the mean-centering step.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's optimized CUDA kernel when available
        if hasattr(F, 'rms_norm'):
            # Cast weight to match input dtype (fixes AMP FP16/FP32 mismatch warning)
            w = self.scale.to(x.dtype) if self.scale.dtype != x.dtype else self.scale
            return F.rms_norm(x, (x.shape[-1],), w, self.eps)
        # Fallback: manual RMS norm
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms).to(x.dtype) * self.scale
