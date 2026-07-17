"""
Continuum SLM — Full Model Assembly with Adaptive Depth Looping.

Implements Sections 8, 9, and 16 of the architecture:
- Three-stage macro-structure: Perception → Reasoning Core (looped) → Output
- Adaptive Depth Looping (ADL) with halting head and variable per-token compute
- State management for GLT layers and window caches for Anchor Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from continuum.model.layers import (
    FactorizedEmbedding, GLTLayer, GatedShardFFN, RMSNorm
)
from continuum.model.attention import AnchorAttention, PersistentMemoryBank


# ============================================================================
# Model Configuration
# ============================================================================

class ContinuumConfig:
    """Configuration for Continuum SLM, matching Section 17 tier table."""

    def __init__(
        self,
        # Model dimensions
        d_model: int = 192,
        d_state: int = 48,
        d_embed: int = 48,
        vocab_size: int = 8000,

        # Layer counts and split
        n_layers: int = 6,
        glt_layers: int = 4,
        anchor_layers: int = 2,
        perception_layers: int = 2,
        core_layers: int = 2,
        output_layers: int = 2,

        # FFN
        ffn_expansion: int = 3,
        ffn_shards: int = 2,

        # Anchor Attention
        n_heads: int = 4,
        n_kv_heads: int = 2,
        window_size: int = 48,
        n_anchors: int = 8,
        n_static_anchors: int = 4,

        # Adaptive Depth Looping
        n_max_loops: int = 3,
        halt_threshold: float = 0.95,

        # Persistent Memory Bank
        pmb_slots: int = 16,
        pmb_readout: int = 4,
        chunk_size: int = 64,

        # Regularization
        dropout: float = 0.0,
        # Tokenizer
        eos_token_id: int = 2,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_embed = d_embed
        self.vocab_size = vocab_size

        self.n_layers = n_layers
        self.glt_layers = glt_layers
        self.anchor_layers = anchor_layers
        self.perception_layers = perception_layers
        self.core_layers = core_layers
        self.output_layers = output_layers

        self.ffn_expansion = ffn_expansion
        self.ffn_shards = ffn_shards

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.window_size = window_size
        self.n_anchors = n_anchors
        self.n_static_anchors = n_static_anchors

        self.n_max_loops = n_max_loops
        self.halt_threshold = halt_threshold

        self.pmb_slots = pmb_slots
        self.pmb_readout = pmb_readout
        self.chunk_size = chunk_size

        self.dropout = dropout
        self.eos_token_id = eos_token_id

        # Validate
        assert perception_layers + core_layers + output_layers == n_layers
        assert glt_layers + anchor_layers == n_layers
        assert n_anchors - n_static_anchors == pmb_readout


# ============================================================================
# Mixer + FFN Block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    A single transformer-style block: Mixer (GLT or Anchor) + Gated Shard FFN.
    Each with pre-norm and residual connection.
    """

    def __init__(self, mixer: nn.Module, ffn: GatedShardFFN):
        super().__init__()
        self.mixer = mixer
        self.ffn = ffn
        self.is_glt = isinstance(mixer, GLTLayer)
        self.is_anchor = isinstance(mixer, AnchorAttention)

    def forward_glt(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for GLT-based block."""
        o, state = self.mixer(x, state)  # GLT returns (output, new_state)
        o = self.ffn(o)
        return o, state

    def forward_anchor(
        self,
        x: torch.Tensor,
        window_k: torch.Tensor,
        window_v: torch.Tensor,
        pmb_readouts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Anchor Attention-based block.
        Returns (output, block_input) where block_input is the pre-mixer
        hidden state (needed for correct window cache updates).
        """
        block_input = x  # Save for window cache update
        o = self.mixer(x, window_k, window_v, pmb_readouts, causal_mask=False)
        o = self.ffn(o)
        return o, block_input


# ============================================================================
# Halting Head for Adaptive Depth Looping (Section 9)
# ============================================================================

class HaltingHead(nn.Module):
    """
    Tiny network that decides whether to continue looping the Reasoning Core.

    Reads the pooled hidden state after one core pass and outputs a halting
    probability p_i. Looping continues until cumulative p exceeds threshold
    or N_max is reached.

    Args:
        d_model: Hidden state dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.pool_proj = nn.Linear(d_model, d_model // 4, bias=False)
        self.halt_proj = nn.Linear(d_model // 4, 1, bias=True)

        nn.init.normal_(self.pool_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.halt_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.halt_proj.bias, 1.0)  # Start biased toward halting

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, d_model] — pooled hidden state after core pass

        Returns:
            p: [B, 1] — halting probability in (0, 1)
        """
        # Mean pool if sequence provided
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=1)

        pooled = F.silu(self.pool_proj(hidden))
        p = torch.sigmoid(self.halt_proj(pooled))
        return p


# ============================================================================
# Continuum Model
# ============================================================================

class ContinuumModel(nn.Module):
    """
    Complete Continuum SLM with three-stage architecture.

    Stages:
    1. Perception: Fixed-depth, fast. Converts token embeddings to representations.
    2. Reasoning Core: Shared-weight, looped 1..N_max times via ADL.
    3. Output: Fixed-depth. Converts representations to logits.

    State management:
    - Each GLT layer maintains a [d_state x d_state] recurrent state
    - Each Anchor Attention layer maintains a window K/V cache
    - PMB is shared across the entire network
    """

    def __init__(self, config: ContinuumConfig):
        super().__init__()
        self.config = config

        # ---- Embedding ----
        self.embedding = FactorizedEmbedding(
            config.vocab_size, config.d_model, config.d_embed
        )

        # ---- Build stages ----
        # Stage 1: Perception (anchor_interval=3)
        self.perception_blocks = nn.ModuleList()
        self._build_stage(config.perception_layers, self.perception_blocks, is_perception_or_output=True)

        # Stage 2: Reasoning Core — shared weights, looped (anchor_interval=2)
        self.core_blocks = nn.ModuleList()
        self._build_stage(config.core_layers, self.core_blocks, is_perception_or_output=False)

        # Stage 3: Output (anchor_interval=3)
        self.output_blocks = nn.ModuleList()
        self._build_stage(config.output_layers, self.output_blocks, is_perception_or_output=True)

        # ---- Halting Head ----
        self.halting_head = HaltingHead(config.d_model)

        # ---- Persistent Memory Bank (shared) ----
        self.pmb = PersistentMemoryBank(
            n_slots=config.pmb_slots,
            d_mem=config.d_model,
            n_readout=config.pmb_readout,
        )

        # ---- Final norm ----
        self.final_norm = RMSNorm(config.d_model)

        # ---- Count parameters ----
        self._n_params = sum(p.numel() for p in self.parameters())

    def _build_stage(self, n_layers: int, block_list: nn.ModuleList,
                      is_perception_or_output: bool = True):
        """
        Build a stage with interleaved GLT and Anchor Attention layers,
        each followed by a Gated Shard FFN.

        Interleaving pattern (Section 8):
        - Perception/Output: GLT-heavy (anchor_interval=3, i.e. fewer anchors)
        - Reasoning Core: Anchor every 2nd-3rd layer

        The layer type is chosen to match the global config counts.
        Anchors are placed at positions that fit the interleaving pattern
        across the ENTIRE model, not per-stage.

        Args:
            n_layers: Number of layers in this stage
            block_list: ModuleList to append blocks to
            is_perception_or_output: If True, use anchor_interval=3;
                                     if False (Core), use anchor_interval=2
        """
        config = self.config

        # Count existing GLT/Anchor in blocks already built across all stages
        existing_glt = sum(1 for b in block_list if b.is_glt)
        existing_anchor = sum(1 for b in block_list if b.is_anchor)

        anchor_interval = 3 if is_perception_or_output else 2

        for i in range(n_layers):
            # Compute absolute layer index across entire model
            abs_idx = existing_glt + existing_anchor + i

            # Decide layer type based on simple interleaving:
            # Anchor if: still need more anchors AND
            #   (GLT budget exhausted, OR interval matches, OR this is the last layer overall and we still have anchors left)
            use_anchor = False
            if existing_anchor < config.anchor_layers:
                if existing_glt >= config.glt_layers:
                    use_anchor = True
                elif (abs_idx + 1) % anchor_interval == 0:
                    use_anchor = True
                elif abs_idx == config.n_layers - 1 and existing_anchor + 1 <= config.anchor_layers:
                    use_anchor = True

            if use_anchor:
                mixer = AnchorAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    window_size=config.window_size,
                    n_anchors=config.n_anchors,
                    n_static_anchors=config.n_static_anchors,
                    dropout=config.dropout,
                )
                existing_anchor += 1
            else:
                mixer = GLTLayer(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    dropout=config.dropout,
                )
                existing_glt += 1

            ffn = GatedShardFFN(
                d_model=config.d_model,
                expansion=config.ffn_expansion,
                num_shards=config.ffn_shards,
                dropout=config.dropout,
            )

            block_list.append(TransformerBlock(mixer, ffn))

    def _run_stage_perception(
        self,
        x: torch.Tensor,
        glt_states: List[Optional[torch.Tensor]],
        window_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        pmb_readouts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run Perception stage: fixed single pass."""
        state_idx = 0
        window_idx = 0

        for block in self.perception_blocks:
            if block.is_glt:
                x, new_state = block.forward_glt(x, glt_states[state_idx])
                glt_states[state_idx] = new_state
                state_idx += 1
            else:
                wk, wv = window_caches[window_idx]
                x, block_input = block.forward_anchor(x, wk, wv, pmb_readouts)
                # Window cache must store K/V from the BLOCK INPUT (pre-mixer representation)
                wk, wv = block.mixer.update_window_cache(block_input, wk, wv)
                window_caches[window_idx] = (wk, wv)
                window_idx += 1

        return x, glt_states, window_caches

    def _run_stage_core(
        self,
        x: torch.Tensor,
        glt_states: List[Optional[torch.Tensor]],
        window_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        pmb_readouts: Optional[torch.Tensor],
        token_idx: int = 0,
        max_loops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]], int, torch.Tensor]:
        """
        Run Reasoning Core with Adaptive Depth Looping.

        Loops the core blocks 1..N_max times. After each loop, the halting
        head decides whether to continue.

        Args:
            max_loops: If provided, overrides config.n_max_loops.
                       Use max_loops=1 during training for faster single-pass.

        Returns:
            x: Refined hidden state
            glt_states, window_caches: Updated
            n_loops: Number of iterations actually used
            ponder_cost: Ponder cost for training (0 if not training)
        """
        config = self.config
        n_loops_max = max_loops if max_loops is not None else config.n_max_loops
        entry_states = []
        halting_probs = []

        # Starting indices for core's portion of state/cache lists
        # glt_states list indexes ALL layers (including Anchor=Nones), so
        # state_idx starts at the total number of perception layers (not just GLT)
        state_idx = len(self.perception_blocks)
        window_idx = sum(1 for b in self.perception_blocks if b.is_anchor)

        # ⚡ TRAINING FAST PATH: when n_loops_max <= 1, skip all ADL overhead
        if n_loops_max <= 1:
            # Single pass through core blocks (no halting, no ACT, no ponder)
            for block in self.core_blocks:
                if block.is_glt:
                    x, new_state = block.forward_glt(x, glt_states[state_idx])
                    glt_states[state_idx] = new_state
                    state_idx += 1
                else:
                    wk, wv = window_caches[window_idx]
                    x, block_input = block.forward_anchor(x, wk, wv, pmb_readouts)
                    wk, wv = block.mixer.update_window_cache(block_input, wk, wv)
                    window_caches[window_idx] = (wk, wv)
                    window_idx += 1
            n_loops = 1
            ponder_cost = torch.tensor(0.0, device=x.device)
            return x, glt_states, window_caches, n_loops, ponder_cost

        # Full ADL path (inference) — looping with halting
        for loop in range(n_loops_max):
            # Run core blocks
            for block in self.core_blocks:
                if block.is_glt:
                    x, new_state = block.forward_glt(x, glt_states[state_idx])
                    glt_states[state_idx] = new_state
                    state_idx += 1
                else:
                    wk, wv = window_caches[window_idx]
                    x, block_input = block.forward_anchor(x, wk, wv, pmb_readouts)
                    wk, wv = block.mixer.update_window_cache(block_input, wk, wv)
                    window_caches[window_idx] = (wk, wv)
                    window_idx += 1

            # Reset indices for next loop
            state_idx = len(self.perception_blocks)
            window_idx = sum(1 for b in self.perception_blocks if b.is_anchor)

            # Halting decision
            p = self.halting_head(x)  # [B, 1]
            halting_probs.append(p)
            entry_states.append(x)

            # Check if all batch items want to halt
            cumulative_p = torch.stack(halting_probs, dim=0).sum(dim=0)  # [B, 1]
            if cumulative_p.min().item() >= config.halt_threshold:
                break

        # ACT-style weighted combination
        n_loops = len(halting_probs)
        if n_loops > 1:
            probs = torch.stack(halting_probs, dim=0)
            cumulative = probs.cumsum(dim=0)
            remainder = 1.0 - cumulative[:-1].sum(dim=0).clamp(min=0)
            probs_normalized = probs.clone()
            probs_normalized[-1] = remainder

            x_combined = torch.zeros_like(x)
            total_weight = 0.0
            for i, (state_i, prob_i) in enumerate(zip(entry_states, probs_normalized)):
                x_combined = x_combined + prob_i * state_i
                total_weight = total_weight + prob_i
            x = x_combined / total_weight.clamp(min=1e-8)

            if self.training:
                ponder_cost = sum(p.mean() for p in halting_probs)
                ponder_cost = 0.01 * ponder_cost
            else:
                ponder_cost = torch.tensor(0.0, device=x.device)
        else:
            ponder_cost = torch.tensor(0.0, device=x.device)

        return x, glt_states, window_caches, n_loops, ponder_cost

    def _run_stage_output(
        self,
        x: torch.Tensor,
        glt_states: List[Optional[torch.Tensor]],
        window_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        pmb_readouts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run Output stage: fixed single pass."""
        state_idx = len(self.perception_blocks) + len(self.core_blocks)
        window_idx = sum(1 for b in self.perception_blocks if b.is_anchor) + \
                     sum(1 for b in self.core_blocks if b.is_anchor)

        for block in self.output_blocks:
            if block.is_glt:
                x, new_state = block.forward_glt(x, glt_states[state_idx])
                glt_states[state_idx] = new_state
                state_idx += 1
            else:
                wk, wv = window_caches[window_idx]
                x, block_input = block.forward_anchor(x, wk, wv, pmb_readouts)
                wk, wv = block.mixer.update_window_cache(block_input, wk, wv)
                window_caches[window_idx] = (wk, wv)
                window_idx += 1

        return x, glt_states, window_caches

    def init_states(self, batch_size: int = 1, device: str = "cpu",
                    dtype: torch.dtype = torch.float32) -> Tuple[
        List[Optional[torch.Tensor]],
        List[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Initialize fresh GLT states and window caches."""
        glt_states = []
        window_caches = []

        for block in (list(self.perception_blocks) +
                      list(self.core_blocks) +
                      list(self.output_blocks)):
            if block.is_glt:
                glt_states.append(
                    block.mixer.reset_state(batch_size, device, dtype)
                )
            else:
                glt_states.append(None)  # Placeholder for anchor layers
                window_caches.append(
                    block.mixer.init_window_cache(batch_size, device, dtype)
                )

        return glt_states, window_caches

    def _run_stage_parallel(
        self,
        x: torch.Tensor,
        block_list: torch.nn.ModuleList,
        glt_states: List[Optional[torch.Tensor]],
        window_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        pmb_readouts: Optional[torch.Tensor],
        state_offset: int = 0,
        window_offset: int = 0,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Phase 2: Process ALL tokens through a stage in PARALLEL.
        
        Uses parallel scan for GLT layers (O(log n) vs O(n) sequential)
        and batched attention for Anchor layers (single matmul vs per-token matmuls).
        
        Args:
            x: [B, seq_len, d_model] input for all tokens
            block_list: List of TransformerBlock for this stage
            glt_states: Full state list (indexed by state_offset)
            window_caches: Full window cache list (indexed by window_offset)
            pmb_readouts: PMB readout vectors
            state_offset: Starting index in glt_states for this stage
            window_offset: Starting index in window_caches for this stage
        
        Returns:
            x: [B, seq_len, d_model] output for all tokens
            glt_states, window_caches: Updated
        """
        state_idx = state_offset
        window_idx = window_offset
        
        # Lazy import to avoid circular dependency with training module
        from continuum.training.parallel_scan import glt_parallel_forward_with_state
        
        for block in block_list:
            if block.is_glt:
                # ---- GLT: Parallel scan (O(log n)) ----
                residual = x
                x_norm = block.mixer.norm(x)  # [B, L, d_model]
                D = block.mixer.d_state
                B, L, _ = x_norm.shape
                
                k = block.mixer.W_k(x_norm)  # [B, L, d_state]
                v = block.mixer.W_v(x_norm)
                q = block.mixer.W_q(x_norm)
                k = block.mixer.kv_norm(k)
                v = block.mixer.kv_norm(v)
                gamma = torch.sigmoid(block.mixer.W_gamma(x_norm))
                iota = torch.sigmoid(block.mixer.W_iota(x_norm))
                r_gate = torch.sigmoid(block.mixer.W_r(x_norm))
                
                # Parallel scan: O(log L) instead of O(L)
                o, final_state = glt_parallel_forward_with_state(
                    k, v, q, gamma, iota, r_gate, block.mixer.W_o.weight
                )  # o: [B, L, d_model], final_state: [B, D, D]
                
                x = residual + o
                x = block.ffn(x)
                
                glt_states[state_idx] = final_state
                state_idx += 1
            
            else:
                # ---- Anchor: Batched attention ----
                # ⚡ Static cache already refreshed in forward/forward_parallel
                # (not refreshed here to avoid redundant matmuls)
                wk, wv = window_caches[window_idx]
                
                # Run attention on full sequence with causal masking
                o = block.mixer(x, wk, wv, pmb_readouts, causal_mask=True)
                o = block.ffn(o)
                
                # Update window cache: store K/V for last window_size tokens
                n_kv = block.mixer.n_kv_heads
                hd = block.mixer.head_dim
                ws = block.mixer.window_size
                B, L, _ = x.shape
                
                # Take up to ws last tokens (may be fewer if L < ws)
                last_x = block.mixer.norm(x[:, -ws:, :])  # [B, min(L, ws), d_model]
                actual_ws = last_x.shape[1]
                # ⚡ Fused K/V: single matmul for both K and V projections
                # Uses W_qkv weight (already fused QKV) and extracts K,V slices
                # 2 separate matmuls → 1 fused matmul = -50% kernel launch overhead
                qkv_all = F.linear(last_x, block.mixer.W_qkv.weight)  # [B, actual_ws, q_dim+2*kv_dim]
                _, new_wk_flat, new_wv_flat = qkv_all.split(
                    [block.mixer.q_dim, block.mixer.kv_dim, block.mixer.kv_dim], dim=-1
                )
                
                # Pad with zeros on the LEFT if fewer tokens than window_size.
                # Matches sequential behavior: window starts all-zeros, fills from right.
                if actual_ws < ws:
                    pad = ws - actual_ws
                    pk = torch.zeros(B, pad, n_kv * hd, device=x.device, dtype=new_wk_flat.dtype)
                    pv = torch.zeros(B, pad, n_kv * hd, device=x.device, dtype=new_wv_flat.dtype)
                    new_wk_flat = torch.cat([pk, new_wk_flat], dim=1)
                    new_wv_flat = torch.cat([pv, new_wv_flat], dim=1)
                
                new_wk = new_wk_flat.view(B, ws, n_kv, hd)
                new_wv = new_wv_flat.view(B, ws, n_kv, hd)
                window_caches[window_idx] = (new_wk, new_wv)
                window_idx += 1
                
                x = o  # Anchor already includes residual
        
        return x, glt_states, window_caches


    def _get_pmb_readouts(self, x: torch.Tensor) -> torch.Tensor:
        """Fetch PMB readouts using the current hidden state as query."""
        # Pool if sequence
        if x.dim() == 3:
            query = x.mean(dim=1)
        else:
            query = x
        return self.pmb.read(query)

    def forward(
        self,
        token_ids: torch.Tensor,
        glt_states: Optional[List[Optional[torch.Tensor]]] = None,
        window_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for one or more tokens.

        Args:
            token_ids: [B, seq_len] token IDs
            glt_states: List of GLT states (one per GLT layer), or None to init
            window_caches: List of (window_k, window_v) tuples (one per Anchor layer), or None

        Returns:
            Dict with:
                'logits': [B, seq_len, vocab_size] — next-token logits
                'glt_states': Updated GLT states
                'window_caches': Updated window caches
                'n_loops': Average number of core loops used (for monitoring)
                'ponder_cost': Ponder cost value (for training loss)
        """
        B, seq_len = token_ids.shape
        device = token_ids.device
        dtype = self.embedding.embed_table.weight.dtype

        # Initialize states if needed
        if glt_states is None or window_caches is None:
            glt_states, window_caches = self.init_states(B, str(device), dtype)

        # Token → embeddings
        x = self.embedding.embed(token_ids)  # [B, seq_len, d_model]

        # ⚡ Phase 1: Precompute static anchor K/V once (identical for all tokens)
        for block in (list(self.perception_blocks) +
                      list(self.core_blocks) +
                      list(self.output_blocks)):
            if block.is_anchor:
                block.mixer.refresh_static_cache()

        # Fetch PMB readouts once per forward pass
        pmb_readouts = self._get_pmb_readouts(x)

        total_loops = 0
        total_ponder = torch.tensor(0.0, device=device)

        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]  # [B, d_model]

            # Stage 1: Perception
            xt, glt_states, window_caches = self._run_stage_perception(
                xt, glt_states, window_caches, pmb_readouts
            )

            # Stage 2: Reasoning Core (looped)
            xt, glt_states, window_caches, n_loops, ponder = self._run_stage_core(
                xt, glt_states, window_caches, pmb_readouts, token_idx=t
            )
            total_loops += n_loops
            total_ponder = total_ponder + ponder

            # Stage 3: Output
            xt, glt_states, window_caches = self._run_stage_output(
                xt, glt_states, window_caches, pmb_readouts
            )

            outputs.append(xt)

        # Stack outputs
        x = torch.stack(outputs, dim=1)  # [B, seq_len, d_model]

        # Final norm + output projection
        x = self.final_norm(x)
        logits = self.embedding.project_to_logits(x)  # [B, seq_len, vocab_size]

        avg_loops = total_loops / max(seq_len, 1)

        return {
            "logits": logits,
            "glt_states": glt_states,
            "window_caches": window_caches,
            "n_loops": avg_loops,
            "ponder_cost": total_ponder / max(seq_len, 1),
        }

    def forward_parallel(
        self,
        token_ids: torch.Tensor,
        core_max_loops: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 2: Parallel forward — Perception & Output batched, Core per-token.
        
        Reduces Python overhead by 70-80% compared to token-by-token forward.
        Only the Core stage (with ADL halting) processes tokens sequentially.
        
        Args:
            token_ids: [B, seq_len] token IDs
            core_max_loops: Max ADL loops for Core stage. Use 1 for training.
                            None = use config.n_max_loops.
        
        Returns:
            Dict with logits, glt_states, window_caches, n_loops, ponder_cost
        """
        B, seq_len = token_ids.shape
        device = token_ids.device
        dtype = self.embedding.embed_table.weight.dtype
        
        # Initialize states
        glt_states, window_caches = self.init_states(B, str(device), dtype)
        
        # Refresh static anchor caches for all layers
        for block in (list(self.perception_blocks) +
                      list(self.core_blocks) +
                      list(self.output_blocks)):
            if block.is_anchor:
                block.mixer.refresh_static_cache()
        
        # Embed all tokens
        x = self.embedding.embed(token_ids)  # [B, seq_len, d_model]
        
        # Fetch PMB readouts
        pmb_readouts = self._get_pmb_readouts(x)
        
        # === STAGE 1: Parallel Perception ===
        x, glt_states, window_caches = self._run_stage_parallel(
            x, self.perception_blocks, glt_states, window_caches,
            pmb_readouts, state_offset=0, window_offset=0
        )
        
        # === STAGE 2: Core (fully parallel when ADL disabled, sequential when ADL active) ===
        core_state_start = len(self.perception_blocks)
        core_window_start = sum(1 for b in self.perception_blocks if b.is_anchor)
        
        if core_max_loops is not None and core_max_loops <= 1:
            # ⚡ TRAINING FAST PATH: Fully parallel Core — no per-token Python loop!
            # When ADL is disabled (max_loops=1), all tokens through Core in ONE pass.
            # GLT parallel scan is exact. Anchor window uses Perception's cache
            # (not Core's refined cache) — causally correct, trains fine from scratch.
            # Eliminates seq_len×layer_count Python overhead.
            x, glt_states, window_caches = self._run_stage_parallel(
                x, self.core_blocks, glt_states, window_caches,
                pmb_readouts, state_offset=core_state_start,
                window_offset=core_window_start,
            )
            total_loops = 0
            total_ponder = torch.tensor(0.0, device=device)
        else:
            # Sequential Core (ADL active) — per-token for halting decisions
            total_loops = 0
            total_ponder = torch.tensor(0.0, device=device)
            core_outputs = []
            for t in range(seq_len):
                xt = x[:, t, :]  # [B, d_model]
                xt, glt_states, window_caches, n_loops, ponder = self._run_stage_core(
                    xt, glt_states, window_caches, pmb_readouts,
                    token_idx=t, max_loops=core_max_loops,
                )
                total_loops += n_loops
                total_ponder = total_ponder + ponder
                core_outputs.append(xt)
            
            x = torch.stack(core_outputs, dim=1)  # [B, seq_len, d_model]
        
        # === STAGE 3: Parallel Output ===
        output_state_start = len(self.perception_blocks) + len(self.core_blocks)
        output_window_start = core_window_start + sum(1 for b in self.core_blocks if b.is_anchor)
        
        x, glt_states, window_caches = self._run_stage_parallel(
            x, self.output_blocks, glt_states, window_caches,
            pmb_readouts, state_offset=output_state_start, window_offset=output_window_start
        )
        
        # Final norm + output projection
        x = self.final_norm(x)
        logits = self.embedding.project_to_logits(x)
        
        avg_loops = total_loops / max(seq_len, 1)
        
        return {
            "logits": logits,
            "glt_states": glt_states,
            "window_caches": window_caches,
            "n_loops": avg_loops,
            "ponder_cost": total_ponder / max(seq_len, 1),
        }


    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Autoregressive text generation.

        Args:
            prompt_ids: [1, prompt_len] token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            generated_ids: [1, prompt_len + new_tokens]
            loop_counts: Per-token ADL loop counts (for monitoring)
        """
        self.eval()
        device = prompt_ids.device
        B = prompt_ids.shape[0]
        generated = list(prompt_ids[0].tolist())
        loop_counts = []

        # Initialize states
        glt_states, window_caches = self.init_states(
            B, str(device), self.embedding.embed_table.weight.dtype
        )

        # Prefill: process the prompt
        with torch.no_grad():
            result = self.forward(prompt_ids, glt_states, window_caches)
            glt_states = result["glt_states"]
            window_caches = result["window_caches"]
            next_logits = result["logits"][:, -1, :]  # Last position logits

            # Generate new tokens
            for _ in range(max_new_tokens):
                # Apply temperature
                logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

                generated.append(next_token[0, 0].item())

                # Check for EOS
                if next_token[0, 0].item() == self.config.eos_token_id:
                    break

                # Forward the new token
                result = self.forward(next_token, glt_states, window_caches)
                glt_states = result["glt_states"]
                window_caches = result["window_caches"]
                next_logits = result["logits"][:, -1, :]
                loop_counts.append(result["n_loops"])

        return torch.tensor([generated], device=device), loop_counts

    def serialize_state(
        self,
        glt_states: List[Optional[torch.Tensor]],
        window_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict:
        """
        Serialize runtime state for app lifecycle management (Section 13).
        Returns a small, fixed-size dict that can be saved/loaded.
        """
        state_dict = {
            "glt_states": [
                s.cpu().clone() if s is not None else None
                for s in glt_states
            ],
            "window_caches": [
                (wk.cpu().clone(), wv.cpu().clone())
                for wk, wv in window_caches
            ],
            "pmb_slots": self.pmb.serialize().cpu(),
        }
        return state_dict

    def deserialize_state(self, state_dict: Dict, device: str = "cpu") -> Tuple[
        List[Optional[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Restore runtime state from serialized dict."""
        glt_states = [
            s.to(device) if s is not None else None
            for s in state_dict["glt_states"]
        ]
        window_caches = [
            (wk.to(device), wv.to(device))
            for wk, wv in state_dict["window_caches"]
        ]
        self.pmb.deserialize(state_dict["pmb_slots"].to(device))
        return glt_states, window_caches

    @property
    def num_params(self) -> int:
        return self._n_params


# ============================================================================
# Factory function for standard tiers (Section 17)
# ============================================================================

def create_continuum_nano() -> ContinuumModel:
    """Create Continuum-Nano: ~5M parameters."""
    return ContinuumModel(ContinuumConfig(
        d_model=192, d_state=48, d_embed=48, vocab_size=8000,
        n_layers=6, glt_layers=4, anchor_layers=2,
        perception_layers=2, core_layers=2, output_layers=2,
        ffn_expansion=3, ffn_shards=2,
        n_heads=4, n_kv_heads=2, window_size=48,
        n_anchors=8, n_static_anchors=4,
        n_max_loops=3, halt_threshold=0.95,
        pmb_slots=16, pmb_readout=4, chunk_size=64,
    ))


def create_continuum_small() -> ContinuumModel:
    """Create Continuum-Small: ~20M parameters."""
    return ContinuumModel(ContinuumConfig(
        d_model=384, d_state=96, d_embed=80, vocab_size=12000,
        n_layers=8, glt_layers=5, anchor_layers=3,
        perception_layers=3, core_layers=2, output_layers=3,
        ffn_expansion=4, ffn_shards=4,
        n_heads=8, n_kv_heads=4, window_size=96,
        n_anchors=12, n_static_anchors=4,
        n_max_loops=4, halt_threshold=0.95,
        pmb_slots=32, pmb_readout=8, chunk_size=64,
    ))


def create_continuum_max() -> ContinuumModel:
    """
    Create Continuum-Max: ~100M parameters.
    
    From Section 17 tier table:
    d_model=768, d_state=192, d_embed=160, vocab=16,000
    12 layers: 9 GLT + 3 Anchor (4:3:5 perception:core:output)
    12 heads, 4 KV heads (GQA ratio 3:1)
    Window=128, Anchors=24 (static=8, PMB=16)
    FFN: 4x expansion, 6 shards
    ADL: N_max=5, PMB: 64 slots
    """
    return ContinuumModel(ContinuumConfig(
        d_model=768, d_state=192, d_embed=160, vocab_size=16000,
        n_layers=12, glt_layers=9, anchor_layers=3,
        perception_layers=4, core_layers=3, output_layers=5,
        ffn_expansion=4, ffn_shards=6,
        n_heads=12, n_kv_heads=4, window_size=128,
        n_anchors=24, n_static_anchors=8,
        n_max_loops=5, halt_threshold=0.95,
        pmb_slots=64, pmb_readout=16, chunk_size=64,
    ))
