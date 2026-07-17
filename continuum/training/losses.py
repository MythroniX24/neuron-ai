"""
Specialized Training Losses for Continuum SLM (Section 18).

Four loss components:
1. Next-token cross-entropy (primary)
2. ADL ponder cost (annealed) — penalizes excessive looping
3. Gated Shard FFN sparsity regularizer — pushes gates toward 0 or 1
4. Memory-only prediction auxiliary loss — forces PMB to be genuinely useful
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ContinuumLoss(nn.Module):
    """
    Combined training loss for Continuum.

    total_loss = ce_loss
               + ponder_weight * ponder_cost
               + sparsity_weight * sparsity_loss
               + memory_weight * memory_aux_loss

    Weights are annealed during training according to the architecture's
    staged curriculum (Section 18).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        ponder_weight: float = 0.01,
        sparsity_weight: float = 0.001,
        memory_weight: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # Current loss weights (may be annealed during training)
        self.ponder_weight = ponder_weight
        self.sparsity_weight = sparsity_weight
        self.memory_weight = memory_weight

        # Base cross-entropy
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)  # -100 = PyTorch standard for ignored tokens
        self._pad_token_id = pad_token_id

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ponder_cost: torch.Tensor,
        ffn_gates: Optional[torch.Tensor] = None,
        memory_logits: Optional[torch.Tensor] = None,
        memory_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: [B, seq_len, vocab_size] — model predictions
            targets: [B, seq_len] — ground truth token IDs
            ponder_cost: scalar — ADL ponder cost
            ffn_gates: [B, seq_len, K, n_layers] — per-shard gate values
            memory_logits: [B, seq_len, vocab_size] — PMB-only predictions
            memory_targets: [B, seq_len] — targets for memory aux loss

        Returns:
            Dict with 'total', 'ce', 'ponder', 'sparsity', 'memory' losses
        """
        B, L, V = logits.shape

        # 1. Next-token cross-entropy (primary signal)
        ce = self.ce_loss(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
        )

        # 2. ADL ponder cost (already scaled in model output)
        ponder = ponder_cost

        # 3. Gated Shard FFN sparsity regularizer
        # Push gate values toward 0 or 1: minimize g*(1-g)
        sparsity = torch.tensor(0.0, device=logits.device)
        if ffn_gates is not None:
            # ffn_gates: [B, L, K, n_layers] in (0, 1)
            # Sparse loss: g * (1-g) averaged over all gates
            sparsity = (ffn_gates * (1 - ffn_gates)).mean()

        # 4. Memory-only prediction auxiliary loss
        memory = torch.tensor(0.0, device=logits.device)
        if memory_logits is not None and memory_targets is not None:
            memory = self.ce_loss(
                memory_logits.reshape(B * L, V),
                memory_targets.reshape(B * L),
            )

        # Combine
        total = (
            ce
            + self.ponder_weight * ponder
            + self.sparsity_weight * sparsity
            + self.memory_weight * memory
        )

        return {
            "total": total,
            "ce": ce,
            "ponder": ponder,
            "sparsity": sparsity,
            "memory": memory,
        }

    def anneal_weights(self, step: int, total_steps: int):
        """
        Anneal loss weights according to training curriculum.

        - Ponder cost: starts near 0, ramps up to full weight
        - Sparsity: constant
        - Memory: starts at 0, ramps up after model has basic language
        """
        progress = min(step / max(total_steps, 1), 1.0)

        # Ponder cost: ramp from 0 to full over first 30% of training
        ponder_ramp = min(progress / 0.3, 1.0) if progress < 0.3 else 1.0
        self.ponder_weight = 0.01 * ponder_ramp

        # Memory aux: ramp from 0 to full over 20-50% of training
        if progress < 0.2:
            self.memory_weight = 0.0
        elif progress < 0.5:
            memory_ramp = (progress - 0.2) / 0.3
            self.memory_weight = 0.1 * memory_ramp
        else:
            self.memory_weight = 0.1


class SparsityMonitor:
    """Track FFN gate sparsity during training for monitoring (Section 18)."""

    def __init__(self):
        self.gate_values = []

    def update(self, gates: torch.Tensor):
        """Record gate values from a forward pass."""
        self.gate_values.append(gates.detach().mean().item())

    def get_stats(self) -> Dict[str, float]:
        """Return sparsity statistics."""
        if not self.gate_values:
            return {"mean_gate": 0.0, "sparsity_ratio": 0.0}

        values = torch.tensor(self.gate_values)
        # Fraction of gates that are close to 0 or 1 (within 0.1)
        sparsity_ratio = ((values < 0.1) | (values > 0.9)).float().mean().item()
        return {
            "mean_gate": values.mean().item(),
            "sparsity_ratio": sparsity_ratio,
        }

    def reset(self):
        self.gate_values = []
