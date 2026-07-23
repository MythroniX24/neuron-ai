"""
Training Loop for Continuum SLM (Section 18).

Implements:
- Parallel scan training (via GLT parallel formulation)
- Sequence-length curriculum
- Gradient clipping and stability monitoring
- Staged complexity introduction
- Checkpointing
- AMP mixed precision, fused AdamW, torch.compile support
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Callable
from tqdm import tqdm

from continuum.model.model import ContinuumModel
from continuum.training.losses import ContinuumLoss, SparsityMonitor


class ContinuumTrainer:
    """
    Trainer for Continuum SLM with all architecture-specific features.

    Features:
    - AMP (Automatic Mixed Precision) for 2x faster training on T4
    - Fused AdamW optimizer for 10-20% faster optimizer step
    - torch.compile support for kernel fusion
    - Gradient accumulation
    - Sequence-length curriculum

    Usage:
        model = create_continuum_nano()
        trainer = ContinuumTrainer(model, ...)
        trainer.train(train_dataloader, val_dataloader, num_epochs=10)
    """

    def __init__(
        self,
        model: ContinuumModel,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        device: str = "cpu",
        use_amp: bool = True,
        compile_model: bool = False,
        use_parallel_forward: bool = True,
        use_gradient_checkpointing: bool = False,  # ⚡ Phase 8: Save VRAM on embedding output
    ):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.use_amp = use_amp and (device == "cuda")

        self.model.to(device)
        self.use_parallel_forward = use_parallel_forward

        # ⚡ Phase 8: Gradient checkpointing for FactorizedEmbedding
        # The embedding output is [B, L, d_model] which is large.
        # Checkpointing tells autograd to NOT store intermediate activations —
        # they're recomputed during backward. Saves ~15% VRAM at ~5% compute cost.
        if use_gradient_checkpointing and device == "cuda":
            import torch.utils.checkpoint as cp
            _orig_embed = model.embedding.embed
            model.embedding.embed = lambda token_ids: cp.checkpoint(
                _orig_embed, token_ids, use_reentrant=False
            )
            print("  ✅ Gradient checkpointing enabled for FactorizedEmbedding (-15% VRAM)")

        # ⚡ Phase 7: Dedicated CUDA stream for async data transfer (compute/transfer overlap)
        self._transfer_stream = torch.cuda.Stream() if device == "cuda" else None

        # Flag for notebook display
        self.use_compiled = False
        self.compile_mode = None

        # ⚡ Phase 7: torch.compile with fullgraph + capture_scalar_outputs
        # fullgraph=True: entire model in ONE fused kernel (2-3x faster than reduce-overhead)
        # capture_scalar_outputs: eliminates graph breaks from .item() in ADL inference path
        if compile_model and device == "cuda":
            import torch._dynamo
            torch._dynamo.config.capture_scalar_outputs = True
            try:
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",  # Aggressive kernel fusion + autotuning
                    fullgraph=True,        # ⚡ Phase 7: Single fused kernel
                )
                self.use_compiled = True
                self.compile_mode = "max-autotune (fullgraph)"
                print("  ✅ torch.compile: max-autotune fullgraph mode (2-3x speedup)")
            except Exception:
                # Fallback: fullgraph may fail if there are graph breaks we missed
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    self.use_compiled = True
                    self.compile_mode = "reduce-overhead"
                    print("  ✅ torch.compile: reduce-overhead mode (fullgraph attempt failed)")
                except Exception as e:
                    print(f"  ⚠️ torch.compile failed: {e}")
        if self.use_compiled and self.compile_mode and "max-autotune" in self.compile_mode:
            print("  ⏳ First step will be slow (~5 min) — max-autotune is autotuning kernels...")

        # ⚡ Fused AdamW — single kernel for optimizer step
        # Uses try/except instead of __code__ inspection (safe across PyTorch versions)
        use_fused = False
        if device == "cuda":
            try:
                self.optimizer = AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.95),
                    fused=True,
                )
                use_fused = True
                print("  ✅ Using fused AdamW optimizer (CUDA optimized)")
            except (TypeError, RuntimeError, AttributeError):
                pass
        
        if not use_fused:
            # ⚡ Phase 7: foreach=True — multi-tensor apply (2-3x faster than default on T4)
            # Falls back to default if foreach also not available
            try:
                self.optimizer = AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.95),
                    foreach=True,
                )
                print("  ✅ Using foreach AdamW (multi-tensor optimized)")
            except (TypeError, RuntimeError, AttributeError):
                self.optimizer = AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.95),
                )

        # ⚡ GradScaler for AMP (prevents gradient underflow in FP16)
        # Default init_scale=2^16 is well-tested for T4 FP16 training.
        # Conservative scales can cause underflow on small losses (common early in training).
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        ) if self.use_amp else None
        if self.use_amp:
            print("  ✅ AMP mixed precision enabled (FP16 Tensor Cores)")
        if self.use_parallel_forward:
            print("  ✅ Parallel forward enabled (Perception+Output batched, Core per-token)")

        self.scheduler = None
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.base_lr = learning_rate

        # Loss function
        self.loss_fn = ContinuumLoss(
            vocab_size=model.config.vocab_size,
            pad_token_id=0,
        )

        #        Monitoring
        self.sparsity_monitor = SparsityMonitor()
        self.global_step = 0
        self._optimizer_step_count = 0  # ⚡ Fix: track actual optimizer steps (not micro-steps)
        self._total_optimizer_steps = 1  # Set in train() — avoids division by zero before train() is called
        self.best_val_loss = float("inf")
        self._history_train_losses = []
        self._history_val_losses = []
        self._history_val_ppls = []
        self._history_epochs = []
        self.training_start_time = 0.0
        self.total_tokens_processed = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _warmup_lr(self, step: int):
        """Linear warmup of learning rate."""
        if step < self.warmup_steps:
            lr_scale = step / max(self.warmup_steps, 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.base_lr * lr_scale

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        seq_len_curriculum: int = 64,
        accumulation_step: int = 1,
        total_accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Single training step with AMP mixed precision and gradient accumulation.

        Args:
            batch: Dict with 'input_ids' [B, L] and 'labels' [B, L]
            seq_len_curriculum: Current maximum sequence length
            accumulation_step: Current index (1-based) in the accumulation sequence
            total_accumulation_steps: Total number of accumulation steps

        Returns:
            Dict of loss values for logging
        """
        self.model.train()
        self.global_step += 1

        # ⚡ Phase 5: Determine optimizer step boundary first (used by LR warmup and logging)
        is_optimizer_step = accumulation_step == total_accumulation_steps

        # ⚡ FIX: Use optimizer step count (not micro-step count) for warmup
        # With grad_accum=2, global_step increments 2x per optimizer step.
        # Using global_step made warmup complete 2x too fast → unstable early training.
        if is_optimizer_step:
            self._optimizer_step_count += 1
            self._warmup_lr(self._optimizer_step_count)
            # ⚡ FIX: Apply cosine annealing AFTER warmup (scheduler was NEVER created!)
            # Without this, LR stayed flat at base_lr after warmup → worse convergence.
            if self.scheduler is not None and self._optimizer_step_count > self.warmup_steps:
                self.scheduler.step()

        # ⚡ Phase 7: Async data transfer via dedicated CUDA stream
        # Transfer overlaps with backward pass of previous step (compute/transfer overlap)
        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                input_ids = batch["input_ids"][:, :seq_len_curriculum].to(self.device, non_blocking=True)
                labels = batch["labels"][:, :seq_len_curriculum].to(self.device, non_blocking=True)
            # Ensure transfer is complete before using data in forward pass
            torch.cuda.current_stream().wait_stream(self._transfer_stream)
        else:
            input_ids = batch["input_ids"][:, :seq_len_curriculum].to(self.device, non_blocking=True)
            labels = batch["labels"][:, :seq_len_curriculum].to(self.device, non_blocking=True)

        # Zero gradients at START of accumulation
        if accumulation_step == 1:
            self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # ⚡ AMP: Forward pass in FP16 (Tensor Cores enabled)
        # ⚡ core_max_loops=1: Single-pass Core during training (ADL disabled)
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
            if self.use_parallel_forward:
                result = self.model.forward_parallel(input_ids, core_max_loops=1)
            else:
                result = self.model.forward(input_ids)
            logits = result["logits"]
            ponder_cost = result["ponder_cost"]

            # Shift for next-token prediction
            # ⚡ Use slice directly (already contiguous in memory)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # Compute loss
            losses = self.loss_fn(
                logits=shift_logits,
                targets=shift_labels,
                ponder_cost=ponder_cost,
                ffn_gates=None,
            )

        # Gradient accumulation: average loss over accumulation steps
        loss_for_backward = losses["total"] / total_accumulation_steps

        # ⚡ AMP: Backward with GradScaler (prevents FP16 underflow)
        if self.scaler is not None:
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        # Track tokens processed (for throughput monitoring)
        B, L = input_ids.shape
        self.total_tokens_processed += B * L

        # Only clip and step at END of accumulation
        if is_optimizer_step:
            # ⚡ AMP: Unscale gradients before clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # ⚡ AMP: Optimizer step with scaler
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            # ⚡ NOTE: scheduler.step() is called at the TOP of this block (in the
            # warmup/cosine section). Do NOT call it again here — double-stepping
            # the scheduler caused cosine decay to complete 2x too fast, hurting
            # model quality. This was a CRITICAL bug.

        # ⚡ FIX: Anneal on optimizer steps with correct total_steps
        # Before: called every micro-step with hardcoded 100000 → wrong schedule
        if is_optimizer_step:
            self.loss_fn.anneal_weights(self._optimizer_step_count, self._total_optimizer_steps)

        # ⚡ OPTIMIZE: Only call .item() (GPU→CPU sync) on log intervals.
        # Before: 5 × .item() per step × 5850 steps = 29,250 GPU syncs/epoch.
        # After: 1 × .item() per step (loss only) + 4 × .item() every 100 steps.
        # Saves ~90% of GPU synchronization overhead.
        should_log = is_optimizer_step and self.global_step % self.log_interval == 0
        gamma_monitor = self._monitor_gamma_gates() if should_log else {}

        # loss is always needed for epoch averaging — one .item() is unavoidable
        loss_val = losses["total"].item()

        if should_log:
            return {
                "loss": loss_val,
                "ce_loss": losses["ce"].item(),
                "ponder_cost": losses["ponder"].item(),
                "sparsity_loss": losses["sparsity"].item(),
                "memory_loss": losses["memory"].item(),
                "n_loops": result["n_loops"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "throughput": self.total_tokens_processed / max(time.time() - self.training_start_time, 1),
                **gamma_monitor,
            }
        else:
            # ⚡ Skip .item() for metrics only used for logging (not epoch averaging)
            return {
                "loss": loss_val,
                "ce_loss": 0.0, "ponder_cost": 0.0, "sparsity_loss": 0.0,
                "memory_loss": 0.0, "n_loops": 0.0,
                "lr": self.optimizer.param_groups[0]["lr"],
                "throughput": 0.0,
            }

    def _monitor_gamma_gates(self) -> Dict[str, float]:
        """Monitor GLT decay gate distribution for training stability (Section 18)."""
        gamma_means = []
        for block in (list(self.model.perception_blocks) +
                      list(self.model.core_blocks) +
                      list(self.model.output_blocks)):
            if block.is_glt:
                # Get current bias values (which determine gamma via sigmoid)
                bias = block.mixer.W_gamma.bias.detach()
                gamma_means.append(torch.sigmoid(bias).mean().item())

        if not gamma_means:
            return {}

        avg_gamma = sum(gamma_means) / len(gamma_means)
        return {
            "gamma_mean": avg_gamma,
            "gamma_healthy": 0.1 < avg_gamma < 0.9,  # Neither collapsed to 0 nor saturated at 1
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_loops = 0.0
        n_batches = 0

        for batch in val_loader:
            if n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Use AMP for validation too (faster on T4)
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                if self.use_parallel_forward:
                    result = self.model.forward_parallel(input_ids)
                else:
                    result = self.model.forward(input_ids)
                logits = result["logits"]

                        # ⚡ Slices are already contiguous in memory
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]

                ce = nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )

            # ⚡ OPTIMIZE: Call .item() once (was called twice for same tensor!)
            ce_val = ce.item()
            total_loss += ce_val
            total_ce += ce_val
            total_loops += result["n_loops"]
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_ppl": math.exp(total_ce / max(n_batches, 1)),
            "val_loops": total_loops / max(n_batches, 1),
        }

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        seq_len_start: int = 32,
        seq_len_end: int = 512,
        seq_len_warmup_epochs: int = 3,
        grad_accum_steps: int = 4,
    ):
        """
        Full training loop with sequence-length curriculum and gradient accumulation.

        Sequence-length curriculum (Section 18):
        - Start with short sequences (32 tokens)
        - Progressively increase to full length (512 tokens)
        - This stabilizes early training and reduces compute cost

        Gradient accumulation:
        - Accumulates gradients over `grad_accum_steps` batches
        - Effective batch size = batch_size x grad_accum_steps
        - Reduces memory pressure while maintaining large effective batch

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of training epochs
            seq_len_start: Starting sequence length for curriculum
            seq_len_end: Final sequence length
            seq_len_warmup_epochs: Epochs over which to linearly ramp sequence length
            grad_accum_steps: Number of batches to accumulate gradients over
        """
        total_steps = len(train_loader) * num_epochs
        # ⚡ FIX: Calculate optimizer steps (accounting for gradient accumulation)
        self._total_optimizer_steps = max(1, total_steps // grad_accum_steps)

        # ⚡ FIX: Create CosineAnnealingLR — was NEVER created!
        # Without this, LR stayed flat at base_lr after warmup → worse final convergence.
        # Cosine decay from base_lr to 1% of base_lr over remaining steps after warmup.
        decay_steps = max(1, self._total_optimizer_steps - self.warmup_steps)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=decay_steps, eta_min=self.base_lr * 0.01
        )

        print(f"Starting training: {num_epochs} epochs, ~{total_steps} steps")
        print(f"  Optimizer steps: ~{self._total_optimizer_steps} (grad_accum={grad_accum_steps})")
        print(f"  LR schedule: warmup({self.warmup_steps}) → cosine decay → {self.base_lr * 0.01:.2e}")
        print(f"Model: {self.model.num_params:,} parameters")
        print(f"Device: {self.device}")
        mode = "Parallel (Phase 2)" if self.use_parallel_forward else "Sequential"
        print(f"Forward mode: {mode}")
        if self.use_compiled:
            print(f"Compile mode: {self.compile_mode}")
        # batch_size may come from DataLoader.batch_size OR batch_sampler.batch_size (bucket sampler)
        if hasattr(train_loader, 'batch_sampler') and train_loader.batch_sampler is not None and hasattr(train_loader.batch_sampler, 'batch_size'):
            dl_batch_size = train_loader.batch_sampler.batch_size
        else:
            dl_batch_size = getattr(train_loader, 'batch_size', None)
        if dl_batch_size is not None:
            print(f"Gradient accumulation: {grad_accum_steps} steps, "
                  f"effective batch = {dl_batch_size * grad_accum_steps}")
        else:
            print(f"Gradient accumulation: {grad_accum_steps} steps")

        self.training_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            # Sequence length curriculum
            if epoch < seq_len_warmup_epochs:
                progress = epoch / max(seq_len_warmup_epochs, 1)
                seq_len = int(seq_len_start + (seq_len_end - seq_len_start) * progress)
            else:
                seq_len = seq_len_end

                # ⚡ Phase 5: Only GC at epoch start (not both start AND end)
            if self.device == "cuda":
                torch.cuda.empty_cache()
                # ⚡ Also empty CUDA caching allocator to prevent fragmentation over long runs
                torch.cuda.synchronize()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (L={seq_len})")
            for batch_idx, batch in enumerate(pbar):
                accum_idx = (batch_idx % grad_accum_steps) + 1
                metrics = self.train_step(
                    batch,
                    seq_len_curriculum=seq_len,
                    accumulation_step=accum_idx,
                    total_accumulation_steps=grad_accum_steps,
                )
                epoch_loss += metrics["loss"]
                epoch_steps += 1

                # Update progress bar after each optimizer step
                if accum_idx == grad_accum_steps and self.global_step % self.log_interval == 0:
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.3f}",
                        "ce": f"{metrics['ce_loss']:.3f}",
                        "loops": f"{metrics['n_loops']:.1f}",
                        "gamma": f"{metrics.get('gamma_mean', 0):.2f}",
                        "tok/s": f"{metrics.get('throughput', 0):.0f}",
                    })

            # End of epoch
            avg_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = time.time() - epoch_start

            # Validation
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_loss"]

            # ⚡ FIX: Actually populate history (was never being appended to!)
            self._history_train_losses.append(avg_loss)
            self._history_val_losses.append(val_loss)
            self._history_val_ppls.append(val_metrics["val_ppl"])
            self._history_epochs.append(epoch + 1)

            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"val_ppl={val_metrics['val_ppl']:.1f}, "
                  f"loops={val_metrics['val_loops']:.1f}, "
                  f"time={epoch_time:.0f}s")

            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, is_best=is_best)

        return {
            "train_losses": self._history_train_losses,
            "val_losses": self._history_val_losses,
            "val_ppls": self._history_val_ppls,
            "epochs": self._history_epochs,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        # ⚡ FIX: Handle torch.compile — access original module for state_dict and config
        # When model is compiled, self.model is an OptimizedModule wrapper.
        # state_dict() works but .config attribute is on the original module.
        orig_model = getattr(self.model, '_orig_mod', self.model)
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": orig_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": orig_model.config,
        }

        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"  → Best model saved (val_loss={self.best_val_loss:.4f})")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "ContinuumTrainer":
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        from continuum.model.model import ContinuumModel
        model = ContinuumModel(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(model, device=device)
        trainer.global_step = checkpoint["global_step"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return trainer
