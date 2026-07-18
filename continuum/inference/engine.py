"""
Inference Engine for Continuum SLM (Section 19).

Features:
- INT8 weight quantization (memory-bandwidth-bound CPU optimization)
- Streaming single-token decode
- State serialization for app lifecycle management
- Dual-mode execution: parallel prefill + sequential decode
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from continuum.model.model import ContinuumModel, ContinuumConfig


class QuantizedLinear(nn.Module):
    """
    INT8 weight-only quantized linear layer.

    Stores weights as INT8 with per-channel scales.
    On forward, dequantizes to float for computation.
    This reduces memory bandwidth (bytes moved from RAM) by ~4x,
    which is the primary bottleneck for CPU inference (Section 1.1).
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias.data.clone() if linear.bias is not None else None

        # Quantize weights to INT8
        weight = linear.weight.data
        self.scale = weight.abs().max(dim=1, keepdim=True)[0] / 127.0
        self.scale = self.scale.clamp(min=1e-8)
        self.weight_int8 = torch.round(weight / self.scale).clamp(-128, 127).to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp = self.weight_int8.float() * self.scale.float()
        return nn.functional.linear(x, weight_fp, self.bias)


class ContinuumInference:
    """
    Inference runtime for Continuum SLM.

    Handles:
    - Model loading with optional quantization
    - Streaming token generation
    - State persistence across app sessions
    - Chat conversation management
    """

    def __init__(
        self,
        model: ContinuumModel,
        tokenizer=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        quantize: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.quantize = quantize

        self.model.to(device)
        self.model.eval()

        if quantize and device == "cpu":
            self._apply_quantization()
            print(f"Quantized model: {self._estimate_model_size():.1f} MB")

        # Conversation state
        self.glt_states = None
        self.window_caches = None
        self.conversation_tokens = []

    def _apply_quantization(self):
        """Apply INT8 quantization to all linear layers (Section 19)."""
        def _quantize_module(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear) and child.in_features > 64:
                    # Only quantize larger linear layers (skip tiny gates)
                    setattr(module, name, QuantizedLinear(child))
                else:
                    _quantize_module(child)

        _quantize_module(self.model)

    def _estimate_model_size(self) -> float:
        """Estimate model size in MB with current dtype/quantization."""
        total_bytes = 0
        for p in self.model.parameters():
            if hasattr(p, 'weight_int8'):
                total_bytes += p.weight_int8.numel()  # INT8 = 1 byte
                total_bytes += p.scale.numel() * 4    # float32 scale
            else:
                total_bytes += p.numel() * (2 if self.dtype == torch.float16 else 4)
        return total_bytes / (1024 * 1024)

    def start_conversation(self) -> str:
        """Initialize a fresh conversation."""
        B = 1
        self.glt_states, self.window_caches = self.model.init_states(
            B, self.device, self.dtype
        )
        self.conversation_tokens = []
        self.model.pmb.reset()
        return "Conversation started. All states reset."

    def resume_conversation(self, state_path: str) -> str:
        """Resume conversation from saved state."""
        with open(state_path, "rb") as f:
            state_dict = torch.load(f, map_location=self.device, weights_only=False)

        self.glt_states, self.window_caches = self.model.deserialize_state(
            state_dict, self.device
        )
        self.conversation_tokens = state_dict.get("conversation_tokens", [])
        return f"Resumed conversation ({len(self.conversation_tokens)} tokens)."

    def save_conversation(self, state_path: Optional[str] = None) -> str:
        """Save current conversation state.
        
        Args:
            state_path: If provided, save to file. If None, just return info string.
        
        Returns:
            Info string about saved state
        """
        if self.glt_states is None:
            return "No active conversation to save."

        state_dict = self.model.serialize_state(self.glt_states, self.window_caches)
        state_dict["conversation_tokens"] = self.conversation_tokens

        if state_path:
            with open(state_path, "wb") as f:
                torch.save(state_dict, f)
            size_kb = os.path.getsize(state_path) / 1024
            return f"Saved ({len(self.conversation_tokens)} tokens, {size_kb:.0f} KB)."
        else:
            return f"Active state: {len(self.conversation_tokens)} tokens"

    def get_state_dict(self) -> Optional[Dict]:
        """Get the current state dict without saving to file.
        Used by ConversationManager for flexible state management."""
        if self.glt_states is None:
            return None
        state_dict = self.model.serialize_state(self.glt_states, self.window_caches)
        state_dict["conversation_tokens"] = self.conversation_tokens
        return state_dict
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from a dict (not from file)."""
        self.glt_states, self.window_caches = self.model.deserialize_state(
            state_dict, self.device
        )
        self.conversation_tokens = state_dict.get("conversation_tokens", [])

    @torch.no_grad()
    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> str:
        """
        Generate full response text (non-streaming).
        
        Returns the complete generated string.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")

        prompt_ids = self.tokenizer.encode_with_special(
            prompt, add_bos=True, add_eos=False
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        self.conversation_tokens.extend(prompt_ids)

        if self.glt_states is None:
            B = 1
            self.glt_states, self.window_caches = self.model.init_states(
                B, self.device, self.dtype
            )

        result = self.model.forward(prompt_tensor, self.glt_states, self.window_caches)
        self.glt_states = result["glt_states"]
        self.window_caches = result["window_caches"]
        next_logits = result["logits"][:, -1, :]

        generated_tokens = []
        for i in range(max_new_tokens):
            logits = next_logits.clone()
            
            # 🛡️ Sanitize logits before any processing
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5e4, neginf=-5e4)
            logits = logits / max(temperature, 0.01)

            if repetition_penalty > 1.0 and generated_tokens:
                for tid in set(generated_tokens[-20:]):
                    logits[0, tid] /= repetition_penalty

            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                top_k_vals, _ = torch.topk(logits, k)
                threshold = top_k_vals[:, -1:]
                logits[logits < threshold] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumsum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                logits[indices_to_remove] = float("-inf")

            # 🛡️ Sanitize logits after filtering
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

            probs = torch.softmax(logits, dim=-1)
            # 🛡️ Sanitize probs: replace NaN with 0, handle all-zero case
            probs = torch.nan_to_num(probs, nan=0.0)
            if probs.sum() < 1e-8:
                # Fallback: uniform distribution over vocab
                probs = torch.ones_like(probs) / probs.shape[-1]
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize

            next_token = torch.multinomial(probs, 1)

            token_id = next_token[0, 0].item()
            generated_tokens.append(token_id)
            self.conversation_tokens.append(token_id)

            if token_id == self.model.config.eos_token_id:
                break

            result = self.model.forward(next_token, self.glt_states, self.window_caches)
            self.glt_states = result["glt_states"]
            self.window_caches = result["window_caches"]
            next_logits = result["logits"][:, -1, :]

        full_response = self.tokenizer.decode(generated_tokens)
        return full_response

    @torch.no_grad()
    def _stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        """
        Generate tokens one at a time (streaming).
        
        Yields individual token strings.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")

        prompt_ids = self.tokenizer.encode_with_special(
            prompt, add_bos=True, add_eos=False
        )
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        self.conversation_tokens.extend(prompt_ids)

        if self.glt_states is None:
            B = 1
            self.glt_states, self.window_caches = self.model.init_states(
                B, self.device, self.dtype
            )

        result = self.model.forward(prompt_tensor, self.glt_states, self.window_caches)
        self.glt_states = result["glt_states"]
        self.window_caches = result["window_caches"]
        next_logits = result["logits"][:, -1, :]

        generated_tokens = []
        for i in range(max_new_tokens):
            logits = next_logits.clone()
            
            # 🛡️ Sanitize logits before any processing
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5e4, neginf=-5e4)
            logits = logits / max(temperature, 0.01)

            if repetition_penalty > 1.0 and generated_tokens:
                for tid in set(generated_tokens[-20:]):
                    logits[0, tid] /= repetition_penalty

            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                top_k_vals, _ = torch.topk(logits, k)
                threshold = top_k_vals[:, -1:]
                logits[logits < threshold] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumsum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                logits[indices_to_remove] = float("-inf")

            # 🛡️ Sanitize logits after filtering
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

            probs = torch.softmax(logits, dim=-1)
            # 🛡️ Sanitize probs: replace NaN with 0, handle all-zero case
            probs = torch.nan_to_num(probs, nan=0.0)
            if probs.sum() < 1e-8:
                # Fallback: uniform distribution over vocab
                probs = torch.ones_like(probs) / probs.shape[-1]
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize

            next_token = torch.multinomial(probs, 1)

            token_id = next_token[0, 0].item()
            generated_tokens.append(token_id)
            self.conversation_tokens.append(token_id)

            if token_id == self.model.config.eos_token_id:
                break

            token_text = self.tokenizer.decode([token_id])
            yield token_text

            result = self.model.forward(next_token, self.glt_states, self.window_caches)
            self.glt_states = result["glt_states"]
            self.window_caches = result["window_caches"]
            next_logits = result["logits"][:, -1, :]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        stream: bool = True,
        repetition_penalty: float = 1.0,
    ):
        """
        Generate a response to a prompt.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            stream: If True, yields tokens one at a time
            repetition_penalty: >1.0 penalizes repeated tokens (1.0 = no penalty)

        Returns:
            Generated text (full string) if stream=False
            Yields tokens one by one if stream=True
        """
        if stream:
            # Returns a generator (has yield in body)
            return self._stream_generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            # Returns a plain string
            return self._generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

    def get_stats(self) -> Dict:
        """Return inference statistics for monitoring."""
        stats = {
            "conversation_tokens": len(self.conversation_tokens),
            "model_params": self.model.num_params,
            "model_size_mb": self._estimate_model_size(),
            "quantized": self.quantize,
            "has_active_state": self.glt_states is not None,
        }

        # Estimate checkpoint size
        if self.glt_states is not None:
            total_bytes = 0
            for s in self.glt_states:
                if s is not None:
                    total_bytes += s.numel() * s.element_size()
            stats["checkpoint_kb"] = total_bytes / 1024

        return stats
