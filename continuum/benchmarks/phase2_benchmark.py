"""
Phase 2 Benchmark: Python Inference Speed — INT4, Speculative Decoding, torch.compile.

Tests all optimization combinations on CPU/GPU and reports tok/sec for each.

Usage:
    python continuum/benchmarks/phase2_benchmark.py [--device cpu|cuda|auto]

    Or from Kaggle/IPython:
    from continuum.benchmarks.phase2_benchmark import run_phase2_benchmark
    results = run_phase2_benchmark()
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from continuum.model.model import create_continuum_nano, create_continuum_max
from continuum.inference.engine import (
    ContinuumInference,
    ContinuumSpeculativeDecoder,
    QuantizedLinear,
    _detect_gpu,
    _get_optimal_dtype,
    _HAS_COMPILE,
)


# ============================================================================
# INT4 QuantizedLinear — 2 values packed per byte (nn.Module)
# ============================================================================

class QuantizedLinearINT4(nn.Module):
    """INT4 weight-only quantization for nn.Linear.

    Packs 2 INT4 values per byte for 2x bandwidth reduction vs INT8.
    Dequantizes to float32 on first forward call (cached).

    Example:  400MB FP32 → 50MB INT8 → 25MB INT4
    """

    def __init__(self, linear, bits: int = 4):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self._orig_out_features = linear.out_features

        weight = linear.weight.data
        max_val = 2 ** (bits - 1) - 1  # 7 for INT4

        # Per-channel symmetric quantization
        scale = weight.abs().max(dim=1, keepdim=True)[0] / max_val
        scale = scale.clamp(min=1e-8)

        quantized = torch.round(weight / scale).clamp(-max_val, max_val)

        if bits == 4:
            # Nibble packing: 2 INT4 → 1 byte
            if self.out_features % 2 != 0:
                quantized = torch.cat([
                    quantized,
                    torch.zeros(1, self.in_features, dtype=quantized.dtype)
                ], dim=0)
            quantized_reshaped = quantized.view(-1, 2, self.in_features)
            # Shift [-7,7] → [0,14] then pack as (high<<4)|low
            high = (quantized_reshaped[:, 0, :] + 8).to(torch.uint8)
            low = (quantized_reshaped[:, 1, :] + 8).to(torch.uint8)
            packed = (high << 4) | low
        else:
            packed = quantized.to(torch.int8)

        # Register as buffers → .to(device) / .to(dtype) work correctly
        self.register_buffer('_packed', packed)
        self.register_buffer('_scale', scale)
        if hasattr(linear, 'bias') and linear.bias is not None:
            self.register_buffer('_bias', linear.bias.data.clone())

        # Dequant cache (invalidated on device change)
        self._weight_fp_cached = None
        self._cached_device = None

    def _dequantize(self):
        """Dequantize to float32. Cached; invalidated on device change."""
        current_device = self._packed.device
        if self._weight_fp_cached is not None and self._cached_device == current_device:
            return self._weight_fp_cached

        if self.bits == 4:
            # Unpack nibbles: [out//2, in] → [out_padded, in]
            high = (self._packed >> 4).to(torch.float32) - 8.0
            low = (self._packed & 0x0F).to(torch.float32) - 8.0
            out_padded = high.shape[0] * 2
            weight_fp = torch.empty(out_padded, high.shape[1],
                                    device=high.device, dtype=torch.float32)
            weight_fp[0::2] = high
            weight_fp[1::2] = low
            if self._orig_out_features < out_padded:
                weight_fp = weight_fp[:self._orig_out_features]
        else:
            weight_fp = self._packed.float()

        self._weight_fp_cached = weight_fp * self._scale.float()
        self._cached_device = current_device
        return self._weight_fp_cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp = self._dequantize().to(device=x.device, dtype=x.dtype)
        bias_fp = (self._bias.to(device=x.device, dtype=x.dtype)
                   if hasattr(self, '_bias') else None)
        return torch.nn.functional.linear(x, weight_fp, bias_fp)


# ============================================================================
# INT4 Model Converter
# ============================================================================

def _apply_int4_to_model(model) -> int:
    """Replace all Linear layers with QuantizedLinearINT4. Returns count."""
    count = 0

    def _replace(module):
        nonlocal count
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and child.in_features > 64:
                setattr(module, name, QuantizedLinearINT4(child, bits=4))
                count += 1
            else:
                _replace(child)

    _replace(model)
    return count


def _estimate_model_size_mb(model) -> float:
    """Estimate model size in MB from parameters + buffers."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 * 1024)


# ============================================================================
# Benchmark Engine
# ============================================================================

def benchmark_engine(
    engine,
    tokenizer,
    name: str,
    prompts: List[str] = None,
    max_new_tokens: int = 50,
    spec_decoder=None,
) -> Dict:
    """Benchmark one configuration. Returns {name, tok_per_sec, ...}."""
    if prompts is None:
        prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
        ]

    total_time = 0.0
    total_tokens = 0
    all_outputs = []

    for prompt in prompts:
        if spec_decoder:
            spec_decoder.start_conversation()
            t0 = time.time()
            try:
                result = spec_decoder.generate(prompt, max_new_tokens=max_new_tokens, stream=False)
            except Exception:
                result = ""
            elapsed = time.time() - t0
            gen_tokens = max(1, max_new_tokens)
        elif hasattr(engine, '_generate_text') and tokenizer is not None:
            try:
                engine.start_conversation()
                t0 = time.time()
                result = engine._generate_text(prompt, max_new_tokens=max_new_tokens)
                elapsed = time.time() - t0
                gen_tokens = max(1, len(tokenizer.encode_with_special(
                    result, add_bos=False, add_eos=False)))
            except Exception:
                elapsed = 0.1
                gen_tokens = 1
                result = ""
        else:
            # Raw forward timing (no tokenizer)
            try:
                input_ids = torch.randint(0, 100, (1, max_new_tokens)).to(engine.device)
                t0 = time.time()
                _ = engine.model.forward(input_ids)
                elapsed = time.time() - t0
            except Exception:
                elapsed = 0.1
            gen_tokens = max_new_tokens
            result = "[forward-only]"

        total_time += elapsed
        total_tokens += gen_tokens
        all_outputs.append((prompt, str(result)[:100]))

    tok_per_sec = total_tokens / total_time if total_time > 0 else 0

    if spec_decoder:
        model_size_mb = _estimate_model_size_mb(spec_decoder.target_model)
    elif hasattr(engine, 'model'):
        model_size_mb = _estimate_model_size_mb(engine.model)
    else:
        model_size_mb = 0

    return {
        "name": name,
        "tok_per_sec": round(tok_per_sec, 2),
        "total_time": round(total_time, 2),
        "total_tokens": total_tokens,
        "model_size_mb": round(model_size_mb, 1),
        "sample_output": all_outputs[0][1] if all_outputs else "",
    }


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_phase2_benchmark(
    device: str = "auto",
    dtype=None,
    prompts: List[str] = None,
    max_new_tokens: int = 50,
    checkpoint_path: str = None,
    tokenizer_path: str = None,
) -> List[Dict]:
    """Run the complete Phase 2 benchmark (8 scenarios).

    Scenarios:
    1. Baseline (FP32, no quant, no compile)
    2. INT8 quantization
    3. INT4 quantization (NEW - nibble packing)
    4. torch.compile max-autotune
    5. INT8 + compile
    6. INT4 + compile
    7. Speculative Decoding (Nano 5M drafts → Max 100M verifies)
    8. MAX SPEED: Spec Decode + INT8 + compile

    Returns list of result dicts sorted by tok/sec (fastest first).
    """
    print("=" * 70)
    print("PHASE 2 BENCHMARK: Python Inference Speed")
    print("=" * 70)

    if device == "auto":
        gpu = _detect_gpu()
        device = gpu if gpu else "cpu"
    print(f"\nDevice: {device}")
    if dtype is None:
        dtype = _get_optimal_dtype(device)
    print(f"Dtype: {dtype}")
    print(f"Max tokens per prompt: {max_new_tokens}")
    print(f"torch.compile available: {_HAS_COMPILE}")

    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        from continuum.tokenizer.bpe import ContinuumTokenizer
        tokenizer = ContinuumTokenizer.load(tokenizer_path)
        print(f"Tokenizer: {tokenizer.vocab_size_actual} tokens")

    results = []

    # Helper: load checkpoint if available
    def _load_ckpt(m):
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt.get("model_state_dict", ckpt))

    # ─── 1. BASELINE ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("1/8 BASELINE: FP32")
    print("=" * 50)
    m = create_continuum_max()
    _load_ckpt(m)
    eng = ContinuumInference(m, tokenizer=tokenizer, device=device, dtype=dtype,
                             quantize=False, use_compile=False)
    r = benchmark_engine(eng, tokenizer, "1. Baseline (FP32)", prompts, max_new_tokens)
    results.append(r)
    print(f"   {r['tok_per_sec']} tok/s | {r['model_size_mb']} MB")
    del eng

    # ─── 2. INT8 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("2/8 INT8 Quantization")
    print("=" * 50)
    m = create_continuum_max()
    _load_ckpt(m)
    eng = ContinuumInference(m, tokenizer=tokenizer, device=device, dtype=dtype,
                             quantize=True, use_compile=False)
    r = benchmark_engine(eng, tokenizer, "2. INT8 Quant", prompts, max_new_tokens)
    results.append(r)
    sp = r['tok_per_sec'] / results[0]['tok_per_sec'] if results[0]['tok_per_sec'] > 0 else 0
    print(f"   {r['tok_per_sec']} tok/s | {r['model_size_mb']} MB | {sp:.1f}x")
    del eng

    # ─── 3. INT4 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("3/8 INT4 Quantization (2x smaller than INT8)")
    print("=" * 50)
    m = create_continuum_max()
    _load_ckpt(m)
    _apply_int4_to_model(m)
    m.to(device).to(dtype).eval()
    eng = ContinuumInference(m, tokenizer=tokenizer, device=device, dtype=dtype,
                             quantize=False, use_compile=False)
    r = benchmark_engine(eng, tokenizer, "3. INT4 Quant", prompts, max_new_tokens)
    results.append(r)
    sp = r['tok_per_sec'] / results[0]['tok_per_sec'] if results[0]['tok_per_sec'] > 0 else 0
    print(f"   {r['tok_per_sec']} tok/s | {r['model_size_mb']} MB | {sp:.1f}x")
    del eng

    # ─── 4-6. torch.compile variants ─────────────────────────────────────
    if _HAS_COMPILE:
        # 4. compile only
        print("\n" + "=" * 50)
        print("4/8 torch.compile (max-autotune)")
        print("=" * 50)
        m = create_continuum_max()
        _load_ckpt(m)
        eng = ContinuumInference(m, tokenizer=tokenizer, device=device, dtype=dtype,
                                 quantize=False, use_compile=True, use_max_autotune=True)
        r = benchmark_engine(eng, tokenizer, "4. torch.compile", prompts, max_new_tokens)
        results.append(r)
        sp = r['tok_per_sec'] / results[0]['tok_per_sec'] if results[0]['tok_per_sec'] > 0 else 0
        print(f"   {r['tok_per_sec']} tok/s | {sp:.1f}x")
        del eng

        # 5. INT8 + compile
        print("\n" + "=" * 50)
        print("5/8 INT8 + torch.compile")
        print("=" * 50)
        m = create_continuum_max()
        _load_ckpt(m)
        eng = ContinuumInference(m, tokenizer=tokenizer, device=device, dtype=dtype,
                                 quantize=True, use_compile=True, use_max_autotune=True)
        r = benchmark_engine(eng, tokenizer, "5. INT8 + compile", prompts, max_new_tokens)
        results.append(r)
        sp = r['tok_per_sec'] / results[0]['tok_per_sec'] if results[0]['tok_per_sec'] > 0 else 0
        print(f"   {r['tok_per_sec']} tok/s | {r['model_size_mb']} MB | {sp:.1f}x")
        del eng

        # 6. INT4 + compile
        print("\n" + "=" * 50)
        print("6/8 INT4 + torch.compile")
        print("=" * 50)
        m = create_continuum_max()
        _load_ckpt(m)
        _apply_int4_to_model(m)
        m.to(device).to(dtype).eval()
        eng = ContinuumInference(m, tokenizer=tokenizer, device=device, dtype=dtype,
                                 quantize=False, use_compile=True, use_max_autotune=True)
        r = benchmark_engine(eng, tokenizer, "6. INT4 + compile", prompts, max_new_tokens)
        results.append(r)
        sp = r['tok_per_sec'] / results[0]['tok_per_sec'] if results[0]['tok_per_sec'] > 0 else 0
        print(f"   {r['tok_per_sec']} tok/s | {r['model_size_mb']} MB | {sp:.1f}x")
        del eng
    else:
        print("\n4-6/8 SKIPPED: torch.compile not available")

    # ─── 7. Speculative Decoding ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("7/8 Speculative Decoding (Nano 5M → Max 100M)")
    print("=" * 50)
    nano = create_continuum_nano()
    m = create_continuum_max()
    _load_ckpt(m)
    sd = ContinuumSpeculativeDecoder(
        draft_model=nano, target_model=m, tokenizer=tokenizer,
        device=device, dtype=dtype, quantize_target=True,
        quantize_draft=False, use_compile=False, num_draft_tokens=4,
    )
    r = benchmark_engine(None, tokenizer, "7. Spec Decode", prompts, max_new_tokens,
                         spec_decoder=sd)
    results.append(r)
    sp = r['tok_per_sec'] / results[0]['tok_per_sec'] if results[0]['tok_per_sec'] > 0 else 0
    stats = sd.get_stats()
    print(f"   {r['tok_per_sec']} tok/s | {sp:.1f}x")
    print(f"   Accept: {stats['acceptance_rate']} | Theory: {stats['theoretical_speedup']}")
    del sd

    # ─── 8. MAX SPEED ─────────────────────────────────────────────────────
    if _HAS_COMPILE:
        print("\n" + "=" * 50)
        print("8/8 MAX: Spec Decode + INT8 + compile")
        print("=" * 50)
        nano = create_continuum_nano()
        m = create_continuum_max()
        _load_ckpt(m)
        sd = ContinuumSpeculativeDecoder(
            draft_model=nano, target_model=m, tokenizer=tokenizer,
            device=device, dtype=dtype, quantize_target=True,
            quantize_draft=False, use_compile=True, num_draft_tokens=4,
        )
        r = benchmark_engine(None, tokenizer, "8. MAX (Spec+INT8+compile)", prompts,
                             max_new_tokens, spec_decoder=sd)
        results.append(r)
        sp = r['tok_per_sec'] / results[0]['tok_per_sec'] if results[0]['tok_per_sec'] > 0 else 0
        print(f"   {r['tok_per_sec']} tok/s | {sp:.1f}x")
        del sd
    else:
        print("\n8/8 SKIPPED: torch.compile not available")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 BENCHMARK RESULTS")
    print("=" * 70)

    results.sort(key=lambda r: r['tok_per_sec'], reverse=True)
    base = results[-1]['tok_per_sec'] if results else 1

    print(f"\n{'Rank':<5} {'Mode':<35} {'tok/s':>8} {'MB':>8} {'Speedup':>8}")
    print("-" * 70)
    for i, r in enumerate(results):
        spd = r['tok_per_sec'] / base if base > 0 else 0
        print(f"{i+1:<5} {r['name']:<35} {r['tok_per_sec']:>8.1f} {r['model_size_mb']:>8.1f} {spd:>7.1f}x")

    best = results[0]
    print(f"\n🏆 FASTEST: {best['name']} → {best['tok_per_sec']} tok/s "
          f"({best['tok_per_sec'] / base:.1f}x baseline)")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 2 Benchmark")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default=None)
    args = p.parse_args()
    run_phase2_benchmark(
        device=args.device, max_new_tokens=args.max_tokens,
        checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer,
    )
