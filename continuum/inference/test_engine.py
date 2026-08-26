"""
Regression tests for the inference engine fixes.

These would have caught two production bugs found in audit:
1. INT8 quantized forward crashed (QuantizedLinear vs _get_fused_kv_weight).
2. state_dict() of a quantized model silently dropped all big layer weights.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from continuum.model.model import create_continuum_nano, create_continuum_small
from continuum.inference.engine import (
    ContinuumInference,
    ContinuumSpeculativeDecoder,
    QuantizedLinear,
)
from continuum.tokenizer.bpe import ContinuumTokenizer


def _make_tokenizer(vocab_size):
    return ContinuumTokenizer(vocab_size=vocab_size)


def _quantize_in_place(model):
    """Apply the engine's quantization without running engine __init__."""
    engine = object.__new__(ContinuumInference)
    engine.model = model
    engine._apply_quantization()
    return engine


def test_quantized_linear_buffers_in_state_dict():
    """Quantized weights/scale/bias must survive state_dict (audit bug C2).

    Checks PER-MODULE that every QuantizedLinear's registered buffers appear in
    state_dict. (Global suffix-counting is wrong: RMSNorm etc. also own `.scale`
    keys, so total counts legitimately differ.)
    """
    model = create_continuum_nano()
    _quantize_in_place(model)

    sd = model.state_dict()
    quant_linears = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, QuantizedLinear)
    ]
    assert len(quant_linears) > 0, "No QuantizedLinear modules found after quantization"

    missing = []
    for name, mod in quant_linears:
        for buf_name in ("weight_int8", "scale"):
            key = f"{name}.{buf_name}"
            if key not in sd:
                missing.append(key)
        if mod.bias is not None and f"{name}.bias" not in sd:
            missing.append(f"{name}.bias")
    assert not missing, f"Quantized buffers dropped by state_dict(): {missing[:6]}"


def test_quantized_model_state_dict_round_trip():
    """A quantized model's state_dict must fully reload into an identical skeleton."""
    src = create_continuum_nano()
    _quantize_in_place(src)
    sd = src.state_dict()

    dst = create_continuum_nano()
    _quantize_in_place(dst)
    result = dst.load_state_dict(sd, strict=True)
    assert not result.missing_keys, f"missing: {result.missing_keys[:5]}"
    assert not result.unexpected_keys, f"unexpected: {result.unexpected_keys[:5]}"


def test_quantized_engine_generate_runs():
    """INT8 quantized inference must not crash (audit bug C1: AttributeError)."""
    model = create_continuum_nano()
    tokenizer = _make_tokenizer(model.config.vocab_size)
    engine = ContinuumInference(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        quantize=True,
        use_compile=False,
    )
    out = engine.generate("Hello", max_new_tokens=3, stream=False)
    assert isinstance(out, str) and len(out) >= 0


def test_pmb_write_is_wired_into_conversation():
    """_maybe_write_pmb must actually update PMB slots every chunk_size tokens."""
    model = create_continuum_nano()
    tokenizer = _make_tokenizer(model.config.vocab_size)
    engine = ContinuumInference(
        model=model, tokenizer=tokenizer, device="cpu",
        quantize=False, use_compile=False,
    )
    engine.start_conversation()

    chunk = model.config.chunk_size
    engine.conversation_tokens = list(range(3, 3 + chunk))
    slots_before = model.pmb.slots.detach().clone()
    engine._maybe_write_pmb()
    slots_after = model.pmb.slots.detach()
    assert not torch.allclose(slots_before, slots_after), \
        "PMB slots unchanged — write path still disconnected"


def test_speculative_decoder_rejects_vocab_mismatch():
    """Draft/target vocab mismatch must fail fast, not crash mid-generation."""
    draft = create_continuum_nano()            # vocab 8000
    target = create_continuum_small()          # vocab 12000
    try:
        ContinuumSpeculativeDecoder(
            draft_model=draft,
            target_model=target,
            tokenizer=_make_tokenizer(8000),
            device="cpu",
            use_compile=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for vocab-size mismatch")
