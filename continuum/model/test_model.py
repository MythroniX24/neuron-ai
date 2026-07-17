"""
Test suite for the full Continuum model assembly and Adaptive Depth Looping.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from continuum.model.model import (
    ContinuumConfig, ContinuumModel, HaltingHead,
    create_continuum_nano, create_continuum_small, create_continuum_max,
)


def test_continuum_config():
    """Verify config validation."""
    cfg = ContinuumConfig()
    assert cfg.d_model == 192
    assert cfg.n_layers == 6
    assert cfg.perception_layers + cfg.core_layers + cfg.output_layers == cfg.n_layers
    assert cfg.glt_layers + cfg.anchor_layers == cfg.n_layers
    print("✓ ContinuumConfig: validated")


def test_halting_head():
    """Test halting head produces valid probabilities."""
    head = HaltingHead(d_model=192)
    x = torch.randn(4, 192)
    p = head(x)
    assert p.shape == (4, 1)
    assert (p >= 0).all() and (p <= 1).all(), "Halting probs must be in [0,1]"

    # Sequence input
    x_seq = torch.randn(2, 10, 192)
    p_seq = head(x_seq)
    assert p_seq.shape == (2, 1)

    print("✓ HaltingHead: valid probabilities")


def test_create_factories():
    """Test factory functions create valid models."""
    nano = create_continuum_nano()
    assert nano.num_params > 0
    print(f"  Continuum-Nano: {nano.num_params:,} params")

    small = create_continuum_small()
    print(f"  Continuum-Small: {small.num_params:,} params")

    # Nano: ~2.9M (architecture target 2-5M)
    assert 2_500_000 < nano.num_params < 5_000_000, \
        f"Nano params {nano.num_params:,} outside 2.5M-5M range"

    print("✓ Factory functions: valid models")


def test_model_forward():
    """Test full forward pass."""
    model = create_continuum_nano()
    model.eval()

    B, seq_len = 2, 8
    token_ids = torch.randint(0, 8000, (B, seq_len))

    with torch.no_grad():
        result = model.forward(token_ids)

    assert result["logits"].shape == (B, seq_len, 8000)
    assert len(result["glt_states"]) == model.config.n_layers  # one per layer
    assert "n_loops" in result
    assert "ponder_cost" in result

    print(f"✓ Model forward: logits shape={result['logits'].shape}, "
          f"avg loops={result['n_loops']:.2f}")


def test_state_management():
    """Test GLT state carry and window cache management across steps."""
    model = create_continuum_nano()
    model.eval()

    B = 1

    # First step
    t1 = torch.randint(0, 8000, (B, 1))
    with torch.no_grad():
        r1 = model.forward(t1)

    # Save copies of first-step states for comparison
    saved_states = [s.clone() if s is not None else None for s in r1["glt_states"]]
    saved_windows = [(wk.clone(), wv.clone()) for wk, wv in r1["window_caches"]]

    # Create SEPARATE copies to pass (forward modifies the list in-place)
    passed_states = [s.clone() if s is not None else None for s in saved_states]
    passed_windows = [(wk.clone(), wv.clone()) for wk, wv in saved_windows]

    # Second step (carry state)
    t2 = torch.randint(0, 8000, (B, 1))
    with torch.no_grad():
        r2 = model.forward(t2, passed_states, passed_windows)

    assert r2["logits"].shape == (B, 1, 8000)

    # States should differ between steps
    # Use saved_states (before update) vs r2["glt_states"] (after update)
    for s_before, s_after in zip(saved_states, r2["glt_states"]):
        if s_before is not None and s_after is not None:
            assert not torch.equal(s_before, s_after), \
                "GLT state should change between steps"

    print("✓ State management: states evolve correctly")


def test_state_serialization():
    """Test save/load of runtime state."""
    model = create_continuum_nano()

    B = 2
    glt_states, window_caches = model.init_states(B)

    # Run one step to populate states
    token_ids = torch.randint(0, 8000, (B, 4))
    with torch.no_grad():
        result = model.forward(token_ids, glt_states, window_caches)

    # Serialize
    state_dict = model.serialize_state(result["glt_states"], result["window_caches"])

    # Deserialize
    glt_loaded, windows_loaded = model.deserialize_state(state_dict)

    # Verify states match
    for s1, s2 in zip(result["glt_states"], glt_loaded):
        if s1 is not None:
            assert torch.equal(s1.cpu(), s2.cpu()), "GLT state mismatch after deserialize"

    # Check PMB
    assert torch.equal(
        state_dict["pmb_slots"], model.pmb.slots.data.cpu()
    ), "PMB slots mismatch in serialization"

    print("✓ State serialization: roundtrip successful")


def test_generation():
    """Test text generation (short)."""
    model = create_continuum_nano()
    model.eval()

    prompt = torch.randint(0, 8000, (1, 5))

    with torch.no_grad():
        generated, loop_counts = model.generate(
            prompt, max_new_tokens=10, temperature=1.0, top_k=40, top_p=0.9
        )

    assert generated.shape[0] == 1
    assert generated.shape[1] > 5  # generated some tokens
    assert len(loop_counts) > 0

    print(f"✓ Generation: generated {generated.shape[1] - 5} new tokens, "
          f"avg loops={sum(loop_counts)/len(loop_counts):.2f}")


def test_layer_composition():
    """Verify correct number of GLT and Anchor layers."""
    model = create_continuum_nano()

    glt_count = 0
    anchor_count = 0
    for block in (list(model.perception_blocks) +
                  list(model.core_blocks) +
                  list(model.output_blocks)):
        if block.is_glt:
            glt_count += 1
        elif block.is_anchor:
            anchor_count += 1

    assert glt_count == model.config.glt_layers, \
        f"Expected {model.config.glt_layers} GLT, got {glt_count}"
    assert anchor_count == model.config.anchor_layers, \
        f"Expected {model.config.anchor_layers} Anchor, got {anchor_count}"

    print(f"✓ Layer composition: {glt_count} GLT + {anchor_count} Anchor = "
          f"{glt_count + anchor_count} total")


def test_parameter_count():
    """Verify approximate parameter counts for Nano tier."""
    model = create_continuum_nano()
    params = model.num_params

    print(f"\n  Continuum-Nano parameter breakdown:")
    print(f"  Total: {params:,}")

    # Architecture formula estimate (Section 17):
    # embedding ~= vocab * d_embed + 2 * d_embed * d_model
    # GLT ~= 7 * d_model * d_state per layer
    # Anchor ~= 4 * d_model^2 per layer (before GQA reduction)
    # FFN ~= 3 * r * d_model^2 per layer
    cfg = model.config
    emb_est = cfg.vocab_size * cfg.d_embed + 2 * cfg.d_embed * cfg.d_model
    glt_est = 4 * (7 * cfg.d_model * cfg.d_state)
    anchor_est = 2 * (4 * cfg.d_model * cfg.d_model * (2/4))  # GQA ratio n_kv/n_heads
    ffn_est = 6 * (3 * cfg.ffn_expansion * cfg.d_model * cfg.d_model)
    pmb_est = cfg.pmb_slots * cfg.d_model + 4 * cfg.d_model * cfg.d_model
    halt_est = cfg.d_model * cfg.d_model // 4 + cfg.d_model // 4
    total_est = emb_est + glt_est + anchor_est + ffn_est + pmb_est + halt_est

    print(f"  Estimated: ~{total_est:,}")

    # Should be 2-5M
    assert 2_500_000 < params < 5_000_000, \
        f"Expected ~3M params, got {params:,}"

    print(f"✓ Parameter count: {params:,} (target ~3M)")


def test_max_parameter_count():
    """Verify Continuum-Max reaches ~100M parameters."""
    model = create_continuum_max()
    params = model.num_params

    print(f"\n  Continuum-Max parameter breakdown:")
    print(f"  Total: {params:,}")

    # Quick estimate
    cfg = model.config
    emb_est = cfg.vocab_size * cfg.d_embed + 2 * cfg.d_embed * cfg.d_model
    glt_est = cfg.glt_layers * (7 * cfg.d_model * cfg.d_state)
    anchor_est = cfg.anchor_layers * (4 * cfg.d_model * cfg.d_model * (cfg.n_kv_heads / cfg.n_heads))
    ffn_est = cfg.n_layers * (3 * cfg.ffn_expansion * cfg.d_model * cfg.d_model)
    pmb_est = cfg.pmb_slots * cfg.d_model + cfg.d_model * cfg.d_model * 2
    total_est = emb_est + glt_est + anchor_est + ffn_est + pmb_est

    print(f"  Embedding est:   {emb_est:>12,}")
    print(f"  GLT layers est:  {glt_est:>12,}")
    print(f"  Anchor layers est:{anchor_est:>12,}")
    print(f"  FFN layers est:  {ffn_est:>12,}")
    print(f"  PMB est:         {pmb_est:>12,}")
    print(f"  Total estimated: {total_est:>12,}")

    # Should be close to 100M (90-110M)
    assert 80_000_000 < params < 120_000_000, \
        f"Expected ~100M params, got {params:,}"

    print(f"\n✓ Continuum-Max: {params:,} parameters (target ~100M)")


def test_max_model_forward():
    """Test that the 100M model can forward."""
    model = create_continuum_max()
    model.eval()

    B, seq_len = 1, 4
    token_ids = torch.randint(0, 16000, (B, seq_len))

    with torch.no_grad():
        result = model.forward(token_ids)

    assert result["logits"].shape == (B, seq_len, 16000)
    print(f"✓ Max model forward: logits shape={result['logits'].shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Continuum Model Assembly Test Suite")
    print("=" * 60)

    test_continuum_config()
    test_halting_head()
    test_create_factories()
    test_model_forward()
    test_state_management()
    test_state_serialization()
    test_generation()
    test_layer_composition()
    test_parameter_count()
    test_max_parameter_count()
    test_max_model_forward()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
