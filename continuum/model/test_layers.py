"""
Test suite for Continuum core layers: Embedding, GLT, Gated Shard FFN.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn

from continuum.model.layers import (
    FactorizedEmbedding, GLTLayer, GatedShardFFN, RMSNorm
)


def test_rms_norm():
    """Test RMSNorm basic properties."""
    norm = RMSNorm(192)
    x = torch.randn(4, 192)
    y = norm(x)

    assert y.shape == x.shape
    # Variance should be approximately 1 (RMS normalization)
    rms = torch.sqrt(torch.mean(y.float() ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.3), \
        f"RMS after norm: {rms}"

    print("✓ RMSNorm: output has unit RMS")


def test_factorized_embedding():
    """Test factorized embedding: shapes, parameter count, and logits output."""
    vocab_size, d_model, d_embed = 8000, 192, 48
    emb = FactorizedEmbedding(vocab_size, d_model, d_embed)

    # Count parameters
    total = sum(p.numel() for p in emb.parameters())
    # Table: 8000 * 48 = 384,000
    # up_proj: 48 * 192 = 9,216
    # down_proj: 192 * 48 = 9,216
    expected = 384_000 + 9_216 + 9_216
    assert total == expected, f"Expected {expected} params, got {total}"

    # Embed
    token_ids = torch.randint(0, vocab_size, (2, 10))
    embeddings = emb.embed(token_ids)
    assert embeddings.shape == (2, 10, d_model)

    # Project to logits
    hidden = torch.randn(2, 10, d_model)
    logits = emb.project_to_logits(hidden)
    assert logits.shape == (2, 10, vocab_size)

    print(f"✓ FactorizedEmbedding: {total:,} params, shapes correct")


def test_glt_layer():
    """Test GLT layer: shapes, state carry, and multiple steps."""
    d_model, d_state = 192, 48
    glt = GLTLayer(d_model, d_state)

    # Count parameters
    total = sum(p.numel() for p in glt.parameters())
    # 7 * d_model * d_state + biases for gates + RMSNorm scale
    expected_approx = 7 * 192 * 48 + 3 * 48 + d_model + d_state
    assert abs(total - expected_approx) < 50, \
        f"Expected ~{expected_approx} params, got {total}"

    batch_size = 2
    x0 = torch.randn(batch_size, d_model)
    x1 = torch.randn(batch_size, d_model)
    x2 = torch.randn(batch_size, d_model)

    # Step 1: fresh state
    o1, s1 = glt(x0)
    assert o1.shape == (batch_size, d_model)
    assert s1.shape == (batch_size, d_state, d_state)
    assert not torch.allclose(s1, torch.zeros_like(s1)), "State should be non-zero after first step"

    # Step 2: carry state
    o2, s2 = glt(x1, s1)
    assert o2.shape == (batch_size, d_model)
    assert s2.shape == (batch_size, d_state, d_state)
    assert not torch.equal(s2, s1), "State should change between steps"

    # Step 3
    o3, s3 = glt(x2, s2)

    # Test sequence mode
    x_seq = torch.randn(batch_size, 5, d_model)
    outputs, final_state = glt.forward_sequence(x_seq)
    assert outputs.shape == (batch_size, 5, d_model)
    assert final_state.shape == (batch_size, d_state, d_state)

    # Test reset_state
    state = glt.reset_state(batch_size=3, device="cpu")
    assert state.shape == (3, d_state, d_state)
    assert torch.all(state == 0)

    print(f"✓ GLTLayer: {total:,} params, state carry works, shapes correct")


def test_gated_shard_ffn():
    """Test Gated Shard FFN: shapes, parameter count, and sparsity behavior."""
    d_model, expansion, num_shards = 192, 3, 2
    ffn = GatedShardFFN(d_model, expansion, num_shards)

    # Count parameters
    total = sum(p.numel() for p in ffn.parameters())

    # Per shard: gate_proj + up_proj + down_proj
    # shard_intermediate = (3*192)//2 = 288
    # gate_proj: 192*288, up_proj: 192*288, down_proj: 288*192
    # Per shard: 3*192*288 = 165,888
    # 2 shards: 331,776
    # gate_head: 192*2 + 2 = 386
    # RMSNorm: 192
    expected = 331_776 + 386 + 192
    assert abs(total - expected) < 50, f"Expected ~{expected} params, got {total}"

    # Single token
    x = torch.randn(4, d_model)  # batch of 4
    y = ffn(x)
    assert y.shape == (4, d_model)

    # Sequence
    x_seq = torch.randn(2, 8, d_model)  # batch 2, seq 8
    y_seq = ffn(x_seq)
    assert y_seq.shape == (2, 8, d_model)

    # Verify residual: output should differ from input
    assert not torch.allclose(y, x)

    # Test eval mode (sparsity threshold applied)
    ffn.eval()
    with torch.no_grad():
        y_eval = ffn(x)
    assert y_eval.shape == x.shape

    print(f"✓ GatedShardFFN: {total:,} params, shapes correct")


def test_glt_numerical_stability():
    """Verify GLT doesn't produce NaN/Inf over many steps."""
    d_model, d_state = 192, 48
    glt = GLTLayer(d_model, d_state)
    glt.eval()

    batch_size = 2
    state = glt.reset_state(batch_size)
    max_val = 0.0

    with torch.no_grad():
        for step in range(1000):
            x = torch.randn(batch_size, d_model) * 0.5
            o, state = glt(x, state)
            assert not torch.isnan(o).any(), f"NaN at step {step}"
            assert not torch.isinf(o).any(), f"Inf at step {step}"
            max_val = max(max_val, o.abs().max().item())

    print(f"✓ GLT numerical stability: 1000 steps, max output={max_val:.4f}, no NaN/Inf")


def test_parameter_count_approximation():
    """Verify the layer parameter counts match the architecture estimates."""
    d_model, d_state = 192, 48
    vocab, d_embed = 8000, 48

    emb = FactorizedEmbedding(vocab, d_model, d_embed)
    emb_params = sum(p.numel() for p in emb.parameters())

    glt = GLTLayer(d_model, d_state)
    glt_params = sum(p.numel() for p in glt.parameters())

    ffn = GatedShardFFN(d_model, expansion=3, num_shards=2)
    ffn_params = sum(p.numel() for p in ffn.parameters())

    # Architecture formulas:
    # embedding ~= vocab * d_embed + 2 * (d_embed * d_model)
    emb_expected = vocab * d_embed + 2 * (d_embed * d_model)
    # GLT ~= 7 * d_model * d_state (+ small bias/norm terms)
    glt_expected = 7 * d_model * d_state + 3 * d_state + d_model + d_state
    # FFN ~= 3 * r * d_model^2 + K * d_model + d_model
    ffn_expected = 3 * 3 * d_model * d_model + 2 * d_model + d_model

    print(f"\n  Architecture vs Implementation:")
    print(f"  Embedding: expected ~{emb_expected:,}, actual {emb_params:,}")
    print(f"  GLT:       expected ~{glt_expected:,}, actual {glt_params:,}")
    print(f"  FFN:       expected ~{ffn_expected:,}, actual {ffn_params:,}")

    # All should be within 5% of expected
    assert abs(emb_params - emb_expected) / emb_expected < 0.05
    assert abs(glt_params - glt_expected) / glt_expected < 0.05
    assert abs(ffn_params - ffn_expected) / ffn_expected < 0.05

    print("✓ Parameter counts match architecture estimates")


if __name__ == "__main__":
    print("=" * 60)
    print("Continuum Core Layers Test Suite")
    print("=" * 60)

    test_rms_norm()
    test_factorized_embedding()
    test_glt_layer()
    test_gated_shard_ffn()
    test_glt_numerical_stability()
    test_parameter_count_approximation()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
