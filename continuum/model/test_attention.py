"""
Test suite for Anchor Attention and Persistent Memory Bank.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from continuum.model.attention import AnchorAttention, PersistentMemoryBank


def test_anchor_attention_shapes():
    """Test basic shapes of Anchor Attention."""
    d_model, n_heads, n_kv_heads = 192, 4, 2
    window_size, n_anchors, n_static = 48, 8, 4

    attn = AnchorAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        window_size=window_size,
        n_anchors=n_anchors,
        n_static_anchors=n_static,
    )

    # Count parameters
    total = sum(p.numel() for p in attn.parameters())
    n_pmb = n_anchors - n_static
    print(f"  AnchorAttention params: {total:,}")

    B = 2

    # Initialize window cache
    wk, wv = attn.init_window_cache(B)

    # Single token forward (no PMB readouts)
    x = torch.randn(B, d_model)
    y = attn(x, wk, wv)
    assert y.shape == (B, d_model), f"Expected ({B}, {d_model}), got {y.shape}"

    # With PMB readouts
    pmb = torch.randn(B, n_pmb, d_model)
    y2 = attn(x, wk, wv, pmb_readouts=pmb)
    assert y2.shape == (B, d_model)

    # Sequence forward
    x_seq = torch.randn(B, 8, d_model)
    wk, wv = attn.init_window_cache(B)
    # For sequence, we need per-position window — use a simple sliding approach
    outputs = []
    for t in range(x_seq.shape[1]):
        o = attn(x_seq[:, t, :], wk, wv, pmb_readouts=pmb)
        outputs.append(o)
        wk, wv = attn.update_window_cache(x_seq[:, t, :], wk, wv)
    output_seq = torch.stack(outputs, dim=1)
    assert output_seq.shape == (B, 8, d_model)

    print(f"✓ AnchorAttention: shapes correct, params={total:,}")


def test_anchor_attention_alibi():
    """Test that ALiBi biases are applied correctly (window only, not anchors)."""
    attn = AnchorAttention(d_model=192, n_heads=4, n_kv_heads=2,
                           window_size=48, n_anchors=8, n_static_anchors=4)

    # Check ALiBi slopes
    assert attn.alibi_slopes.shape == (4,), f"Expected 4 slopes, got {attn.alibi_slopes.shape}"
    # Slopes should be decreasing (more negative for later heads)
    assert attn.alibi_slopes[0] > attn.alibi_slopes[-1], "ALiBi slopes should decrease"
    # Check precomputed bias shape — [1, 1, n_heads, window_size] for correct broadcasting
    assert attn.alibi_bias_full.shape == (1, 1, 4, 48), \
        f"Expected (1,1,4,48), got {attn.alibi_bias_full.shape}"
    # Bias should be negative (penalty for distance)
    assert (attn.alibi_bias_full <= 0).all(), "ALiBi bias should be non-positive"

    print("✓ AnchorAttention ALiBi: slopes correct, bias non-positive")


def test_window_cache_update():
    """Test sliding window cache management."""
    attn = AnchorAttention(d_model=192, n_heads=4, n_kv_heads=2,
                           window_size=4, n_anchors=4, n_static_anchors=2)

    B = 2
    wk, wv = attn.init_window_cache(B)

    # Feed 6 tokens through
    for i in range(6):
        x = torch.randn(B, 192)
        wk, wv = attn.update_window_cache(x, wk, wv)
        # Cache should always be window_size
        assert wk.shape[1] == 4
        assert wv.shape[1] == 4

    print("✓ Window cache: size bounded correctly")


def test_persistent_memory_bank():
    """Test PMB read/write operations."""
    n_slots, d_mem, n_readout = 16, 192, 4
    pmb = PersistentMemoryBank(n_slots, d_mem, n_readout)

    B = 2

    # Check initial slot shape
    assert pmb.slots.shape == (n_slots, d_mem)

    # Read: should return k slots
    query = torch.randn(B, d_mem)
    readouts = pmb.read(query, k=n_readout)
    assert readouts.shape == (B, n_readout, d_mem)

    # Write: update slots based on chunk summary
    chunk = torch.randn(B, d_mem)
    old_slots = pmb.slots.data.clone()
    pmb.write(chunk)
    # Slots should have changed
    assert not torch.equal(pmb.slots.data, old_slots), "PMB slots should change after write"

    # Serialize/deserialize
    saved = pmb.serialize()
    assert saved.shape == (n_slots, d_mem)

    pmb2 = PersistentMemoryBank(n_slots, d_mem, n_readout)
    pmb2.deserialize(saved)
    assert torch.equal(pmb2.slots.data, pmb.slots.data)

    # Reset
    pmb.reset()
    assert not torch.equal(pmb.slots.data, old_slots)

    print(f"✓ PersistentMemoryBank: {n_slots} slots, read/write/serialize work")


def test_attention_with_pmb_integration():
    """Test end-to-end: PMB read -> Anchor Attention."""
    d_model = 192
    attn = AnchorAttention(d_model=d_model, n_heads=4, n_kv_heads=2,
                           window_size=48, n_anchors=8, n_static_anchors=4)
    pmb = PersistentMemoryBank(n_slots=16, d_mem=d_model, n_readout=4)

    B = 2

    # Write something to PMB
    chunk = torch.randn(B, d_model)
    pmb.write(chunk)

    # Read from PMB
    query = torch.randn(B, d_model)
    pmb_readouts = pmb.read(query)

    # Use in attention
    wk, wv = attn.init_window_cache(B)
    x = torch.randn(B, d_model)
    y = attn(x, wk, wv, pmb_readouts=pmb_readouts)
    assert y.shape == (B, d_model)

    print("✓ PMB-Attention integration: end-to-end working")


def test_gqa_kv_repeat():
    """Test GQA key/value head expansion."""
    attn = AnchorAttention(d_model=192, n_heads=4, n_kv_heads=2,
                           window_size=8, n_anchors=4, n_static_anchors=2)

    # Create KV with 2 heads
    kv = torch.randn(2, 10, 2, 48)  # [B, seq, n_kv_heads, head_dim]
    expanded = attn._repeat_kv_for_gqa(kv)
    assert expanded.shape == (2, 10, 4, 48), f"Expected (2,10,4,48), got {expanded.shape}"

    # Verify grouping: head 0 and 1 should be identical (same KV group)
    assert torch.equal(expanded[0, 0, 0], expanded[0, 0, 1]), \
        "GQA: heads in same group should be identical"

    print("✓ GQA KV repeat: correct head expansion")


if __name__ == "__main__":
    print("=" * 60)
    print("Continuum Anchor Attention & PMB Test Suite")
    print("=" * 60)

    test_anchor_attention_shapes()
    test_anchor_attention_alibi()
    test_window_cache_update()
    test_persistent_memory_bank()
    test_attention_with_pmb_integration()
    test_gqa_kv_repeat()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
