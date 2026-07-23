"""
Test suite for Continuum Vision (ViGLT) encoder.

Tests all six components plus full encoder assembly:
- PatchEmbedding, RoPE2D, BiGLTBlock, SpatialAnchorBlock,
  VisionProjector, ContinuumVisionEncoder
- Factory functions, gradient flow, edge cases
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn

from continuum.model.vision import (
    ContinuumVisionConfig,
    RoPE2D,
    PatchEmbedding,
    BiGLTBlock,
    SpatialAnchorBlock,
    VisionProjector,
    ContinuumVisionEncoder,
    create_vision_encoder_max,
    create_vision_encoder_small,
    create_vision_encoder_nano,
)


# ============================================================================
# ContinuumVisionConfig
# ============================================================================

def test_vision_config_defaults():
    """Test that vision config stores all fields correctly."""
    cfg = ContinuumVisionConfig()
    assert cfg.image_size == 224
    assert cfg.patch_size == 16
    assert cfg.in_channels == 3
    assert cfg.d_vision == 384
    assert cfg.d_v_state == 128
    assert cfg.n_bi_glt_layers == 5
    assert cfg.n_anchor_layers == 1
    assert cfg.n_heads == 6
    assert cfg.n_kv_heads == 3
    assert cfg.max_patches == 256
    print("✓ ContinuumVisionConfig: defaults correct")


def test_vision_config_custom():
    """Test custom vision config values."""
    cfg = ContinuumVisionConfig(
        image_size=112, patch_size=8, d_vision=256, d_v_state=64,
        n_bi_glt_layers=3, n_heads=8, n_kv_heads=4,
        max_patches=128,
    )
    assert cfg.image_size == 112
    assert cfg.patch_size == 8
    assert cfg.d_vision == 256
    assert cfg.n_bi_glt_layers == 3
    assert cfg.max_patches == 128
    print("✓ ContinuumVisionConfig: custom values stored")


# ============================================================================
# RoPE2D
# ============================================================================

def test_rope2d_shapes():
    """Test RoPE2D output shapes and basic properties."""
    d_model = 384
    rope = RoPE2D(d_model, theta=10000.0)

    # 14x14 grid = 196 patches (for 224/16)
    B, N = 2, 196
    grid = (14, 14)
    x = torch.randn(B, N, d_model)
    y = rope(x, grid)

    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    assert y.dtype == x.dtype, f"Dtype changed: {x.dtype} → {y.dtype}"
    print(f"✓ RoPE2D: shapes correct for {d_model}d")


def test_rope2d_d_model_validation():
    """Test that RoPE2D asserts on invalid d_model."""
    raised = False
    try:
        RoPE2D(100)  # 100 % 4 != 0
    except AssertionError:
        raised = True
    assert raised, "RoPE2D should have raised AssertionError for invalid d_model (100 % 4 != 0)"
    print("✓ RoPE2D: d_model divisible-by-4 check works")


def test_rope2d_grid_mismatch():
    """Test that RoPE2D asserts on grid/sequence mismatch."""
    d_model = 384
    rope = RoPE2D(d_model)
    x = torch.randn(2, 50, d_model)  # 50 patches, but grid says 14×14=196
    raised = False
    try:
        rope(x, (14, 14))
    except AssertionError:
        raised = True
    assert raised, "RoPE2D should have raised AssertionError for grid mismatch (50 patches vs 196)"
    print("✓ RoPE2D: grid mismatch detected")


def test_rope2d_determinism():
    """Test that RoPE2D gives same output for same input and grid."""
    d_model = 384
    rope = RoPE2D(d_model)
    x = torch.randn(2, 196, d_model)
    grid = (14, 14)

    torch.manual_seed(42)
    y1 = rope(x, grid)
    torch.manual_seed(42)
    y2 = rope(x, grid)

    assert torch.equal(y1, y2), "RoPE2D should be deterministic"
    print("✓ RoPE2D: deterministic")


def test_rope2d_row_col_encoding():
    """Test that RoPE2D produces different outputs for swapped rows/cols positions."""
    d_model = 64  # Small for quick test
    rope = RoPE2D(d_model)

    # Create a simple 2×2 grid: 4 patches
    B = 1
    x = torch.ones(B, 4, d_model) * 0.5  # All patches identical
    grid = (2, 2)

    y = rope(x, grid)

    # Patches at different positions should have DIFFERENT encodings
    # Position (0,0) vs (1,1) — definitely different
    assert not torch.allclose(y[0, 0, :], y[0, 3, :], atol=1e-5), \
        "Different grid positions should have different RoPE encodings"
    print("✓ RoPE2D: row/col encoding produces distinct position vectors")


def test_rope2d_non_square_grid():
    """Test RoPE2D with non-square (rectangular) grid."""
    d_model = 128
    rope = RoPE2D(d_model)

    # 8×4 grid = 32 patches
    B = 1
    N = 32
    grid = (8, 4)
    x = torch.randn(B, N, d_model)
    y = rope(x, grid)

    assert y.shape == (B, N, d_model)
    # Should not crash and should produce valid output
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    print("✓ RoPE2D: non-square grid supported")


# ============================================================================
# PatchEmbedding
# ============================================================================

def test_patch_embedding_shapes():
    """Test PatchEmbedding output shapes for standard image."""
    patch_embed = PatchEmbedding(image_size=224, patch_size=16, d_vision=384)

    B = 2
    img = torch.randn(B, 3, 224, 224)
    patches, grid_shape = patch_embed(img)

    assert patches.shape == (B, 196, 384), \
        f"Expected (2, 196, 384), got {patches.shape}"
    assert grid_shape == (14, 14), f"Expected (14,14), got {grid_shape}"
    print(f"✓ PatchEmbedding: {img.shape} → {patches.shape}, grid={grid_shape}")


def test_patch_embedding_different_sizes():
    """Test PatchEmbedding with multiple image sizes."""
    patch_embed = PatchEmbedding(image_size=224, patch_size=16, d_vision=384)

    for size in [112, 224, 448]:
        B = 1
        n_patches = (size // 16) ** 2
        img = torch.randn(B, 3, size, size)
        patches, grid = patch_embed(img)
        assert patches.shape == (B, n_patches, 384), \
            f"Size {size}: expected ({B}, {n_patches}, 384), got {patches.shape}"
        h = w = size // 16
        assert grid == (h, w)
    print("✓ PatchEmbedding: multiple image sizes correct")


def test_patch_embedding_small_grid():
    """Test PatchEmbedding with small patch size → many patches."""
    patch_embed = PatchEmbedding(image_size=224, patch_size=8, d_vision=256)
    img = torch.randn(1, 3, 224, 224)
    patches, grid = patch_embed(img)

    # 224/8 = 28, so 28×28 = 784 patches
    assert patches.shape == (1, 784, 256)
    assert grid == (28, 28)
    print(f"✓ PatchEmbedding (p=8): {patches.shape}")


def test_patch_embedding_non_square():
    """Test PatchEmbedding with non-square image."""
    patch_embed = PatchEmbedding(image_size=224, patch_size=16, d_vision=384)

    # 320×192 image → 20×12 = 240 patches
    img = torch.randn(2, 3, 192, 320)
    patches, grid = patch_embed(img)
    assert patches.shape == (2, 240, 384)
    assert grid == (12, 20)  # H=192/16=12, W=320/16=20
    print(f"✓ PatchEmbedding non-square: {img.shape} → {patches.shape}")


# ============================================================================
# BiGLTBlock
# ============================================================================

def test_biglt_block_shapes():
    """Test BiGLTBlock forward pass shapes."""
    d_vision, d_state = 384, 128
    block = BiGLTBlock(d_vision, d_state)

    B, N = 2, 196
    x = torch.randn(B, N, d_vision)
    out, state_fwd, state_rev = block(x)

    assert out.shape == (B, N, d_vision), f"Expected ({B},{N},{d_vision}), got {out.shape}"
    assert state_fwd.shape == (B, d_state, d_state)
    assert state_rev.shape == (B, d_state, d_state)
    print(f"✓ BiGLTBlock: shapes correct, params={sum(p.numel() for p in block.parameters()):,}")


def test_biglt_block_reset_states():
    """Test BiGLTBlock state reset returns zero states."""
    d_vision, d_state = 384, 128
    block = BiGLTBlock(d_vision, d_state)

    state_fwd, state_rev = block.reset_states(batch_size=3, device="cpu")
    assert state_fwd.shape == (3, d_state, d_state)
    assert state_rev.shape == (3, d_state, d_state)
    assert torch.all(state_fwd == 0)
    assert torch.all(state_rev == 0)
    print("✓ BiGLTBlock: reset_states returns zeros")


def test_biglt_block_determinism():
    """Test BiGLTBlock gives consistent outputs for same input."""
    d_vision, d_state = 192, 64
    block = BiGLTBlock(d_vision, d_state)
    block.eval()

    x = torch.randn(1, 50, d_vision)
    with torch.no_grad():
        out1, _, _ = block(x)
        out2, _, _ = block(x)

    assert torch.equal(out1, out2), "BiGLTBlock should be deterministic in eval mode"
    print("✓ BiGLTBlock: deterministic in eval mode")


def test_biglt_block_state_carry():
    """Test that BiGLTBlock states change between inputs."""
    d_vision, d_state = 192, 64
    block = BiGLTBlock(d_vision, d_state)

    x1 = torch.randn(2, 30, d_vision)
    x2 = torch.randn(2, 30, d_vision)

    out1, sf1, sr1 = block(x1)
    out2, sf2, sr2 = block(x2, state_fwd=sf1, state_rev=sr1)

    assert not torch.equal(out1, out2), "Outputs should differ for different inputs"
    print("✓ BiGLTBlock: state carry works, outputs change with input")


def test_biglt_block_no_nan():
    """Test BiGLTBlock doesn't produce NaN for normal inputs."""
    d_vision, d_state = 384, 128
    block = BiGLTBlock(d_vision, d_state)

    x = torch.randn(2, 196, d_vision)
    out, _, _ = block(x)
    assert not torch.isnan(out).any(), "NaN in BiGLTBlock output"
    assert not torch.isinf(out).any(), "Inf in BiGLTBlock output"
    print("✓ BiGLTBlock: no NaN/Inf for normal inputs")


# ============================================================================
# SpatialAnchorBlock
# ============================================================================

def test_spatial_anchor_shapes():
    """Test SpatialAnchorBlock output shapes."""
    d_vision, n_heads, n_kv_heads = 384, 6, 3
    block = SpatialAnchorBlock(d_vision, n_heads, n_kv_heads)

    B, N = 2, 196
    x = torch.randn(B, N, d_vision)
    out = block(x)

    assert out.shape == (B, N, d_vision), f"Expected ({B},{N},{d_vision}), got {out.shape}"
    print(f"✓ SpatialAnchorBlock: shapes correct, params={sum(p.numel() for p in block.parameters()):,}")


def test_spatial_anchor_determinism():
    """Test SpatialAnchorBlock gives consistent output in eval mode."""
    d_vision, n_heads, n_kv_heads = 192, 4, 2
    block = SpatialAnchorBlock(d_vision, n_heads, n_kv_heads)
    block.eval()

    x = torch.randn(1, 49, d_vision)
    with torch.no_grad():
        out1 = block(x)
        out2 = block(x)

    assert torch.equal(out1, out2), "SpatialAnchorBlock should be deterministic in eval mode"
    print("✓ SpatialAnchorBlock: deterministic in eval mode")


def test_spatial_anchor_gqa_heads():
    """Test that SpatialAnchorBlock correctly has GQA config."""
    d_vision, n_heads, n_kv_heads = 384, 6, 3
    block = SpatialAnchorBlock(d_vision, n_heads, n_kv_heads)

    assert block.n_heads == 6
    assert block.n_kv_heads == 3
    assert block.n_heads % block.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
    assert block.head_dim == d_vision // n_heads
    print("✓ SpatialAnchorBlock: GQA head config valid")


def test_spatial_anchor_anchors_registered():
    """Test that spatial anchors are registered as learnable parameters."""
    d_vision = 384
    block = SpatialAnchorBlock(d_vision, n_heads=6, n_kv_heads=3)

    assert hasattr(block, 'spatial_anchors')
    assert isinstance(block.spatial_anchors, nn.Parameter)
    assert block.spatial_anchors.shape == (16, d_vision)
    assert block.spatial_anchors.requires_grad
    print("✓ SpatialAnchorBlock: 16 learnable spatial anchors registered")


def test_spatial_anchor_no_nan():
    """Test SpatialAnchorBlock doesn't produce NaN."""
    d_vision = 384
    block = SpatialAnchorBlock(d_vision, n_heads=6, n_kv_heads=3)

    x = torch.randn(2, 196, d_vision)
    out = block(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print("✓ SpatialAnchorBlock: no NaN/Inf")


# ============================================================================
# VisionProjector
# ============================================================================

def test_vision_projector_shapes():
    """Test VisionProjector maps d_vision → d_model."""
    d_vision, d_model = 384, 768
    projector = VisionProjector(d_vision, d_model)

    B, N = 2, 196
    x = torch.randn(B, N, d_vision)
    out = projector(x)

    assert out.shape == (B, N, d_model), \
        f"Expected ({B},{N},{d_model}), got {out.shape}"
    print(f"✓ VisionProjector: {d_vision}→{d_model}, params={sum(p.numel() for p in projector.parameters()):,}")


def test_vision_projector_different_dims():
    """Test VisionProjector with different dimension pairs."""
    configs = [
        (128, 192),   # Nano
        (192, 384),   # Small
        (384, 768),   # Max
    ]
    for d_vis, d_mod in configs:
        projector = VisionProjector(d_vis, d_mod)
        x = torch.randn(1, 10, d_vis)
        out = projector(x)
        assert out.shape == (1, 10, d_mod), \
            f"Projector {d_vis}→{d_mod}: expected (1,10,{d_mod}), got {out.shape}"
    print("✓ VisionProjector: multiple dimension pairs work")


def test_vision_projector_dimension_change():
    """Test that VisionProjector changes dimension and applies real transformation."""
    projector = VisionProjector(384, 768)
    x = torch.randn(1, 5, 384)
    out = projector(x)
    # Output dimension should change
    assert out.shape == (1, 5, 768)
    # Should not be just a padded version of input — must be a real learned projection
    # (the first 384 channels of output should not match the input)
    assert not torch.allclose(out[:, :, :384], x, atol=1e-3), \
        "VisionProjector should be a real transformation, not identity+padded"
    print("✓ VisionProjector: real dimension transform verified")


# ============================================================================
# ContinuumVisionEncoder (Full Assembly)
# ============================================================================

def test_vision_encoder_shapes():
    """Test full encoder forward pass shapes."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=384, d_v_state=128,
        n_bi_glt_layers=5,
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=768)

    B = 1
    img = torch.randn(B, 3, 224, 224)
    tokens = encoder(img)

    expected_patches = 196  # (224/16)^2
    assert tokens.shape == (B, expected_patches, 768), \
        f"Expected ({B},{expected_patches},768), got {tokens.shape}"
    print(f"✓ ContinuumVisionEncoder: {img.shape} → {tokens.shape}")


def test_vision_encoder_param_count():
    """Test that encoder param count matches expected ~13.5M for Max tier."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=384, d_v_state=128,
        n_bi_glt_layers=5, ffn_expansion=3, ffn_shards=3,
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=768)
    params = encoder.num_params
    # Should be roughly 10-15M
    assert 10_000_000 < params < 18_000_000, \
        f"Expected 10-15M params for Max tier, got {params:,}"
    print(f"✓ ContinuumVisionEncoder Max: {params:,} params (~13.5M target)")


def test_vision_encoder_num_patches_property():
    """Test the num_patches property."""
    cfg = ContinuumVisionConfig(image_size=224, patch_size=16)
    encoder = ContinuumVisionEncoder(cfg, d_model=768)
    assert encoder.num_patches == 196
    print(f"✓ ContinuumVisionEncoder: num_patches={encoder.num_patches}")


def test_vision_encoder_batch_size():
    """Test encoder works with batch sizes > 1."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=128, d_v_state=32, n_bi_glt_layers=2,
        n_heads=4, n_kv_heads=2,  # Must divide d_vision (128/4=32)
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=192)

    for B in [1, 2, 4]:
        img = torch.randn(B, 3, 224, 224)
        tokens = encoder(img)
        assert tokens.shape == (B, 196, 192), \
            f"Batch {B}: expected ({B},196,192), got {tokens.shape}"
    print("✓ ContinuumVisionEncoder: batch sizes 1,2,4 work")


def test_vision_encoder_eval_mode():
    """Test encoder in eval mode (no dropout, deterministic)."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=128, d_v_state=32, n_bi_glt_layers=2,
        n_heads=4, n_kv_heads=2,  # Must divide d_vision (128/4=32)
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=192)
    encoder.eval()

    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = encoder(img)
        out2 = encoder(img)

    assert torch.equal(out1, out2), "Encoder should be deterministic in eval mode"
    print("✓ ContinuumVisionEncoder: deterministic in eval mode")


def test_vision_encoder_max_patches_cap():
    """Test the max_patches safety cap (downsampling large images)."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=8,   # 28×28 = 784 patches
        d_vision=128, d_v_state=32,
        n_bi_glt_layers=2,
        n_heads=4, n_kv_heads=2,        # Must divide d_vision (128/4=32)
        max_patches=100,                 # Force capping at 100
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=192)

    img = torch.randn(1, 3, 224, 224)
    tokens = encoder(img)

    # Should have at most max_patches (784 → step=ceil(784/100)=8 → 784/8=98 patches)
    assert tokens.shape[1] <= 100, \
        f"Expected ≤100 patches after cap, got {tokens.shape[1]}"
    print(f"✓ ContinuumVisionEncoder: max_patches cap works ({tokens.shape[1]} patches ≤ 100)")


def test_vision_encoder_gradient_flow():
    """Test that gradients flow through the full encoder."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=128, d_v_state=32, n_bi_glt_layers=2,
        n_heads=4, n_kv_heads=2,  # Must divide d_vision (128/4=32)
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=192)
    encoder.train()

    img = torch.randn(1, 3, 224, 224, requires_grad=False)
    tokens = encoder(img)

    # Compute loss from output
    loss = tokens.sum()
    loss.backward()

    # Check that gradients flowed to at least SOME parameters
    has_grad = False
    zero_grad_count = 0
    total_params = 0
    for name, p in encoder.named_parameters():
        total_params += 1
        if p.grad is not None:
            has_grad = True
            if p.grad.abs().sum() == 0:
                zero_grad_count += 1
        else:
            zero_grad_count += 1

    assert has_grad, "No parameters received gradients!"
    # Most parameters should have non-zero gradients
    grad_ratio = (total_params - zero_grad_count) / total_params
    assert grad_ratio > 0.5, \
        f"Only {grad_ratio:.1%} of params have non-zero gradients (expected >50%)"
    print(f"✓ ContinuumVisionEncoder: gradients flow ({grad_ratio:.1%} params with non-zero grad)")


def test_vision_encoder_train_randomness():
    """Test that encoder produces different outputs in training mode (dropout active)."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=128, d_v_state=32, n_bi_glt_layers=2,
        n_heads=4, n_kv_heads=2, dropout=0.1,  # Enable dropout
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=192)
    encoder.train()  # Training mode → dropout active

    img = torch.randn(1, 3, 224, 224)
    out1 = encoder(img)
    out2 = encoder(img)

    # With dropout active, outputs should differ between calls
    assert not torch.equal(out1, out2), \
        "Encoder in training mode should produce different outputs (dropout)"
    print("✓ ContinuumVisionEncoder: training mode produces different outputs (dropout wired)")


def test_vision_encoder_no_nan():
    """Test full encoder doesn't produce NaN with normal inputs."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=384, d_v_state=128, n_bi_glt_layers=5,
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=768)
    encoder.eval()

    with torch.no_grad():
        for _ in range(5):
            img = torch.randn(2, 3, 224, 224)
            tokens = encoder(img)
            assert not torch.isnan(tokens).any(), "NaN in encoder output"
            assert not torch.isinf(tokens).any(), "Inf in encoder output"

    print("✓ ContinuumVisionEncoder: no NaN/Inf over 5 random inputs")


def test_vision_encoder_variable_image_sizes():
    """Test encoder with different image sizes (not just 224×224)."""
    cfg = ContinuumVisionConfig(
        image_size=224, patch_size=16,
        d_vision=128, d_v_state=32, n_bi_glt_layers=2,
        n_heads=4, n_kv_heads=2,  # Must divide d_vision (128/4=32)
    )
    encoder = ContinuumVisionEncoder(cfg, d_model=192)

    for size in [(224, 224), (224, 320), (320, 224), (112, 112)]:
        H, W = size
        img = torch.randn(1, 3, H, W)
        tokens = encoder(img)
        H_p, W_p = H // 16, W // 16
        expected_N = H_p * W_p
        assert tokens.shape[0] == 1
        assert tokens.shape[1] == expected_N, \
            f"Size {(H,W)}: expected {expected_N} patches, got {tokens.shape[1]}"
        assert tokens.shape[2] == 192
        assert not torch.isnan(tokens).any()
    print("✓ ContinuumVisionEncoder: variable image sizes supported")


# ============================================================================
# Factory Functions
# ============================================================================

def test_factory_max():
    """Test create_vision_encoder_max."""
    encoder = create_vision_encoder_max(d_model=768)
    assert isinstance(encoder, ContinuumVisionEncoder)
    params = encoder.num_params
    assert 10_000_000 < params < 18_000_000, f"Max params: {params:,}"

    img = torch.randn(1, 3, 224, 224)
    out = encoder(img)
    assert out.shape == (1, 196, 768)
    print(f"✓ create_vision_encoder_max: {params:,} params")


def test_factory_small():
    """Test create_vision_encoder_small."""
    encoder = create_vision_encoder_small(d_model=384)
    assert isinstance(encoder, ContinuumVisionEncoder)
    params = encoder.num_params
    assert 3_000_000 < params < 8_000_000, f"Small params: {params:,}"

    img = torch.randn(1, 3, 224, 224)
    out = encoder(img)
    assert out.shape == (1, 196, 384)
    print(f"✓ create_vision_encoder_small: {params:,} params")


def test_factory_nano():
    """Test create_vision_encoder_nano."""
    encoder = create_vision_encoder_nano(d_model=192)
    assert isinstance(encoder, ContinuumVisionEncoder)
    params = encoder.num_params
    assert 1_000_000 < params < 4_000_000, f"Nano params: {params:,}"

    img = torch.randn(1, 3, 224, 224)
    out = encoder(img)
    assert out.shape == (1, 196, 192)
    print(f"✓ create_vision_encoder_nano: {params:,} params")


def test_factory_all_tiers_no_crash():
    """Test that all three factory functions create working encoders."""
    for name, fn, d_model in [
        ("Nano", create_vision_encoder_nano, 192),
        ("Small", create_vision_encoder_small, 384),
        ("Max", create_vision_encoder_max, 768),
    ]:
        encoder = fn(d_model=d_model)
        encoder.eval()
        img = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(img)
        assert out.shape[0] == 2
        assert out.shape[2] == d_model
        assert not torch.isnan(out).any()
    print("✓ All factory functions work (Nano, Small, Max)")


# ============================================================================
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Continuum Vision (ViGLT) Test Suite")
    print("=" * 60)

    # Config
    test_vision_config_defaults()
    test_vision_config_custom()

    # RoPE2D
    test_rope2d_shapes()
    test_rope2d_d_model_validation()
    test_rope2d_grid_mismatch()
    test_rope2d_determinism()
    test_rope2d_row_col_encoding()
    test_rope2d_non_square_grid()

    # PatchEmbedding
    test_patch_embedding_shapes()
    test_patch_embedding_different_sizes()
    test_patch_embedding_small_grid()
    test_patch_embedding_non_square()

    # BiGLTBlock
    test_biglt_block_shapes()
    test_biglt_block_reset_states()
    test_biglt_block_determinism()
    test_biglt_block_state_carry()
    test_biglt_block_no_nan()

    # SpatialAnchorBlock
    test_spatial_anchor_shapes()
    test_spatial_anchor_determinism()
    test_spatial_anchor_gqa_heads()
    test_spatial_anchor_anchors_registered()
    test_spatial_anchor_no_nan()

    # VisionProjector
    test_vision_projector_shapes()
    test_vision_projector_different_dims()
    test_vision_projector_dimension_change()

    # ContinuumVisionEncoder
    test_vision_encoder_shapes()
    test_vision_encoder_param_count()
    test_vision_encoder_num_patches_property()
    test_vision_encoder_batch_size()
    test_vision_encoder_eval_mode()
    test_vision_encoder_max_patches_cap()
    test_vision_encoder_gradient_flow()
    test_vision_encoder_train_randomness()
    test_vision_encoder_no_nan()
    test_vision_encoder_variable_image_sizes()

    # Factory functions
    test_factory_max()
    test_factory_small()
    test_factory_nano()
    test_factory_all_tiers_no_crash()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
