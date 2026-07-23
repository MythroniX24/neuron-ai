"""
Continuum Vision (ViGLT) — Custom Vision Encoder for Continuum SLM.

A ground-up vision encoder built from the SAME architectural primitives
as the language model: GLT (Gated Linear Trace), GatedShardFFN, RMSNorm,
and Anchor Attention. Uses bidirectional GLT for spatial processing.

Architecture:
  1. Patch Convolution — Conv2D 16×16 stride 16 → d_vision features
  2. 2D RoPE — Continuous rotary positional encoding for any aspect ratio
  3. ViGLT Blocks (×5) — Bidirectional GLT + GatedShardFFN
  4. Spatial Anchor (×1) — Anchor Attention for global cross-referencing
  5. Vision Projector — GatedShardFFN (d_vision → d_model)

Total: ~13.5M extra parameters, ~115M multimodal total.

Design principles (from Continuum architecture):
  - No growing cache, O(1) memory per vision patch
  - Bidirectional GLT = same recurrence, two directions
  - Conditional compute via gating at all levels
  - Native variable image size support (no fixed grid)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from continuum.model.layers import GLTLayer, GatedShardFFN, RMSNorm


# ============================================================================
# Vision Configuration
# ============================================================================

class ContinuumVisionConfig:
    """Configuration for the Continuum Vision encoder."""

    def __init__(
        self,
        # Image dimensions
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,

        # Vision model dimensions
        d_vision: int = 384,
        d_v_state: int = 128,
        n_vision_layers: int = 6,       # 5 Bi-GLT + 1 Anchor
        n_bi_glt_layers: int = 5,
        n_anchor_layers: int = 1,

        # Anchor attention (spatial reference)
        n_heads: int = 6,
        n_kv_heads: int = 3,
        spatial_window: int = 49,        # 7×7 spatial window

        # FFN
        ffn_expansion: int = 3,
        ffn_shards: int = 3,

        # RoPE
        rope_theta: float = 10000.0,

        # Normalization
        dropout: float = 0.0,

        # Max patches (mobile safety)
        max_patches: int = 256,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_vision = d_vision
        self.d_v_state = d_v_state
        self.n_vision_layers = n_vision_layers
        self.n_bi_glt_layers = n_bi_glt_layers
        self.n_anchor_layers = n_anchor_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.spatial_window = spatial_window
        self.ffn_expansion = ffn_expansion
        self.ffn_shards = ffn_shards
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.max_patches = max_patches


# ============================================================================
# 2D Rotary Position Embedding (RoPE)
# ============================================================================

class RoPE2D(nn.Module):
    """
    Continuous 2D Rotary Position Embedding for spatial patch grids.

    Encodes (row, col) position into the feature dimension using
    rotary embeddings, allowing the Bi-GLT to distinguish spatial
    relationships (up/down/left/right) during the 1D scan.

    Args:
        d_model: Feature dimension (must be divisible by 4 for 2D split)
        theta: Base frequency (default 10000.0, standard RoPE)
    """

    def __init__(self, d_model: int, theta: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4 for 2D RoPE"
        self.d_model = d_model
        self.theta = theta

        # Half for row encoding, half for column encoding
        d_half = d_model // 2

        # Frequency bands
        freqs = 1.0 / (theta ** (torch.arange(0, d_half // 2, dtype=torch.float32) * 2.0 / d_half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Apply 2D RoPE to patch features.

        Args:
            x: Patch features [B, num_patches, d_model]
            grid_shape: (H_patches, W_patches) — the 2D grid layout

        Returns:
            x with 2D RoPE applied: same shape [B, num_patches, d_model]
        """
        H, W = grid_shape
        B, N, D = x.shape
        assert N == H * W, f"Patches ({N}) must match grid ({H}×{W}={H*W})"
        device = x.device
        dtype = x.dtype

        # Build position indices for each patch in snake-scan order
        # Row-major flattened: (0,0), (0,1), ..., (0,W-1), (1,0), ..., (H-1,W-1)
        rows = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1)  # [N]
        cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1)  # [N]

        # Compute rotary angles for rows and columns
        # Each half of the embedding gets a different frequency
        d_half = D // 2

        # Row angles: freqs * row_pos — applied to first half
        row_angles = rows.float().unsqueeze(1) * self.freqs.unsqueeze(0)  # [N, d_half//2]
        row_cos = torch.cos(row_angles).unsqueeze(0)  # [1, N, d_half//2]
        row_sin = torch.sin(row_angles).unsqueeze(0)

        # Column angles: freqs * col_pos — applied to second half
        col_angles = cols.float().unsqueeze(1) * self.freqs.unsqueeze(0)  # [N, d_half//2]
        col_cos = torch.cos(col_angles).unsqueeze(0)
        col_sin = torch.sin(col_angles).unsqueeze(0)

        # Split x into two halves: row half and col half
        x_row = x[:, :, :d_half]       # [B, N, d_half]
        x_col = x[:, :, d_half:]       # [B, N, d_half]

        # Apply rotary: rotate pairs within each half
        # For row half: rotate adjacent pairs (even, odd)
        x_row_pairs = x_row.reshape(B, N, -1, 2)  # [B, N, d_half//2, 2]
        x_r0, x_r1 = x_row_pairs[..., 0], x_row_pairs[..., 1]

        x_r0_rot = x_r0 * row_cos - x_r1 * row_sin
        x_r1_rot = x_r1 * row_cos + x_r0 * row_sin
        x_row_rot = torch.stack([x_r0_rot, x_r1_rot], dim=-1).reshape(B, N, d_half)

        # For col half: same rotary pattern
        x_col_pairs = x_col.reshape(B, N, -1, 2)
        x_c0, x_c1 = x_col_pairs[..., 0], x_col_pairs[..., 1]

        x_c0_rot = x_c0 * col_cos - x_c1 * col_sin
        x_c1_rot = x_c1 * col_cos + x_c0 * col_sin
        x_col_rot = torch.stack([x_c0_rot, x_c1_rot], dim=-1).reshape(B, N, d_half)

        return torch.cat([x_row_rot, x_col_rot], dim=-1)


# ============================================================================
# Patch Embedding (Conv Stem)
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Conv2D-based patch embedding.

    Converts an image [B, C, H, W] → patch features [B, N, d_vision]
    using a single strided convolution (no MLP needed).

    Args:
        image_size: Input image size (assumed square, e.g. 224)
        patch_size: Patch edge size (e.g. 16)
        in_channels: Input channels (3 for RGB)
        d_vision: Output feature dimension
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_vision: int = 384,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_vision = d_vision

        # Single conv: patch extraction + projection in one operation
        # kernel=patch_size, stride=patch_size → exactly non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels, d_vision,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )

        # Initialize
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: [B, C, H, W] input image

        Returns:
            patches: [B, N, d_vision] flattened patch features
            grid_shape: (H_patches, W_patches)
        """
        B, C, H, W = x.shape

        # Convolutional patch embedding
        x = self.proj(x)  # [B, d_vision, H_patches, W_patches]

        H_p, W_p = x.shape[2], x.shape[3]
        N = H_p * W_p

        # Flatten spatial dimensions: [B, d_vision, H, W] → [B, N, d_vision]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d_vision]

        return x, (H_p, W_p)


# ============================================================================
# Bidirectional GLT Block
# ============================================================================

class BiGLTBlock(nn.Module):
    """
    A single bidirectional GLT block.

    Contains:
      - Forward GLT: processes patches left-to-right (standard 1D scan)
      - Reverse GLT: processes patches right-to-left (flipped scan)
      - GatedShardFFN: after combining both directions

    The two GLT directions have DECOUPLED states — no shared recurrence.
    Outputs are added/averaged, then passed through FFN + residual.

    Args:
        d_vision: Feature dimension
        d_state: GLT internal state dimension
        dropout: Dropout rate
    """

    def __init__(self, d_vision: int, d_state: int, dropout: float = 0.0):
        super().__init__()
        self.d_vision = d_vision
        self.d_state = d_state

        # Forward GLT
        self.glt_fwd = GLTLayer(d_vision, d_state, dropout)

        # Reverse GLT (separate weights, no shared recurrence)
        self.glt_rev = GLTLayer(d_vision, d_state, dropout)

        # Pre-norm for each direction
        self.norm_fwd = RMSNorm(d_vision)
        self.norm_rev = RMSNorm(d_vision)

        # FFN after merging directions
        self.ffn = GatedShardFFN(d_vision, expansion=3, num_shards=3, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        state_fwd: Optional[torch.Tensor] = None,
        state_rev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, d_vision] patch features
            state_fwd: Forward GLT state [B, d_state, d_state] or None
            state_rev: Reverse GLT state [B, d_state, d_state] or None

        Returns:
            out: [B, N, d_vision]
            new_state_fwd: Updated forward state
            new_state_rev: Updated reverse state
        """
        residual = x

        # Forward pass (left → right)
        x_fwd, new_state_fwd = self.glt_fwd.forward_sequence(
            self.norm_fwd(x), state_fwd
        )

        # Reverse pass (right → left): flip sequence, process, flip back
        x_rev_flipped = torch.flip(self.norm_rev(x), dims=[1])
        x_rev_out, new_state_rev = self.glt_rev.forward_sequence(
            x_rev_flipped, state_rev
        )
        x_rev = torch.flip(x_rev_out, dims=[1])  # flip back to original order

        # Combine directions: average (stable, doesn't amplify)
        x_combined = (x_fwd + x_rev) * 0.5

        # FFN with residual
        out = self.ffn(x_combined + residual)

        return out, new_state_fwd, new_state_rev

    def reset_states(self, batch_size: int = 1, device: str = "cpu",
                     dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create fresh zero states for both directions."""
        state_fwd = self.glt_fwd.reset_state(batch_size, device, dtype)
        state_rev = self.glt_rev.reset_state(batch_size, device, dtype)
        return state_fwd, state_rev


# ============================================================================
# Spatial Anchor Block
# ============================================================================

class SpatialAnchorBlock(nn.Module):
    """
    Lightweight spatial anchor attention block.

    Uses a small set of learnable spatial anchor tokens + a local
    spatial window to cross-reference distant patch regions.
    This is the only attention-based operation in the vision encoder.

    Uses standard MHA (not AnchorAttention, which requires window cache
    management) since this processes the full image in one pass.

    Args:
        d_vision: Feature dimension
        n_heads: Number of attention heads
        n_kv_heads: KV heads for GQA
        window_size: Spatial window for local attention
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_vision: int = 384,
        n_heads: int = 6,
        n_kv_heads: int = 3,
        window_size: int = 49,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_vision = d_vision
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_vision // n_heads
        self.window_size = window_size

        assert d_vision % n_heads == 0

        # Fused QKV projection
        q_dim = n_heads * self.head_dim
        kv_dim = n_kv_heads * self.head_dim
        self.W_qkv = nn.Linear(d_vision, q_dim + 2 * kv_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(q_dim, d_vision, bias=False)

        # Learnable spatial anchor tokens (global context references)
        self.n_anchors = 16  # Fixed, small
        self.spatial_anchors = nn.Parameter(
            torch.randn(self.n_anchors, d_vision) * 0.02
        )

        # Norms
        self.norm = RMSNorm(d_vision)
        self.norm_q = RMSNorm(d_vision)
        self.norm_kv = RMSNorm(d_vision)

        # FFN
        self.ffn = GatedShardFFN(d_vision, expansion=3, num_shards=3, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_qkv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_o.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_vision] patch features

        Returns:
            out: [B, N, d_vision]
        """
        B, N, D = x.shape
        residual = x
        x_norm = self.norm(x)

        # Project queries from patches, K/V from patches + anchors
        qkv = self.W_qkv(x_norm)
        q, k_patches, v_patches = qkv.split(
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.view(B, N, self.n_heads, self.head_dim)

        # K/V from spatial anchors (broadcast to batch)
        anchor_qkv = self.W_qkv(self.norm_kv(
            self.spatial_anchors.unsqueeze(0).expand(B, -1, -1)
        ))
        _, k_anchors, v_anchors = anchor_qkv.split(
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        k_anchors = k_anchors.view(B, self.n_anchors, self.n_kv_heads, self.head_dim)
        v_anchors = v_anchors.view(B, self.n_anchors, self.n_kv_heads, self.head_dim)

        k_patches = k_patches.view(B, N, self.n_kv_heads, self.head_dim)
        v_patches = v_patches.view(B, N, self.n_kv_heads, self.head_dim)

        # Concatenate: anchors first, then patches
        k = torch.cat([k_anchors, k_patches], dim=1)  # [B, N_anchors+N, n_kv_heads, hd]
        v = torch.cat([v_anchors, v_patches], dim=1)

        # Use SDPA with GQA (repeat KV heads)
        n_groups = self.n_heads // self.n_kv_heads
        k_gqa = k.repeat_interleave(n_groups, dim=2)
        v_gqa = v.repeat_interleave(n_groups, dim=2)

        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # [B, n_heads, N, hd]
            k_gqa.transpose(1, 2),  # [B, n_heads, T, hd]
            v_gqa.transpose(1, 2),
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(B, N, self.n_heads * self.head_dim)

        # Output projection
        output = self.W_o(output)

        # Residual + FFN
        output = self.ffn(output + residual)

        return output


# ============================================================================
# Vision Projector (d_vision → d_model)
# ============================================================================

class VisionProjector(nn.Module):
    """
    Projects vision features from d_vision to the language model's d_model.

    Uses a 2-layer GatedShardFFN (same primitive as the rest of the architecture)
    for consistent inductive bias. SwiGLU non-linearity maps vision features
    into the text token manifold.

    Args:
        d_vision: Vision feature dimension (384)
        d_model: Language model dimension (768 for Max)
    """

    def __init__(self, d_vision: int = 384, d_model: int = 768):
        super().__init__()
        self.d_vision = d_vision
        self.d_model = d_model

        # Two-layer GatedShardFFN: d_vision → d_model → d_model
        # Layer 1: expand vision features
        self.proj_in = GatedShardFFN(d_vision, expansion=2, num_shards=2)
        # Layer 2: refine to text space
        self.proj_out = GatedShardFFN(d_vision, expansion=2, num_shards=2)

        # Final linear: d_vision → d_model
        self.linear = nn.Linear(d_vision, d_model, bias=False)

        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_vision] vision features

        Returns:
            out: [B, N, d_model] projected features (ready for LLM)
        """
        x = self.proj_in(x)
        x = self.proj_out(x)
        x = self.linear(x)
        return x


# ============================================================================
# Continuum Vision Encoder (Full Assembly)
# ============================================================================

class ContinuumVisionEncoder(nn.Module):
    """
    Complete Continuum Vision encoder (ViGLT).

    Pipeline:
      Image → PatchEmbedding → 2D RoPE → 5× BiGLT → 1× SpatialAnchor → Projector

    Outputs [B, N_patches, d_model] features ready to be prepended
    to text tokens in the main ContinuumModel.

    Args:
        config: ContinuumVisionConfig
        d_model: Language model's hidden dimension (for projector output)
    """

    def __init__(self, config: ContinuumVisionConfig, d_model: int = 768):
        super().__init__()
        self.config = config
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            d_vision=config.d_vision,
        )

        # 2D RoPE
        self.rope = RoPE2D(config.d_vision, theta=config.rope_theta)

        # Bi-GLT blocks
        self.bi_glt_blocks = nn.ModuleList([
            BiGLTBlock(config.d_vision, config.d_v_state, config.dropout)
            for _ in range(config.n_bi_glt_layers)
        ])

        # Spatial anchor block
        self.spatial_anchor = SpatialAnchorBlock(
            d_vision=config.d_vision,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            window_size=config.spatial_window,
            dropout=config.dropout,
        )

        # Final norm before projection
        self.final_norm = RMSNorm(config.d_vision)

        # Vision → Language projector
        self.projector = VisionProjector(config.d_vision, d_model)

        # Count parameters
        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into vision tokens.

        Args:
            pixel_values: [B, C, H, W] normalized image tensor
                          Expected range: [0, 1] or [-1, 1]

        Returns:
            vision_tokens: [B, N_patches, d_model] tokens ready for LLM
        """
        B = pixel_values.shape[0]
        config = self.config

        # 1. Patch embedding
        x, grid_shape = self.patch_embed(pixel_values)  # [B, N, d_vision]
        N = x.shape[1]

        # Safety: cap patches for mobile
        if N > config.max_patches:
            # Downsample by taking every step-th patch (simple, effective)
            # This preserves original aspect ratio better than resizing
            step = math.ceil(N / config.max_patches)
            x = x[:, ::step, :]
            N = x.shape[1]
            # Adjust grid shape: preserve aspect ratio, trim to exact H_p*W_p
            H_p_orig, W_p_orig = grid_shape
            ratio = W_p_orig / max(H_p_orig, 1)
            H_p = max(1, round(math.sqrt(N / ratio)))
            W_p = N // H_p
            # Trim patches to exact grid product (lose at most H_p-1 patches)
            exact_N = H_p * W_p
            x = x[:, :exact_N, :]
            N = exact_N
            grid_shape = (H_p, W_p)

        # 2. Apply 2D RoPE
        x = self.rope(x, grid_shape)

        # 3. Bi-GLT blocks (bidirectional recurrent processing)
        for block in self.bi_glt_blocks:
            x, _, _ = block(x)  # States are transient, not needed after encoding

        # 4. Spatial anchor attention (global cross-referencing)
        x = self.spatial_anchor(x)

        # 5. Final norm + projection to d_model
        x = self.final_norm(x)
        x = self.projector(x)  # [B, N, d_model]

        return x

    @property
    def num_params(self) -> int:
        return self._n_params

    @property
    def num_patches(self) -> int:
        """Number of patches for default image size."""
        return (self.config.image_size // self.config.patch_size) ** 2


# ============================================================================
# Factory functions
# ============================================================================

def create_vision_encoder_max(d_model: int = 768) -> ContinuumVisionEncoder:
    """Create vision encoder for Continuum-Max (~13.5M params)."""
    config = ContinuumVisionConfig(
        image_size=224,
        patch_size=16,
        d_vision=384,
        d_v_state=128,
        n_bi_glt_layers=5,
        n_anchor_layers=1,
        n_heads=6,
        n_kv_heads=3,
        spatial_window=49,
        ffn_expansion=3,
        ffn_shards=3,
    )
    return ContinuumVisionEncoder(config, d_model)


def create_vision_encoder_small(d_model: int = 384) -> ContinuumVisionEncoder:
    """Create vision encoder for Continuum-Small (~5M params)."""
    config = ContinuumVisionConfig(
        image_size=224,
        patch_size=16,
        d_vision=192,
        d_v_state=64,
        n_bi_glt_layers=3,
        n_anchor_layers=1,
        n_heads=4,
        n_kv_heads=2,
        spatial_window=25,
        ffn_expansion=3,
        ffn_shards=2,
    )
    return ContinuumVisionEncoder(config, d_model)


def create_vision_encoder_nano(d_model: int = 192) -> ContinuumVisionEncoder:
    """Create vision encoder for Continuum-Nano (~2M params)."""
    config = ContinuumVisionConfig(
        image_size=224,
        patch_size=16,
        d_vision=128,
        d_v_state=32,
        n_bi_glt_layers=2,
        n_anchor_layers=1,
        n_heads=4,
        n_kv_heads=2,
        spatial_window=25,
        ffn_expansion=2,
        ffn_shards=2,
    )
    return ContinuumVisionEncoder(config, d_model)
