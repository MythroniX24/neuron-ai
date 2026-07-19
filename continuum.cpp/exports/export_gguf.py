#!/usr/bin/env python3
"""
Export Continuum SLM to GGUF format (v3).

GGUF is the standard format used by llama.cpp ecosystem.
This enables loading our custom Continuum architecture on Android via JNI.

GGUF Binary Layout:
  [Header]: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
  [Metadata KVs]: array of key-value pairs (architecture, hyperparams, tokenizer)
  [Tensor Infos]: array of {name, n_dims, dims[], type, offset}
  [Padding]: to ALIGNMENT bytes
  [Tensor Data]: raw weight bytes

Usage:
    python exports/export_gguf.py --checkpoint model.pt --output model.gguf

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import struct
import sys
import os
import argparse
from typing import List, Dict, Tuple, Any
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ============================================================================
# GGUF Constants
# ============================================================================

GGUF_MAGIC   = 0x47554746  # "GGUF"
GGUF_VERSION = 3
ALIGNMENT    = 32

# Metadata value types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# Tensor types (ggml_type enum)
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9


# ============================================================================
# GGUF Binary Writer
# ============================================================================

class GGUFFile:
    """Writes a GGUF file with proper alignment."""

    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.offset = 0

    def write_raw(self, data: bytes):
        self.f.write(data)
        self.offset += len(data)

    def write_u8(self, val: int):
        self.write_raw(struct.pack("<B", val))

    def write_i8(self, val: int):
        self.write_raw(struct.pack("<b", val))

    def write_u16(self, val: int):
        self.write_raw(struct.pack("<H", val))

    def write_i16(self, val: int):
        self.write_raw(struct.pack("<h", val))

    def write_u32(self, val: int):
        self.write_raw(struct.pack("<I", val))

    def write_i32(self, val: int):
        self.write_raw(struct.pack("<i", val))

    def write_u64(self, val: int):
        self.write_raw(struct.pack("<Q", val))

    def write_i64(self, val: int):
        self.write_raw(struct.pack("<q", val))

    def write_f32(self, val: float):
        self.write_raw(struct.pack("<f", val))

    def write_f64(self, val: float):
        self.write_raw(struct.pack("<d", val))

    def write_bool(self, val: bool):
        self.write_u8(1 if val else 0)

    def write_string(self, s: str):
        data = s.encode("utf-8")
        if len(data) > 65535:
            raise ValueError(f"String too long: {len(data)} bytes")
        self.write_u64(len(data))
        self.write_raw(data)

    def write_value(self, val_type: int, val):
        """Write a typed value."""
        if val_type == GGUF_TYPE_UINT8:
            self.write_u8(val)
        elif val_type == GGUF_TYPE_INT8:
            self.write_i8(val)
        elif val_type == GGUF_TYPE_UINT16:
            self.write_u16(val)
        elif val_type == GGUF_TYPE_INT16:
            self.write_i16(val)
        elif val_type == GGUF_TYPE_UINT32:
            self.write_u32(val)
        elif val_type == GGUF_TYPE_INT32:
            self.write_i32(val)
        elif val_type == GGUF_TYPE_UINT64:
            self.write_u64(val)
        elif val_type == GGUF_TYPE_INT64:
            self.write_i64(val)
        elif val_type == GGUF_TYPE_FLOAT32:
            self.write_f32(val)
        elif val_type == GGUF_TYPE_FLOAT64:
            self.write_f64(val)
        elif val_type == GGUF_TYPE_BOOL:
            self.write_bool(val)
        elif val_type == GGUF_TYPE_STRING:
            self.write_string(val)
        elif val_type == GGUF_TYPE_ARRAY:
            arr_type, arr_len, arr_data = val
            self.write_u32(arr_type)
            self.write_u32(arr_len)
            for item in arr_data:
                self.write_value(arr_type, item)
        else:
            raise ValueError(f"Unknown value type: {val_type}")

    def write_metadata_kv(self, key: str, val_type: int, val):
        """Write one metadata KV pair."""
        self.write_string(key)
        self.write_u32(val_type)
        self.write_value(val_type, val)

    def write_tensor_info(self, name: str, shape: List[int], tensor_type: int, offset: int):
        """Write one tensor info entry."""
        self.write_string(name)
        self.write_u32(len(shape))
        for dim in reversed(shape):  # GGUF uses reverse dimension order
            self.write_u64(dim)
        self.write_u32(tensor_type)
        self.write_u64(offset)

    def pad_to_alignment(self):
        remainder = self.offset % ALIGNMENT
        if remainder:
            self.write_raw(b"\x00" * (ALIGNMENT - remainder))

    def tell(self) -> int:
        return self.offset

    def close(self):
        self.f.close()


# ============================================================================
# Export Function
# ============================================================================

def export_gguf(model, output_path: str):
    """Export Continuum SLM model to GGUF format."""
    cfg = model.config
    f = GGUFFile(output_path)

    # ─── Collect metadata KV pairs ───
    metadata: List[Tuple[str, int, Any]] = []

    # General architecture info
    metadata.append(("general.architecture", GGUF_TYPE_STRING, "continuum"))
    metadata.append(("general.name", GGUF_TYPE_STRING, "Continuum SLM"))
    metadata.append(("general.quantization_version", GGUF_TYPE_UINT32, 1))
    metadata.append(("general.file_type", GGUF_TYPE_UINT32, 1))  # 1=mostly F32

    # Continuum hyperparameters
    metadata.append(("continuum.context_length", GGUF_TYPE_UINT32, cfg.chunk_size))
    metadata.append(("continuum.embedding_length", GGUF_TYPE_UINT32, cfg.d_model))
    metadata.append(("continuum.block_count", GGUF_TYPE_UINT32, cfg.n_layers))
    metadata.append(("continuum.feed_forward_length", GGUF_TYPE_UINT32,
                     cfg.d_model * cfg.ffn_expansion))

    metadata.append(("continuum.attention.head_count", GGUF_TYPE_UINT32, cfg.n_heads))
    metadata.append(("continuum.attention.head_count_kv", GGUF_TYPE_UINT32, cfg.n_kv_heads))
    metadata.append(("continuum.attention.head_dim", GGUF_TYPE_UINT32,
                     getattr(cfg, 'head_dim', cfg.d_model // cfg.n_heads)))
    metadata.append(("continuum.attention.window_size", GGUF_TYPE_UINT32, cfg.window_size))
    metadata.append(("continuum.attention.n_anchors", GGUF_TYPE_UINT32, cfg.n_anchors))
    metadata.append(("continuum.attention.n_static_anchors", GGUF_TYPE_UINT32, cfg.n_static_anchors))

    metadata.append(("continuum.glt.state_dim", GGUF_TYPE_UINT32, cfg.d_state))
    metadata.append(("continuum.glt.n_layers", GGUF_TYPE_UINT32, cfg.glt_layers))
    metadata.append(("continuum.glt.n_max_loops", GGUF_TYPE_UINT32, cfg.n_max_loops))
    metadata.append(("continuum.glt.halt_threshold", GGUF_TYPE_FLOAT32, cfg.halt_threshold))

    metadata.append(("continuum.ffn.expansion", GGUF_TYPE_UINT32, cfg.ffn_expansion))
    metadata.append(("continuum.ffn.shards", GGUF_TYPE_UINT32, cfg.ffn_shards))
    # ffn_total_intermediate may not exist on config directly — compute safely
    ffn_total = getattr(cfg, 'ffn_total_intermediate',
                         cfg.d_model * cfg.ffn_expansion * cfg.ffn_shards)
    metadata.append(("continuum.ffn.total_intermediate", GGUF_TYPE_UINT32, ffn_total))

    metadata.append(("continuum.embed.dim", GGUF_TYPE_UINT32, cfg.d_embed))
    metadata.append(("continuum.embed.vocab_size", GGUF_TYPE_UINT32, cfg.vocab_size))
    metadata.append(("continuum.embed.eos_token_id", GGUF_TYPE_UINT32, cfg.eos_token_id))

    metadata.append(("continuum.pmb.slots", GGUF_TYPE_UINT32, cfg.pmb_slots))
    metadata.append(("continuum.pmb.readout", GGUF_TYPE_UINT32, cfg.pmb_readout))

    metadata.append(("continuum.perception_layers", GGUF_TYPE_UINT32, cfg.perception_layers))
    metadata.append(("continuum.core_layers", GGUF_TYPE_UINT32, cfg.core_layers))
    metadata.append(("continuum.output_layers", GGUF_TYPE_UINT32, cfg.output_layers))

    # ─── Collect tensor infos ───
    tensor_infos: List[Tuple[str, List[int], int]] = []

    def add_tensor(name: str, shape: List[int]):
        tensor_infos.append((name, shape, GGML_TYPE_F32))

    # Embedding
    add_tensor("token_embd.weight", [cfg.vocab_size, cfg.d_embed])
    add_tensor("embed_up.weight", [cfg.d_embed, cfg.d_model])
    add_tensor("embed_down.weight", [cfg.d_model, cfg.d_embed])

    # Build block order
    all_blocks = (
        list(model.perception_blocks) +
        list(model.core_blocks) +
        list(model.output_blocks)
    )

    glt_idx = 0
    anchor_idx = 0

    head_dim = getattr(cfg, 'head_dim', cfg.d_model // cfg.n_heads)
    for i, block in enumerate(all_blocks):
        if block.is_glt:
            prefix = f"blk.{i}.glt."
            d = cfg.d_state
            add_tensor(f"{prefix}W_k.weight", [d, cfg.d_model])
            add_tensor(f"{prefix}W_v.weight", [d, cfg.d_model])
            add_tensor(f"{prefix}W_q.weight", [d, cfg.d_model])
            add_tensor(f"{prefix}W_gamma.weight", [d, cfg.d_model])
            add_tensor(f"{prefix}W_gamma.bias", [d])
            add_tensor(f"{prefix}W_iota.weight", [d, cfg.d_model])
            add_tensor(f"{prefix}W_iota.bias", [d])
            add_tensor(f"{prefix}W_r.weight", [d, cfg.d_model])
            add_tensor(f"{prefix}W_r.bias", [d])
            add_tensor(f"{prefix}W_o.weight", [cfg.d_model, d])
            add_tensor(f"{prefix}norm.weight", [cfg.d_model])
            add_tensor(f"{prefix}kv_norm.weight", [d])
            glt_idx += 1
        else:
            prefix = f"blk.{i}.attn."
            q_dim = cfg.n_heads * head_dim
            kv_dim = cfg.n_kv_heads * head_dim
            add_tensor(f"{prefix}W_qkv.weight", [cfg.d_model, q_dim + 2*kv_dim])
            add_tensor(f"{prefix}W_o.weight", [q_dim, cfg.d_model])
            add_tensor(f"{prefix}static_anchors.weight", [cfg.n_static_anchors, cfg.d_model])
            add_tensor(f"{prefix}alibi_slopes.weight", [cfg.n_heads])
            add_tensor(f"{prefix}norm.weight", [cfg.d_model])
            anchor_idx += 1

        # FFN (every block)
        prefix = f"blk.{i}.ffn."
        add_tensor(f"{prefix}gate_proj.weight", [cfg.d_model, ffn_total])
        add_tensor(f"{prefix}up_proj.weight", [cfg.d_model, ffn_total])
        add_tensor(f"{prefix}down_proj.weight", [ffn_total, cfg.d_model])
        add_tensor(f"{prefix}gate_head.weight", [cfg.d_model, cfg.ffn_shards])
        add_tensor(f"{prefix}gate_head.bias", [cfg.ffn_shards])
        add_tensor(f"{prefix}norm.weight", [cfg.d_model])

    # Halting head
    add_tensor("halting.pool_proj.weight", [cfg.d_model // 4, cfg.d_model])
    add_tensor("halting.halt_proj.weight", [1, cfg.d_model // 4])
    add_tensor("halting.halt_proj.bias", [1])

    # PMB
    add_tensor("pmb.slots.weight", [cfg.pmb_slots, cfg.d_model])
    add_tensor("pmb.W_update.weight", [1, cfg.d_model * 2])
    add_tensor("pmb.W_update.bias", [1])

    # Final norm
    add_tensor("final_norm.weight", [cfg.d_model])

    # ─── Write Header ───
    f.write_u32(GGUF_MAGIC)
    f.write_u32(GGUF_VERSION)
    f.write_u64(len(tensor_infos))
    f.write_u64(len(metadata))

    # ─── Write Metadata KVs ───
    for key, vtype, val in metadata:
        f.write_metadata_kv(key, vtype, val)

    # ─── Write Tensor Infos ───
    # First pass: calculate offsets
    tensor_offsets = {}
    data_offset = 0
    for name, shape, tensor_type in tensor_infos:
        tensor_offsets[name] = data_offset
        n_elements = 1
        for d in shape:
            n_elements *= d
        data_size = n_elements * 4  # F32 = 4 bytes
        # Align to 32 bytes
        if data_size % ALIGNMENT:
            data_size = ((data_size // ALIGNMENT) + 1) * ALIGNMENT
        data_offset += data_size

    # Write tensor infos
    for name, shape, tensor_type in tensor_infos:
        f.write_tensor_info(name, shape, tensor_type, tensor_offsets[name])

    # ─── Pad to alignment ───
    f.pad_to_alignment()

    # ─── Write Tensor Data ───
    data_start = f.tell()

    def write_weight(tensor, name: str):
        """Write tensor data, transposing to match GGUF convention (row-major)."""
        arr = tensor.detach().cpu().float().numpy()
        f.write_tensor_data(arr)
        # Pad to alignment
        remainder = (f.tell() - data_start) % ALIGNMENT
        if remainder:
            f.write_raw(b"\x00" * (ALIGNMENT - remainder))

    # Embedding (write in PyTorch native order — GGUF uses same layout)
    emb = model.embedding
    write_weight(emb.embed_table.weight, "token_embd.weight")       # [vocab, d_embed]
    write_weight(emb.up_proj.weight.T, "embed_up.weight")            # [d_embed, d_model]
    write_weight(emb.down_proj.weight.T, "embed_down.weight")        # [d_model, d_embed]

    glt_idx = 0
    anchor_idx = 0

    for i, block in enumerate(all_blocks):
        if block.is_glt:
            mixer = block.mixer
            write_weight(mixer.W_k.weight.T, f"blk.{i}.glt.W_k.weight")
            write_weight(mixer.W_v.weight.T, f"blk.{i}.glt.W_v.weight")
            write_weight(mixer.W_q.weight.T, f"blk.{i}.glt.W_q.weight")
            write_weight(mixer.W_gamma.weight.T, f"blk.{i}.glt.W_gamma.weight")
            write_weight(mixer.W_gamma.bias, f"blk.{i}.glt.W_gamma.bias")
            write_weight(mixer.W_iota.weight.T, f"blk.{i}.glt.W_iota.weight")
            write_weight(mixer.W_iota.bias, f"blk.{i}.glt.W_iota.bias")
            write_weight(mixer.W_r.weight.T, f"blk.{i}.glt.W_r.weight")
            write_weight(mixer.W_r.bias, f"blk.{i}.glt.W_r.bias")
            write_weight(mixer.W_o.weight.T, f"blk.{i}.glt.W_o.weight")
            write_weight(mixer.norm.scale, f"blk.{i}.glt.norm.weight")
            write_weight(mixer.kv_norm.scale, f"blk.{i}.glt.kv_norm.weight")
            glt_idx += 1
        else:
            mixer = block.mixer
            write_weight(mixer.W_qkv.weight.T, f"blk.{i}.attn.W_qkv.weight")
            write_weight(mixer.W_o.weight.T, f"blk.{i}.attn.W_o.weight")
            write_weight(mixer.static_anchors, f"blk.{i}.attn.static_anchors.weight")
            write_weight(mixer.alibi_slopes, f"blk.{i}.attn.alibi_slopes.weight")
            write_weight(mixer.norm.scale, f"blk.{i}.attn.norm.weight")
            anchor_idx += 1

        ffn = block.ffn
        write_weight(ffn.gate_proj_fused.weight.T, f"blk.{i}.ffn.gate_proj.weight")
        write_weight(ffn.up_proj_fused.weight.T, f"blk.{i}.ffn.up_proj.weight")
        write_weight(ffn.down_proj_fused.weight.T, f"blk.{i}.ffn.down_proj.weight")
        write_weight(ffn.gate_head.weight.T, f"blk.{i}.ffn.gate_head.weight")
        write_weight(ffn.gate_head.bias, f"blk.{i}.ffn.gate_head.bias")
        write_weight(ffn.norm.scale, f"blk.{i}.ffn.norm.weight")

    # Halting head
    hh = model.halting_head
    write_weight(hh.pool_proj.weight.T, "halting.pool_proj.weight")
    write_weight(hh.halt_proj.weight, "halting.halt_proj.weight")
    write_weight(hh.halt_proj.bias, "halting.halt_proj.bias")

    # PMB
    pmb = model.pmb
    write_weight(pmb.slots, "pmb.slots.weight")
    write_weight(pmb.W_update.weight.T, "pmb.W_update.weight")
    write_weight(pmb.W_update.bias, "pmb.W_update.bias")

    # Final norm
    write_weight(model.final_norm.scale, "final_norm.weight")

    f.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported GGUF to {output_path} ({size_mb:.1f} MB)")
    print(f"   Architecture: continuum")
    print(f"   Tensors: {len(tensor_infos)}")
    print(f"   GLT layers: {glt_idx}, Anchor: {anchor_idx}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Continuum SLM to GGUF")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="continuum.cpp/build/model.gguf",
                        help="Output GGUF path")
    parser.add_argument("--model", default="max", choices=["nano", "max"],
                        help="Model variant")
    args = parser.parse_args()

    # Import torch (lazy — not needed for syntax check)
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. This script requires PyTorch.")
        print("  pip install torch")
        sys.exit(1)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from continuum.model.model import create_continuum_max, create_continuum_nano

    print(f"Loading {args.model} model...")
    if args.model == "max":
        model = create_continuum_max()
    else:
        model = create_continuum_nano()

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"  {model.num_params:,} parameters")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_gguf(model, args.output)
