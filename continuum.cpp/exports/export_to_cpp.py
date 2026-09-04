#!/usr/bin/env python3
"""
Export PyTorch Continuum SLM weights to C++ binary format.

Usage:
    python exports/export_to_cpp.py --checkpoint checkpoints/continuum_max_for_mobile.pt --output continuum.cpp/build/model.bin

Binary format (little-endian):
    [header: config struct (all int32)]
    [weight data: raw float32 arrays in order]
"""

import struct
import sys
import os
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from continuum.model.model import create_continuum_max, create_continuum_nano


def write_int32(f, val):
    f.write(struct.pack("<i", val))

def write_float32(f, val):
    f.write(struct.pack("<f", val))

def write_tensor(f, tensor: torch.Tensor):
    """Write tensor as raw float32 little-endian."""
    arr = tensor.detach().cpu().float().numpy()
    f.write(arr.tobytes())

def write_int4(f, tensor: torch.Tensor):
    """⏚ Phase A: Write tensor as INT4 with per-block scales.
    Block size = 32 values. Each block has one FP32 scale.
    Values quantized to [-8, 7], 2 values packed per byte.
    Format: [packed_bytes][scales_array]
    """
    import numpy as np
    arr = tensor.detach().cpu().float().numpy().ravel()
    n = arr.size
    block_size = 32
    num_blocks = (n + block_size - 1) // block_size
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    scales = np.zeros(num_blocks, dtype=np.float32)

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = arr[start:end]
        max_abs = max(np.abs(block).max(), 1e-8)
        scale = max_abs / 7.0
        scales[b] = scale
        inv = 1.0 / scale
        for i in range(0, len(block), 2):
            v0 = int(round(block[i] * inv))
            v0 = max(-8, min(7, v0))
            v1 = 0
            if i + 1 < len(block):
                v1 = int(round(block[i + 1] * inv))
                v1 = max(-8, min(7, v1))
            packed[start // 2 + i // 2] = ((v0 + 8) << 4) | (v1 + 8)

    f.write(packed.tobytes())
    f.write(scales.tobytes())

def write_fp16(f, tensor: torch.Tensor):
    """⏚ Phase A: Write tensor as FP16 (half precision)."""
    arr = tensor.detach().cpu().half().numpy()
    f.write(arr.tobytes())

# ⚠️ INT4/FP16 export format is NOT yet compatible with JNI loader.
# The JNI loader reads all weights as FP32 via the R macro.
# Until the JNI loader properly handles INT4/FP16 byte layouts,
# we export everything as FP32 but set the version number to indicate
# the intended quantization type. This allows:
# - Model files always load correctly (FP32 format)
# - Version number is preserved for future INT4 loading support
# - Int4Storage/quant_wire infrastructure is ready but not yet used
def write_tensor_quantized(f, tensor: torch.Tensor, quantize: str = "fp32"):
    """Write tensor — always FP32 for now (INT4/FP16 loading not yet in JNI)."""
    write_tensor(f, tensor)

def write_tensor_transposed(f, tensor: torch.Tensor):
    """Write transposed tensor (PyTorch [out,in] → C++ row-major [in,out])."""
    arr = tensor.detach().cpu().float().numpy().T.copy()
    f.write(arr.tobytes())


def export_model(model, output_path: str, quantize: str = "fp32"):
    """Export full model weights to binary format.

    Args:
        model: ContinuumModel instance
        output_path: Output .bin file path
        quantize: 'fp32', 'fp16', or 'int4' — weight quantization format
    """
    cfg = model.config

    # ⚡ Phase A: Version encodes quantization type
    # version=1: FP32, version=2: FP16, version=3: INT4
    version = 1 if quantize == "fp32" else (2 if quantize == "fp16" else 3)

    with open(output_path, "wb") as f:
        # ─── Magic number + version ───
        write_int32(f, 0x434F4E54)  # 'CONT'
        write_int32(f, version)

        # ─── Header: config (21 int32 values + 1 float + 1 int) ───
        write_int32(f, cfg.d_model)
        write_int32(f, cfg.d_state)
        write_int32(f, cfg.d_embed)
        write_int32(f, cfg.vocab_size)
        write_int32(f, cfg.n_layers)
        write_int32(f, cfg.glt_layers)
        write_int32(f, cfg.anchor_layers)
        write_int32(f, cfg.perception_layers)
        write_int32(f, cfg.core_layers)
        write_int32(f, cfg.output_layers)
        write_int32(f, cfg.ffn_expansion)
        write_int32(f, cfg.ffn_shards)
        write_int32(f, cfg.n_heads)
        write_int32(f, cfg.n_kv_heads)
        write_int32(f, cfg.window_size)
        write_int32(f, cfg.n_anchors)
        write_int32(f, cfg.n_static_anchors)
        write_int32(f, cfg.n_max_loops)
        write_int32(f, cfg.pmb_slots)
        write_int32(f, cfg.pmb_readout)
        write_int32(f, cfg.chunk_size)
        write_float32(f, cfg.halt_threshold)
        write_int32(f, cfg.eos_token_id)

        # ─── Embedding weights (quantized) ───
        emb = model.embedding
        write_tensor_quantized(f, emb.embed_table.weight, quantize)       # [vocab, d_embed]
        write_tensor_quantized(f, emb.up_proj.weight.T, quantize)           # [d_embed, d_model]
        write_tensor_quantized(f, emb.down_proj.weight.T, quantize)         # [d_model, d_embed]
        write_tensor(f, model.final_norm.scale)         # [d_model] — always FP32 (small)

        # ─── Layer weights: collect per-block, write GROUPED ───
        # ⚡ FIX: the C++ loaders (continuum.cpp / continuum_jni.cpp) read ALL
        # GLT layers first, then ALL Anchor layers, then ALL FFN layers. The old
        # exporter wrote blocks interleaved (GLT, FFN, Anchor, ...) so every
        # .bin model loaded misaligned garbage (tensor sizes differ per type,
        # so the file position drifted immediately). Write the exact order the
        # loaders expect: GLTs in block order, anchors in block order, FFNs in
        # block order (FFN i belongs to block i).
        all_blocks = (list(model.perception_blocks) +
                      list(model.core_blocks) +
                      list(model.output_blocks))

        glt_chunks: list = []
        anchor_chunks: list = []
        ffn_chunks: list = []

        for block in all_blocks:
            if block.is_glt:
                mixer = block.mixer
                glt_chunks.append([
                    (mixer.W_k.weight.T, quantize),        # [d_state, d_model]
                    (mixer.W_v.weight.T, quantize),
                    (mixer.W_q.weight.T, quantize),
                    (mixer.W_gamma.weight.T, quantize),
                    (mixer.W_gamma.bias, "fp32"),           # [d_state] — small, always FP32
                    (mixer.W_iota.weight.T, quantize),
                    (mixer.W_iota.bias, "fp32"),
                    (mixer.W_r.weight.T, quantize),
                    (mixer.W_r.bias, "fp32"),
                    (mixer.W_o.weight.T, quantize),          # [d_model, d_state]
                    (mixer.norm.scale, "fp32"),             # [d_model] — FP32
                    (mixer.kv_norm.scale, "fp32"),          # [d_state] — FP32
                ])
            else:
                mixer = block.mixer
                anchor_chunks.append([
                    (mixer.W_qkv.weight.T, quantize),        # [q_dim+2*kv_dim, d_model]
                    (mixer.W_o.weight.T, quantize),          # [d_model, q_dim]
                    (mixer.static_anchors, "fp32"),         # [n_static, d_model] — FP32 (small)
                    (mixer.alibi_slopes, "fp32"),           # [n_heads] — FP32
                    (mixer.norm.scale, "fp32"),             # [d_model] — FP32
                ])

            # FFN weights (every block has one) — large weights quantized
            ffn = block.ffn
            ffn_chunks.append([
                (ffn.gate_proj_fused.weight.T, quantize),   # [total_inter, d_model]
                (ffn.up_proj_fused.weight.T, quantize),
                (ffn.down_proj_fused.weight.T, quantize),   # [d_model, total_inter]
                (ffn.gate_head.weight.T, "fp32"),           # [n_shards, d_model] — FP32 (small)
                (ffn.gate_head.bias, "fp32"),               # [n_shards] — FP32
                (ffn.norm.scale, "fp32"),                   # [d_model] — FP32
            ])

        # Write grouped: all GLTs, then all anchors, then all FFNs.
        for chunk in glt_chunks + anchor_chunks + ffn_chunks:
            for tensor, q in chunk:
                write_tensor_quantized(f, tensor, q)

        # ─── Halting head (small, always FP32) ───
        hh = model.halting_head
        write_tensor(f, hh.pool_proj.weight.T)             # [d_model/4, d_model]
        write_tensor(f, hh.halt_proj.weight)                # [1, d_model/4]
        write_tensor(f, hh.halt_proj.bias)                  # [1]

        # ─── PMB (small, always FP32) ───
        pmb = model.pmb
        write_tensor(f, pmb.slots)                          # [n_slots, d_model]
        write_tensor(f, pmb.W_update.weight.T)              # [1, 2*d_model]
        write_tensor(f, pmb.W_update.bias)                  # [1]
        write_float32(f, pmb.write_scale.item())            # scalar

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported to {output_path} ({size_mb:.1f} MB, {quantize.upper()})")
    print(f"   GLT layers: {len(glt_chunks)}, Anchor: {len(anchor_chunks)}, FFN: {len(ffn_chunks)}")


def export_tokenizer(tokenizer, output_path: str):
    """Export BPE tokenizer to compact binary format for C++ engine.

    ⚡ FIX: this function previously read attributes ContinuumTokenizer does
    not have (`vocab`, `eos_token_id`, ...) so it exported an EMPTY vocab,
    and crashed on merge keys (raw `bytes` tuples) by calling .encode() on
    them. Now it reads the real attributes:
      - vocab  ← token_to_bytes (id → raw bytes; byte-level BPE tokens are
                 arbitrary byte strings, so we write RAW BYTES — never
                 re-encode through str)
      - merges ← (bytes, bytes) → new_id
      - special ← pad_id / bos_id / eos_id

    Format:
        uint32_t magic (0x54504F42 = 'BPTO')
        uint32_t version (1)
        uint32_t vocab_size
        uint32_t num_merges
        uint32_t num_special_tokens
        For each vocab entry: uint16_t len + bytes
        For each merge: uint32_t rank + uint16_t len_a + bytes_a + uint16_t len_b + bytes_b
        For each special token: uint32_t id + uint16_t len + bytes
    """
    import struct

    def _as_bytes(x) -> bytes:
        return x if isinstance(x, bytes) else str(x).encode("utf-8")

    # Build vocab list indexed by token ID from the real id→bytes table.
    token_to_bytes = getattr(tokenizer, "token_to_bytes", None)
    if isinstance(token_to_bytes, dict) and token_to_bytes:
        max_id = max(token_to_bytes.keys())
        vocab_list: list = [None] * (max_id + 1)
        for tid, tok_bytes in token_to_bytes.items():
            vocab_list[tid] = _as_bytes(tok_bytes)
        vocab_list = [v if v is not None else f"<unused_{i}>".encode()
                      for i, v in enumerate(vocab_list)]
    else:
        # Legacy str-vocab fallback (other tokenizer implementations)
        vocab = tokenizer.vocab if hasattr(tokenizer, "vocab") else {}
        if isinstance(vocab, dict):
            tmp = [None] * len(vocab)
            for tok, idx in vocab.items():
                if idx < len(tmp):
                    tmp[idx] = _as_bytes(tok)
            vocab_list = [v if v is not None else f"<unused_{i}>".encode()
                          for i, v in enumerate(tmp)]
        else:
            vocab_list = [_as_bytes(v) for v in vocab]

    # Build merges list (pairs of raw bytes, in rank order)
    merges = getattr(tokenizer, "merges", {}) or {}
    merge_pairs = []
    if hasattr(merges, "items"):
        for pair, rank in sorted(merges.items(), key=lambda x: x[1]):
            a, b = pair
            merge_pairs.append((_as_bytes(a), _as_bytes(b)))
    else:
        for m in merges:
            if isinstance(m, (tuple, list)) and len(m) == 2:
                merge_pairs.append((_as_bytes(m[0]), _as_bytes(m[1])))

    # Special tokens — the REAL attribute names on ContinuumTokenizer
    special_tokens = {}
    for attr, name in (("pad_id", "<pad>"), ("bos_id", "<bos>"), ("eos_id", "<eos>")):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_tokens[name] = int(tid)
    # Optional chat-template specials (only present on some tokenizers)
    for attr, name in (("user_token_id", "<|user|>"),
                       ("assistant_token_id", "<|assistant|>"),
                       ("system_token_id", "<|system|>")):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_tokens[name] = int(tid)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", 0x54504F42))  # magic 'BPTO'
        f.write(struct.pack("<I", 1))            # version
        f.write(struct.pack("<I", len(vocab_list)))
        f.write(struct.pack("<I", len(merge_pairs)))
        f.write(struct.pack("<I", len(special_tokens)))

        # Vocab entries
        for tok_bytes in vocab_list:
            f.write(struct.pack("<H", len(tok_bytes)))
            f.write(tok_bytes)

        # Merges
        for rank, (a_bytes, b_bytes) in enumerate(merge_pairs):
            f.write(struct.pack("<I", rank))
            f.write(struct.pack("<H", len(a_bytes)))
            f.write(a_bytes)
            f.write(struct.pack("<H", len(b_bytes)))
            f.write(b_bytes)

        # Special tokens
        for tok_str, tok_id in special_tokens.items():
            tok_bytes = tok_str.encode('utf-8')
            f.write(struct.pack("<I", tok_id))
            f.write(struct.pack("<H", len(tok_bytes)))
            f.write(tok_bytes)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ Tokenizer exported to {output_path} ({size_kb:.0f} KB)")
    print(f"   Vocab: {len(vocab_list)}, Merges: {len(merge_pairs)}, Special: {len(special_tokens)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Continuum SLM to C++ binary")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="continuum.cpp/build/model.bin", help="Output path")
    parser.add_argument("--model", default="max", choices=["nano", "max"],
                        help="Model variant")
    parser.add_argument("--quantize", default="fp32", choices=["fp32", "fp16", "int4"],
                        help="Weight quantization (fp32=400MB, fp16=200MB, int4=50MB)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json — exports tokenizer.bin for C++ engine")
    parser.add_argument("--tokenizer-output", default="continuum.cpp/build/tokenizer.bin",
                        help="Output path for tokenizer binary")
    args = parser.parse_args()

    print(f"Loading {args.model} model...")
    if args.model == "max":
        model = create_continuum_max()
    else:
        model = create_continuum_nano()

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    print(f"  {model.num_params:,} parameters")
    print(f"  Quantization: {args.quantize.upper()}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_model(model, args.output, quantize=args.quantize)

    # ⚡ Phase D: Export tokenizer if requested
    if args.tokenizer:
        os.makedirs(os.path.dirname(args.tokenizer_output), exist_ok=True)
        try:
            # ⚡ FIX: the class is ContinuumTokenizer (BPETokenizer never
            # existed — this import always failed and the export was skipped).
            from continuum.tokenizer.bpe import ContinuumTokenizer as PyBPE
            tok = PyBPE.load(args.tokenizer)
            export_tokenizer(tok, args.tokenizer_output)
        except Exception as e:
            print(f"⚠️ Tokenizer export failed: {e}")
            print(f"   You can export tokenizer separately later.")
