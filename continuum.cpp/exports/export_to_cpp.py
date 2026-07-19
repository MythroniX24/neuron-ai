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

def write_tensor_transposed(f, tensor: torch.Tensor):
    """Write transposed tensor (PyTorch [out,in] → C++ row-major [in,out])."""
    arr = tensor.detach().cpu().float().numpy().T.copy()
    f.write(arr.tobytes())


def export_model(model, output_path: str):
    """Export full model weights to binary format."""
    cfg = model.config

    with open(output_path, "wb") as f:
        # ─── Magic number + version ───
        write_int32(f, 0x434F4E54)  # 'CONT'
        write_int32(f, 1)           # version 1

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

        # ─── Embedding weights ───
        emb = model.embedding
        write_tensor(f, emb.embed_table.weight)       # [vocab, d_embed]
        write_tensor(f, emb.up_proj.weight.T)           # [d_embed, d_model] → row-major
        write_tensor(f, emb.down_proj.weight.T)         # [d_model, d_embed]
        write_tensor(f, model.final_norm.scale)         # [d_model]

        # ─── GLT layers ───
        all_blocks = (list(model.perception_blocks) +
                      list(model.core_blocks) +
                      list(model.output_blocks))

        glt_idx = 0
        anchor_idx = 0
        ffn_idx = 0

        for block in all_blocks:
            if block.is_glt:
                mixer = block.mixer
                write_tensor(f, mixer.W_k.weight.T)       # [d_state, d_model]
                write_tensor(f, mixer.W_v.weight.T)
                write_tensor(f, mixer.W_q.weight.T)
                write_tensor(f, mixer.W_gamma.weight.T)
                write_tensor(f, mixer.W_gamma.bias)        # [d_state]
                write_tensor(f, mixer.W_iota.weight.T)
                write_tensor(f, mixer.W_iota.bias)
                write_tensor(f, mixer.W_r.weight.T)
                write_tensor(f, mixer.W_r.bias)
                write_tensor(f, mixer.W_o.weight.T)        # [d_model, d_state]
                write_tensor(f, mixer.norm.scale)           # [d_model]
                write_tensor(f, mixer.kv_norm.scale)        # [d_state]
                glt_idx += 1
            else:
                mixer = block.mixer
                write_tensor(f, mixer.W_qkv.weight.T)      # [q_dim+2*kv_dim, d_model]
                write_tensor(f, mixer.W_o.weight.T)         # [d_model, q_dim]
                write_tensor(f, mixer.static_anchors)       # [n_static, d_model]
                write_tensor(f, mixer.alibi_slopes)         # [n_heads]
                write_tensor(f, mixer.norm.scale)           # [d_model]
                anchor_idx += 1

            # FFN weights (every block has one)
            ffn = block.ffn
            write_tensor(f, ffn.gate_proj_fused.weight.T)  # [total_inter, d_model]
            write_tensor(f, ffn.up_proj_fused.weight.T)
            write_tensor(f, ffn.down_proj_fused.weight.T)  # [d_model, total_inter]
            write_tensor(f, ffn.gate_head.weight.T)         # [n_shards, d_model]
            write_tensor(f, ffn.gate_head.bias)             # [n_shards]
            write_tensor(f, ffn.norm.scale)                 # [d_model]
            ffn_idx += 1

        # ─── Halting head ───
        hh = model.halting_head
        write_tensor(f, hh.pool_proj.weight.T)             # [d_model/4, d_model]
        write_tensor(f, hh.halt_proj.weight)                # [1, d_model/4]
        write_tensor(f, hh.halt_proj.bias)                  # [1]

        # ─── PMB ───
        pmb = model.pmb
        write_tensor(f, pmb.slots)                          # [n_slots, d_model]
        write_tensor(f, pmb.W_update.weight.T)              # [1, 2*d_model]
        write_tensor(f, pmb.W_update.bias)                  # [1]
        write_float32(f, pmb.write_scale.item())            # scalar

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported to {output_path} ({size_mb:.1f} MB)")
    print(f"   GLT layers: {glt_idx}, Anchor: {anchor_idx}, FFN: {ffn_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Continuum SLM to C++ binary")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="continuum.cpp/build/model.bin", help="Output path")
    parser.add_argument("--model", default="max", choices=["nano", "max"],
                        help="Model variant")
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_model(model, args.output)
