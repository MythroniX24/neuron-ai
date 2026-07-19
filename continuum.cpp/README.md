# Continuum SLM — C++ Inference Engine (Phase 3)

**Target: 20-50 tok/s on Android ARM CPUs**

C++ inference engine for the Continuum SLM custom architecture (GLT, Anchor Attention, GatedShardFFN).

## Architecture

```
continuum.cpp/
├── include/
│   ├── tensor.h      # Minimal tensor library (arena-based, zero-alloc inference)
│   ├── model.h        # Model config, weight structs, layer declarations
│   └── sampler.h      # Token sampling (temp, top-k, top-p, rep penalty)
├── src/
│   ├── tensor.cpp     # Basic tensor ops (matmul, RMSNorm, softmax, activations)
│   ├── model.cpp      # GLT, Anchor Attention, FFN forward passes
│   ├── sampler.cpp    # Sampling implementation
│   └── continuum.cpp  # Main engine: weight loader, generate(), CLI
├── exports/
│   └── export_to_cpp.py  # Python → C++ binary weight export
├── CMakeLists.txt
├── Makefile
└── README.md
```

## Build

```bash
# Desktop (Linux/macOS)
make

# Android ARM64
make android-arm64

# Or with CMake
mkdir build && cd build && cmake .. && make
```

## Usage

```bash
# 1. Export weights from Python checkpoint
python exports/export_to_cpp.py \
    --checkpoint checkpoints/continuum_max_for_mobile.pt \
    --output build/model.bin

# 2. Run inference
./build/continuum build/model.bin \
    --prompt "What is AI?" \
    --temp 0.8 --max-tokens 100
```

## Custom Architecture Ported

| Module | Python | C++ | Status |
|--------|--------|-----|--------|
| RMSNorm | `RMSNorm.forward()` | `tensor_rms_norm()` | ✅ |
| FactorizedEmbedding | `embed()` `project_to_logits()` | `embed_forward()` `output_projection()` | ✅ |
| GLTLayer | `GLTLayer.forward()` | `glt_forward()` | ✅ |
| AnchorAttention | `AnchorAttention.forward()` | `anchor_forward()` | ✅ |
| GatedShardFFN | `GatedShardFFN.forward()` | `ffn_forward()` | ✅ |
| HaltingHead (ADL) | `HaltingHead.forward()` | `halting_forward()` | ✅ |
| PersistentMemoryBank | `PMB.read()` `PMB.write()` | `pmb_read()` `pmb_write()` | ✅ |
| ContinuumModel | `forward()` | `continuum_forward()` | ✅ |
| Sampling | `_fast_sample()` | `Sampler.sample()` | ✅ |

## Performance Targets

| Platform | Tok/sec |
|----------|---------|
| Desktop CPU (-O3) | 10-30 |
| Android ARM NEON | 20-50 |
| With BLAS | 30-80 |

## Next: Phase 4

- NEON intrinsics for ARM matmul
- INT8/INT4 quantized inference
- GGUF format compatibility
- Android APK packaging
