<div align="center">
  
# 🧠 Neuron AI — Continuum SLM

### A Ground-Up Small Language Model for On-Device CPU Inference

[![Kaggle](https://img.shields.io/badge/Kaggle-Training%20Notebook-20BEFF?logo=kaggle)](https://kaggle.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/MythroniX24/neuron-ai?style=social)](https://github.com/MythroniX24/neuron-ai)
[![CI](https://github.com/MythroniX24/neuron-ai/actions/workflows/python-tests.yml/badge.svg)](https://github.com/MythroniX24/neuron-ai/actions/workflows/python-tests.yml)
[![Open Issues](https://img.shields.io/github/issues/MythroniX24/neuron-ai)](https://github.com/MythroniX24/neuron-ai/issues)
[![Last Commit](https://img.shields.io/github/last-commit/MythroniX24/neuron-ai)](https://github.com/MythroniX24/neuron-ai/commits/master)

**Train your own conversational AI — from scratch — in ~1.5 hours on free Kaggle GPUs!**

</div>

---

## ✨ Overview

**Neuron AI (Continuum SLM)** is a custom, from-scratch Small Language Model architecture designed for **efficient on-device inference** on mobile CPUs. Unlike models that adapt existing Transformer architectures, Continuum is built on **four original, cooperating mechanisms** designed from first principles for the mobile constraint:

| Component | What It Does | Why It Matters |
|---|---|---|
| **Gated Linear Trace (GLT)** | O(1) memory recurrent backbone — replaces self-attention as the default mixer | Never grows with conversation length |
| **Anchor Attention** | Bounded-size real softmax attention (window + persistent anchors), GQA + ALiBi | Precise recall where recurrence falls short |
| **Adaptive Depth Looping (ADL)** | Shared-weight reasoning core that loops 1–5× per token | More reasoning depth without more parameters |
| **Persistent Memory Bank (PMB)** | Fixed-size content-addressable long-term memory, written every K-token chunk | Survives across app sessions (kill & resume) |
| **🆕 Continuum Vision (ViGLT)** | Bidirectional GLT vision encoder — same primitives | Multimodal: image + text understanding |

**Headline properties:**
- **No growing KV-cache** — ever. O(1) memory per layer regardless of context length.
- **Streaming single-token decode** — no prompt reprocessing between turns.
- **Instant conversation resume** — ~100 KB state checkpoint (not megabytes).
- **~100 MB INT8 model** — fits comfortably on any modern phone; weights survive save/load round-trips.
- **Trainable from scratch** on free GPUs (Kaggle T4 ×2 ≈ 1.5 hours for the 102M model).
- **🆕 Multimodal vision** — unified GLT primitives across text and image encoders.
- **✅ 79-test regression suite** running in CI on every push (model, layers, attention, vision, tokenizer, trainer, inference engine).

---

## 🏗️ Architecture at a Glance

```
                                CONTINUUM — MACRO DATA FLOW
                                ============================

  Input text
      |
      v
 +-----------+      +--------------+
 | Tokenizer |----->| Embedding    |    byte-level BPE,
 | (BPE)     |      | (factorized) |    tied input/output
 +-----------+      +--------------+
                           |
                           v
   ======================================================
   |              STAGE 1 : PERCEPTION                   |
   |     [ GLT ] -> [ GLT ] -> [ Anchor Attention ]       |
   |            single fixed-depth pass                  |
   ======================================================
                           |
                           v
   ======================================================
   |         STAGE 2 : REASONING CORE  (looped)           |
   |                                                       |
   |   +-------------------------------------+             |
   |   |  [ GLT ] -> [Anchor Attn] -> [ GLT ] | <----+      |
   |   +------------------+--------------------+     |      |
   |                      |                          |      |
   |                      v                          |      |
   |               [ Halting Head ]                  |      |
   |               /              \                  |      |
   |         confident?        not yet?               |      |
   |             |             (loop again,           |      |
   |             |              up to N_max times) ---+      |
   |             v                                          |
   |     (weights are SHARED across every loop iteration)    |
   ======================================================
                           |
                           v
   ======================================================
   |               STAGE 3 : OUTPUT                       |
   |   [GLT] -> [GLT] -> [GLT] -> [ Anchor Attention ]     |
   ======================================================
                           |
                           v
                  +-------------------+
                  | Output projection |   tied embedding + softmax
                  +-------------------+
                           |
                           v
                 next-token probability distribution


   Side-channel, crosses every stage:
   +--------------------------------------------------------------+
   |                 PERSISTENT MEMORY BANK (PMB)                  |
   |  read by every Anchor Attention layer through its anchor      |
   |  tokens  --  written once per K-token chunk via a gated,      |
   |  content-addressed update (fixed slot count, never grows)     |
   +--------------------------------------------------------------+
```

> 📖 **Full architecture document:** [`continuum-slm-architecture.md`](./continuum-slm-architecture.md) — 23 sections of detailed technical explanation.

---

## 📊 Model Tiers

| Tier | Params | +Vision | `d_model` | Layers | Vocab | Window | FFN Shards | ADL Max | Factory |
|---|---|---|---|---|---|---|---|---|---|
| **Nano** | ~5M | +0.9M | 192 | 6 | 8,000 | 48 | 2 | 3 | `create_continuum_nano()` |
| **Small** | ~20M | +2.6M | 384 | 8 | 12,000 | 96 | 4 | 4 | `create_continuum_small()` |
| **Base**¹ | ~50M | — | 576 | 10 | 16,000 | 128 | 4 | 4 | *config-only* |
| **Max** 🏆 | **~102M** | **+13.5M** | **768** | **12** | **16,000** | **128** | **6** | **5** | `create_continuum_max()` |

> ¹ Base tier has no preset factory function yet — instantiate via `ContinuumConfig` with the values above. All tiers share the same architecture: one codebase, any size.

---

## 🚀 Quick Start — Train Your Own AI

### Option 1: Kaggle (Recommended — Free GPU!)

**⏱️ Training time:** ~1.5 hours on T4 ×2 GPU (Kaggle free tier)

1. Go to [Kaggle](https://kaggle.com) → **File → Import Notebook → GitHub**
2. Select: `MythroniX24/neuron-ai` → branch `master` → `continuum/kaggle/continuum_100m_training.ipynb`
3. **Runtime → Change runtime type → GPU T4 ×2** (or T4 ×1 / P100)
4. Click **Run All** ☝️

The notebook will:
- ✅ Clone the project & install dependencies
- ✅ Download the **52K-example Alpaca instruction dataset** (Dolly support exists but is disabled by default — see note below)
- ✅ Train the **102M parameter Continuum-Max** model from scratch
- ✅ Save FP16 + INT8 checkpoints and the tokenizer
- ✅ Push results to GitHub (optional, set `GITHUB_TOKEN` secret)
- ✅ Test the model — chat with your freshly trained AI!

> ℹ️ **Why no Dolly?** The Databricks Dilly dataset endpoint currently serves broken signed URLs via HuggingFace's Xet CDN (`get_dolly_data()` prints a warning and returns empty). Training runs fine on Alpaca alone; flip `include_dolly=True` once upstream fixes it.

**📥 Download your trained model:**

```
Kaggle → Data tab → /kaggle/working/checkpoints/
  → continuum_max_for_mobile.pt    (FP16 master, ~389 MB)
  → continuum_max_int8_phone.pt    (INT8 phone build, ~100 MB)
  → tokenizer_4k.json              (trained BPE tokenizer)
```

Drop any `.pt` into the local `checkpoints/` folder (or point `NEURON_MODEL_PATH` at it) and the chat CLI / web UI will find it automatically.

### Option 2: Google Colab

1. Open: [Colab Notebook](https://colab.research.google.com/github/MythroniX24/neuron-ai/blob/master/continuum/colab/continuum_100m_training.ipynb)
2. **Runtime → Change runtime type → GPU T4**
3. Click **Run All**

### Option 3: Local (Linux / macOS / WSL)

```bash
# Clone
git clone https://github.com/MythroniX24/neuron-ai.git
cd neuron-ai

# Install (CPU-only PyTorch is enough for inference & tests)
pip install -r continuum/requirements.txt

# Sanity check — build the nano model
python3 -c "
from continuum.model.model import create_continuum_nano
print(f'Model: {create_continuum_nano().num_params:,} parameters')
"

# Run the test suite (79 tests, ~6 min on a low-end CPU)
cd continuum && python -m pytest \
  inference/test_engine.py model/test_audit_fixes.py \
  model/test_model.py model/test_layers.py model/test_attention.py \
  model/test_vision.py tokenizer/test_tokenizer.py -q && cd ..

# Tiny local training smoke-run (CPU — real training belongs on Kaggle/Colab)
# ⚠️ run.py train expects PLAIN TEXT (one text per line). For JSONL
# {"instruction":..., "response":...} use the ConversationalDataset API
# (see Custom Dataset Training below) or pre-convert:
#   python -c "import json,sys; [print(json.loads(l)['instruction']+'\n'+json.loads(l)['response']) for l in open(sys.argv[1]) if l.strip()]" my_data.jsonl > corpus.txt
python3 continuum/run.py train --data my_data.txt --epochs 1 --batch_size 4
```

> 💡 **Weak CPU?** Export `OMP_NUM_THREADS=1` before running tests — on 2-core machines this roughly halves wall-clock time.

---

## 💬 Chat With Your Trained Model

**CLI chat** (loads any checkpoint under `checkpoints/`, or pass one explicitly):

```bash
python3 continuum/run.py chat --model checkpoints/continuum_max_for_mobile.pt --port 5000
```

**Python API:**

```python
import torch
from continuum.conversation.manager import ConversationManager
from continuum.model.model import create_continuum_max
from continuum.tokenizer.bpe import ContinuumTokenizer

# Load model + tokenizer
model = create_continuum_max()
model.load_state_dict(torch.load(
    "checkpoints/continuum_max_for_mobile.pt", map_location="cpu"
))
model.eval()

tokenizer = ContinuumTokenizer.load("checkpoints/tokenizer_4k.json")

# Chat! Recurrent state carries across turns — history is NOT re-sent each turn.
manager = ConversationManager(model=model, tokenizer=tokenizer, device="cpu")
response = manager.chat("What is the capital of France?", max_new_tokens=80)
print(response)
```

**Mobile deployment (INT8 quantized):**

```python
manager = ConversationManager(
    model=model, tokenizer=tokenizer, device="cpu", quantize=True
)
# Quantized weights are registered buffers — they survive
# state_dict() save/load, so exported phone builds are complete.
```

**Web UI:**

```bash
# Auto-discovers checkpoints/*.pt (or set NEURON_MODEL_PATH=/path/to/model.pt)
python3 continuum/ui/app.py --port 5000

# UI-only demo mode (no model needed)
python3 continuum/ui/app.py --demo
```

---

## 🧩 Project Structure

```
neuron-ai/
├── continuum/
│   ├── __init__.py              # Package init
│   ├── run.py                   # CLI entry points: train / chat
│   ├── model/
│   │   ├── model.py             # 🧠 Main model: stages, ADL, multimodal, state IO
│   │   ├── layers.py            # GLT layer, GatedShardFFN, RMSNorm, FactorizedEmbedding
│   │   ├── attention.py         # Anchor Attention + Persistent Memory Bank
│   │   ├── vision.py            # 🆕 ViGLT vision encoder (patch embed → BiGLT → projector)
│   │   ├── test_model.py        # Unit tests
│   │   ├── test_layers.py
│   │   ├── test_attention.py
│   │   ├── test_vision.py
│   │   └── test_audit_fixes.py  # Regression tests for trainer/model fixes
│   ├── training/
│   │   ├── trainer.py           # Training loop, curriculum learning, AMP, logging
│   │   ├── losses.py            # ContinuumLoss: CE + ponder cost + sparsity
│   │   └── parallel_scan.py     # Associative scan for GLT (parallel training!)
│   ├── conversation/
│   │   ├── dataset.py           # Alpaca loading (+ Dolly behind a flag), bucket sampler
│   │   ├── manager.py           # Conversation manager (incremental prompting)
│   │   └── template.py          # Chat template format
│   ├── inference/
│   │   ├── engine.py            # INT8 quantization, speculative decoding, generation
│   │   └── test_engine.py       # Regression tests for quantize/PMB/speculative paths
│   ├── tokenizer/
│   │   ├── bpe.py               # Byte-level BPE tokenizer (train + encode + decode)
│   │   └── tokenizer_4k.json    # Pre-trained tokenizer
│   ├── ui/
│   │   ├── app.py               # Flask web UI (auto-loads trained checkpoints)
│   │   └── templates/chat.html  # Chat interface
│   ├── kaggle/
│   │   └── continuum_100m_training.ipynb  # 📓 Kaggle training notebook
│   ├── colab/
│   │   └── continuum_100m_training.ipynb  # 📓 Colab training notebook
│   └── requirements.txt
├── checkpoints/                  # Saved model checkpoints (gitignored)
├── .github/workflows/            # CI: syntax checks + regression tests
├── continuum-slm-architecture.md # 📖 Complete architecture document
└── README.md                     # This file
```

---

## 🛡️ Verification & Hardening

The codebase went through a full audit; every finding below is locked in by a regression test that runs in CI:

| Area | Guarantee |
|---|---|
| **INT8 quantization** | Quantized weights/scales are registered buffers — `state_dict()` round-trips are complete; fused-KV hot path is quant-format aware |
| **Attention backends** | FlexAttention used only where fully supported (CUDA); CPU transparently falls back to SDPA; `torch.compile` failures degrade to eager instead of crashing |
| **Training entry points** | `run.py train` consumes dict batches with `-100` label padding — padding tokens never contribute to loss |
| **Parallel training path** | Causal mask, window cache and dynamic ALiBi stay aligned when sequence length ≠ window size |
| **ADL looping** | Window caches update exactly once per generated token, regardless of loop count |
| **PMB persistence** | Memory-bank writes are wired into the live conversation loop (every `chunk_size` tokens) |
| **Speculative decoding** | Draft/target vocabulary mismatches fail fast with a clear error |
| **Conversation manager** | Incremental prompting — recurrent state means history is never double-counted |

Run everything locally:

```bash
cd continuum
OMP_NUM_THREADS=1 python -m pytest \
  inference/test_engine.py model/test_audit_fixes.py \
  model/test_model.py model/test_layers.py model/test_attention.py \
  model/test_vision.py tokenizer/test_tokenizer.py -v
```

---

## ⚡ Performance Optimizations

The training pipeline has been aggressively optimized for Kaggle's T4/P100 GPUs:

| Optimization | Speedup | Details |
|---|---|---|
| **Parallel forward (Phase 2)** | ~2–3× | Batch Perception + Output stages across all tokens |
| **GLT associative scan** | ~O(log n) | Parallel scan replaces O(n) sequential recurrence (FP32-stable) |
| **Bucket sampler** | ~30% less padding | Groups same-length sequences |
| **CUDA Graphs + torch.compile** | ~15–20% | Reduces Python overhead in Core stage |
| **Fused QKV projection** | ~10% | Single matmul instead of 3 separate |
| **AMP FP16** | ~2× | Automatic mixed precision on T4 Tensor Cores |
| **Precomputed static anchors** | ~5% | Static K/V computed once per forward pass |
| **Circular buffer window cache** | ~3% | No tensor copies for window K/V |

**Result:** Continuum-Max (102M params) trains in **~1.5 hours** on Kaggle T4 ×2 GPU over 2 epochs of the 52K-example Alpaca dataset.

---

## 📖 Documentation

| Document | Description |
|---|---|
| [`continuum-slm-architecture.md`](./continuum-slm-architecture.md) | **Full 23-section architecture document** — design philosophy, every module explained, training strategy, self-critique, and Section 23: ViGLT Vision |
| [Kaggle Notebook](./continuum/kaggle/continuum_100m_training.ipynb) | One-click training notebook — open in Kaggle and Run All |
| [Colab Notebook](./continuum/colab/continuum_100m_training.ipynb) | Google Colab version (same as Kaggle) |

---

## 🔬 Advanced Usage

### Custom Dataset Training

```python
from continuum.conversation.dataset import ConversationalDataset

# Load your own data (JSONL format: {"instruction": ..., "response": ...})
dataset = ConversationalDataset(
    data_files=["my_data.jsonl"],
    tokenizer=tokenizer,
    max_seq_len=512,
)
```

Or straight from the CLI (⚠️ `run.py train` expects PLAIN TEXT lines; for
Alpaca-style JSONL use the ConversationalDataset API above, or convert):

```bash
python3 continuum/run.py train \
  --data my_data.txt \
  --epochs 10 --batch_size 8 --lr 3e-4 \
  --seq_len_start 32 --seq_len_end 512 \
  --checkpoint_dir checkpoints
```

### Different Model Tier

```python
from continuum.model.model import create_continuum_small, create_continuum_nano

# ~20M parameters — trains in ~30 min on Kaggle
model = create_continuum_small()

# ~5M parameters — trains in ~10 min, runs anywhere
model = create_continuum_nano()
```

### State Checkpointing (Mobile App Lifecycle)

```python
# Save state (Android onPause)
state = model.serialize_state(glt_states, window_caches)
torch.save(state, "conversation_state.pt")

# Restore state (Android onResume)
state = torch.load("conversation_state.pt")
glt_states, window_caches = model.deserialize_state(state, device="cpu")
# Continue generation EXACTLY where you left off — no reprocessing!
```

### Multimodal Vision

```python
import torch
from continuum.model.model import create_continuum_max

# Create multimodal model (~115.5M params)
model = create_continuum_max(with_vision=True)
model.eval()

# Forward pass: image + text
image = torch.randn(1, 3, 224, 224)           # Dummy image
text_ids = torch.randint(0, 16000, (1, 32))   # Text tokens

with torch.no_grad():
    result = model.forward_multimodal(
        pixel_values=image,
        token_ids=text_ids,
        core_max_loops=1,  # 1 for speed, None for ADL
    )
    logits = result["logits"]  # [1, patches + text_len, vocab_size]

# Autoregressive generation with vision
generated_ids, loop_counts = model.generate_multimodal(
    pixel_values=image,
    prompt_ids=torch.tensor([[1, 2, 3]]),
    max_new_tokens=80,
    temperature=0.8,
)
```

**Vision encoder uses the SAME building blocks** as the LLM — GLT, GatedShardFFN, RMSNorm — adapted for bidirectional spatial processing (RoPE2D + spatial anchors). Variable image sizes are supported up to a configurable `max_patches` cap for mobile budgets. See the architecture doc for the full design.

---

## 🧠 Design Philosophy (In Brief)

> **The real constraint is not FLOPs — it's bytes moved per token.**

On a phone CPU running a small model at batch size 1, the arithmetic finishes almost instantly — the CPU then sits idle waiting for the next chunk of weights/cache to arrive from RAM. This is **memory-bandwidth-bound execution**, the dominant regime for on-device LLM inference.

**Seven principles derived from this constraint:**

1. ✅ Default mixer must have **O(1) memory** and **O(n) compute** → GLT
2. ✅ Real attention allowed only in **small, bounded doses** → Anchor Attention
3. ✅ Reasoning depth from **reused compute**, not stored parameters → ADL
4. ✅ Every expensive operation must be **conditional** → gating at all levels
5. ✅ **Nothing grows unboundedly** with context length
6. ✅ Training must be **parallelizable** despite recurrent inference
7. ✅ Model is a **mobile app citizen** — process lifecycle is a first-class input

---

## 🤝 Contributing

This is an open-source research project. Contributions welcome!

- **Bug reports & feature requests** — Open a GitHub Issue
- **Pull requests** — For optimizations, fixes, or new features
- **Discussion** — Architecture ideas, training improvements, mobile deployment tips

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgements

- Built with [PyTorch](https://pytorch.org/) — the best deep learning framework
- Training data: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (52K instructions)
- Free GPU compute: [Kaggle](https://kaggle.com) and [Google Colab](https://colab.research.google.com)
- Architecture inspired by first-principles analysis of mobile constraints, not by copying existing designs

---

<div align="center">
  
**Made with ❤️ for on-device AI**

⭐ Star this repo if you find it useful! ⭐

</div>
