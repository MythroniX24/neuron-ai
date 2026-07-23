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

**Neuron AI (Continuum SLM)** is a custom, from-scratch Small Language Model architecture designed for **efficient on-device inference** on mobile CPUs. Unlike models that adapt existing Transformer architectures, Continuum is built on **three original, cooperating mechanisms** designed from first principles for the mobile constraint:

| Component | What It Does | Why It Matters |
|---|---|---|
| **Gated Linear Trace (GLT)** | O(1) memory recurrent backbone — replaces self-attention as the default mixer | Never grows with conversation length |
| **Anchor Attention** | Bounded-size real softmax attention (window + persistent anchors) | Precise recall where recurrence falls short |
| **Adaptive Depth Looping (ADL)** | Shared-weight reasoning core that loops 1–5× per token | More reasoning depth without more parameters |
| **Persistent Memory Bank (PMB)** | Fixed-size content-addressable long-term memory | Survives across app sessions (kill & resume) |
| **🆕 Continuum Vision (ViGLT)** | Bidirectional GLT vision encoder — same primitives | Multimodal: image + text understanding |

**Headline properties:**
- **No growing KV-cache** — ever. O(1) memory per layer regardless of context length.
- **Streaming single-token decode** — no prompt reprocessing.
- **Instant conversation resume** — ~100 KB state checkpoint (not megabytes).
- **~100 MB INT8 model** — fits comfortably on any modern phone.
- **Trainable from scratch** on free GPUs (Kaggle T4 ×2 ~1.5 hours for 100M model).
- **🆕 Multimodal vision** — ~13.5M ViGLT encoder, unified architecture, mobile-friendly.

---

## 🏗️ Architecture at a Glance

```
                                CONTINUUM — MACRO DATA FLOW
                                ============================

  Input text
      |
      v
 +-----------+      +--------------+
 | Tokenizer |----->| Embedding    |    byte-level BPE (16K vocab),
 | (BPE)     |      | (factorized) |    tied input/output, single-digit numbers
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

> 📖 **Full architecture document:** [`continuum-slm-architecture.md`](./continuum-slm-architecture.md) — 22 sections, ~25K words of detailed technical explanation.

---

## 📊 Model Tiers

| Tier | Params | +Vision | `d_model` | Layers | Vocab | Window | FFN Shards | ADL Max |
|---|---|---|---|---|---|---|---|---|
| **Nano** | ~5M | ~7M | 192 | 6 | 8,000 | 48 | 2 | 3 |
| **Small** | ~20M | ~25M | 384 | 8 | 12,000 | 96 | 4 | 4 |
| **Base** | ~50M | ~58M | 576 | 10 | 16,000 | 128 | 4 | 4 |
| **Max** 🏆 | **~102M** | **~115M** | **768** | **12** | **16,000** | **128** | **6** | **5** |

All tiers share the same architecture — scale up or down by changing config values. One codebase, any size.

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
- ✅ Download **67K conversational examples** (Alpaca 52K + Dolly 15K)
- ✅ Train the **102M parameter Continuum-Max** model from scratch
- ✅ Save model checkpoint + tokenizer to `/kaggle/working/checkpoints/`
- ✅ Push results to GitHub (optional, set `GITHUB_TOKEN` secret)
- ✅ Test the model — chat with your freshly trained AI!

**📥 Download your trained model:**
```
Kaggle → Data tab → /kaggle/working/checkpoints/
  → continuum_max_for_mobile.pt  (~389 MB model)
  → tokenizer_16k.json           (tokenizer vocab)
  → training_summary.json        (training logs)
```

### Option 2: Google Colab

1. Open: [Colab Notebook](https://colab.research.google.com/github/MythroniX24/neuron-ai/blob/master/continuum/colab/continuum_100m_training.ipynb)
2. **Runtime → Change runtime type → GPU T4**
3. Click **Run All**

### Option 3: Local (Linux / macOS / WSL)

```bash
# Clone
git clone https://github.com/MythroniX24/neuron-ai.git
cd neuron-ai

# Install
pip install -r continuum/requirements.txt

# Quick test
python3 -c "
from continuum.model.model import create_continuum_nano
model = create_continuum_nano()
print(f'Model: {model.num_params:,} parameters')
"

# Full training (CPU — will be slow, use for small tests)
python3 continuum/run.py
```

---

## 💬 Chat With Your Trained Model

After training, run inference:

```python
from continuum.conversation.manager import ConversationManager
from continuum.model.model import create_continuum_max
from continuum.tokenizer.bpe import BPETokenizer

# Load model
model = create_continuum_max()
model.load_state_dict(torch.load("checkpoints/continuum_max_for_mobile.pt", map_location="cpu"))
model.eval()

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load("checkpoints/tokenizer_16k.json")

# Chat!
manager = ConversationManager(model=model, tokenizer=tokenizer, device="cpu")
response = manager.chat("What is the capital of France?", max_new_tokens=80)
print(response)  # "The capital of France is Paris."
```

**For mobile deployment (INT8 quantized):**
```python
manager = ConversationManager(model=model, tokenizer=tokenizer, device="cpu", quantize=True)
# Model size drops to ~100 MB — runs on any modern phone CPU!
```

---

## 🧩 Project Structure

```
neuron-ai/
├── continuum/
│   ├── __init__.py              # Package init
│   ├── model/
│   │   ├── model.py             # 🧠 Main model: stages, ADL, forward pass
│   │   ├── layers.py            # GLT layer, GatedShardFFN, RMSNorm, FactorizedEmbedding
│   │   ├── attention.py         # Anchor Attention + Persistent Memory Bank
│   │   ├── test_model.py        # Unit tests
│   │   ├── test_layers.py
│   │   ├── test_attention.py
│   ├── training/
│   │   ├── trainer.py           # Training loop, curriculum learning, logging
│   │   ├── losses.py            # ContinuumLoss: CE + ponder cost + sparsity
│   │   ├── parallel_scan.py     # Associative scan for GLT (parallel training!)
│   ├── conversation/
│   │   ├── dataset.py           # Alpaca + Dolly dataset loading, bucket sampler
│   │   ├── manager.py           # Conversation manager (chat interface)
│   │   ├── template.py          # Chat template format
│   ├── inference/
│   │   ├── engine.py            # Inference engine with generation
│   ├── tokenizer/
│   │   ├── bpe.py               # Byte-level BPE tokenizer (train + encode + decode)
│   ├── ui/
│   │   ├── app.py               # Web UI (Flask)
│   │   ├── templates/chat.html  # Chat interface
│   ├── kaggle/
│   │   └── continuum_100m_training.ipynb  # 📓 Kaggle training notebook
│   ├── colab/
│   │   └── continuum_100m_training.ipynb  # 📓 Colab training notebook
├── checkpoints/                  # Saved model checkpoints (local training)
├── output/                       # Training output (logs + checkpoints)
├── continuum-slm-architecture.md # 📖 Complete architecture document
└── README.md                     # This file
```

---

## ⚡ Performance Optimizations

The training pipeline has been aggressively optimized for Kaggle's T4/P100 GPUs:

| Optimization | Speedup | Details |
|---|---|---|
| **Parallel forward (Phase 2)** | ~2–3× | Batch Perception + Output stages across all tokens |
| **GLT associative scan** | ~O(log n) | Parallel scan replaces O(n) sequential recurrence |
| **Bucket sampler** | ~30% less padding | Groups same-length sequences |
| **CUDA Graphs + torch.compile** | ~15–20% | Reduces Python overhead in Core stage |
| **Fused QKV projection** | ~10% | Single matmul instead of 3 separate |
| **AMP FP16** | ~2× | Automatic mixed precision on T4 Tensor Cores |
| **Precomputed static anchors** | ~5% | Static K/V computed once per forward pass |
| **Circular buffer window cache** | ~3% | No tensor copies for window K/V |

**Result:** Continuum-Max (102M params) trains in **~1.5 hours** on Kaggle T4 ×2 GPU with **67K training examples** over 2 epochs.

---

## 📖 Documentation

| Document | Description |
|---|---|
| [`continuum-slm-architecture.md`](./continuum-slm-architecture.md) | **Full 23-section architecture document** — design philosophy, every module explained, training strategy, self-critique, and **Section 23: ViGLT Vision** |
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

### Different Model Tier

```python
from continuum.model.model import create_continuum_small, create_continuum_nano

# ~20M parameters — trains in ~30 min on Kaggle
model = create_continuum_small()

# ~5M parameters — trains in ~10 min
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

### 🆕 Multimodal Vision (NEW!)

```python
import torch
from continuum.model.model import create_continuum_max

# Create multimodal model (~115.5M params)
model = create_continuum_max(with_vision=True)
model.eval()

# Forward pass: image + text
image = torch.randn(1, 3, 224, 224)  # Dummy image
text_ids = torch.randint(0, 16000, (1, 32))  # Text tokens

with torch.no_grad():
    result = model.forward_multimodal(
        pixel_values=image,
        token_ids=text_ids,
        core_max_loops=1,  # 1 for speed, None for ADL
    )
    logits = result["logits"]  # [1, 196+32, vocab_size]

# Autoregressive generation with vision
# (processes image prefix once, then generates token-by-token)
generated_ids, loop_counts = model.generate_multimodal(
    pixel_values=image,
    prompt_ids=torch.tensor([[1, 2, 3]]),  # Text prompt
    max_new_tokens=80,
    temperature=0.8,
)
```

**Vision encoder uses the SAME building blocks** as the LLM — GLT, GatedShardFFN, RMSNorm — just adapted for bidirectional spatial processing. See [`continuum-slm-architecture.md#23`](./continuum-slm-architecture.md) for the full design.

---

## 🧪 Test Suite

```bash
cd continuum
python -m pytest model/test_model.py -v
python -m pytest model/test_layers.py -v
python -m pytest model/test_attention.py -v
python -m pytest tokenizer/test_tokenizer.py -v

# Vision encoder test
python3 -c "
from continuum.model.model import create_continuum_max
import torch
model = create_continuum_max(with_vision=True)
img = torch.randn(1, 3, 224, 224)
out = model.vision_encoder(img)
print(f'Vision output: {out.shape}')  # [1, 196, 768]
print(f'Vision params: {model.vision_encoder.num_params:,}')
"
```

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
- Training datasets: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (52K instructions), [Databricks Dolly](https://github.com/databrickslabs/dolly) (15K conversational examples)
- Free GPU compute: [Kaggle](https://kaggle.com) and [Google Colab](https://colab.research.google.com)
- Architecture inspired by first-principles analysis of mobile constraints, not by copying existing designs

---

<div align="center">
  
**Made with ❤️ for on-device AI**

⭐ Star this repo if you find it useful! ⭐

</div>
