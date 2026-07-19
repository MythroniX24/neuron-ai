# 🚀 Continuum SLM — Mobile Speed Master Plan (Target: 20-50 tok/s)

> **Current:** 0.3 tok/s, garbage output (broken tokenizer)  
> **Target:** 20-50 tok/s, coherent Hindi/English on Android phone  
> **Model:** 102M params, 10.1 MB INT8 (5 MB INT4)

---

## 🔴 PHASE 1: FIX THE FUNDAMENTALS (Must Do First)

### 1.1 Fix Tokenizer — Root Cause of Garbage Output

**Problem:** Tokenizer has only 259 tokens (raw bytes). Model expects 16,000.  
Any token ID > 258 → `"?"` → all output is garbage.

**Fix in Kaggle Notebook CELL 3:**
```python
# Train BPE tokenizer on the full dataset
from continuum.tokenizer.bpe import ContinuumTokenizer

tokenizer = ContinuumTokenizer(vocab_size=16000)
tokenizer.train([
    item['prompt'] + item['response'] 
    for item in dataset.samples
], vocab_size=16000)
tokenizer.save("checkpoints/tokenizer_16k.json")
```

**Verify:**
```python
print(tokenizer.vocab_size_actual)  # Must be > 1000, ideally 16000
```

### 1.2 Retrain Model — More Epochs

**Problem:** Only 0.18 epochs (12K steps), model hasn't learned language.

**Fix:**
- Minimum: **3 epochs** (~36K steps)
- Recommended: **5 epochs** (~60K steps)
- Time on T4×2 GPU: ~5-8 hours

**Kaggle setup:**
```python
NUM_EPOCHS = 5  # Was 3, now 5 for quality
```

### 🎯 Phase 1 Result: Coherent text generation (but still slow, ~0.3 tok/s)

---

## 🟡 PHASE 2: PYTHON SPEED — Already Coded, Just Enable!

All these optimizations are ALREADY implemented. Just use them with the retrained model.

### 2.1 INT4 Quantization (2x speed, 5 MB model)
```python
# In engine.py — already supported!
QuantizedLinear(linear, bits=4)  # INT4 = half the memory bandwidth of INT8
```

### 2.2 Speculative Decoding (3-5x, zero quality loss)
```python
from continuum.inference.engine import ContinuumSpeculativeDecoder

decoder = ContinuumSpeculativeDecoder(
    draft_model=create_continuum_nano(),   # 3M params, blazing fast
    target_model=create_continuum_max(),   # 100M params, verifies accuracy
    tokenizer=tokenizer,
    device="cpu",  # Mobile!
    num_draft_tokens=5
)
response = decoder.generate("Hello!", max_new_tokens=50)
print(f"Acceptance rate: {decoder.acceptance_rate:.1%}")  # Target: 70-90%
```

### 2.3 torch.compile (1.5-2x)
```python
engine = ContinuumInference(
    model, tokenizer=tokenizer,
    device="cpu",
    quantize=True,
    use_compile=True,          # ← Enable!
    use_max_autotune=False     # ← ARM doesn't support max-autotune
)
```

### 🎯 Phase 2 Result: 2-8 tok/s (INT4 + Spec Decode + compile)

---

## 🟢 PHASE 3: MOBILE-SPECIFIC PYTHON OPTIMIZATIONS

These need to be implemented. They're Python-only, achievable in 2-3 days.

### 3.1 ARM NEON-Aware Compilation
```python
# Use inductor CPU backend with ARM tuning
torch.backends.mkldnn.is_available()  # Check
os.environ["TORCHINDUCTOR_CPP_WRAPPER"] = "1"  # Enable C++ codegen
```

### 3.2 Reduce Model Size (102M → 50M)
Create a `create_continuum_mobile()` variant:
- d_model=512 (was 768) → 45M params
- FFN expansion=3 (was 4)
- 8 layers (was 12)
- This gives 2x speed + 2x less RAM

### 3.3 Thread Pool Optimization
```python
torch.set_num_threads(4)  # Mobile CPUs have 4-8 cores
torch.set_num_interop_threads(2)
```

### 3.4 Pre-computed Embeddings Cache
Cache the embedding lookup table so token→embedding is O(1) lookup, not matrix multiply.

### 🎯 Phase 3 Result: 5-20 tok/s

---

## 🔵 PHASE 4: C++/GGUF — THE REAL 20-50 TOK/S

This is the ONLY path to 50 tok/s. Pure Python has a ceiling at ~10-20 tok/s.

### 4.1 Export to GGUF Format

**What GGUF is:**
- Binary format used by llama.cpp
- Memory-mapped (mmap) = instant load, no RAM copy
- Supports INT4/Q4_K_M quantization
- SIMD-optimized C++ inference (ARM NEON on phones)

**Challenge:** llama.cpp only supports Transformer architectures (GPT, Llama, Mistral, etc.)  
Our architecture is custom: GLT (matrix recurrence) + Anchor Attention + GatedShardFFN

**Solution:** Use GGML library (the tensor backend behind llama.cpp) to implement our custom ops.

### 4.2 Architecture Porting Plan

| Module | GGML Equivalent | Difficulty |
|---|---|---|
| Linear layers | `ggml_mul_mat` | ✅ Easy |
| RMSNorm | Custom ggml op (~50 lines C) | 🟡 Medium |
| GLT recurrence | Custom C kernel (~200 lines) | 🔴 Hard |
| Anchor Attention | SDPA + sliding window (~300 lines C) | 🔴 Hard |
| GatedShardFFN | Multiple `ggml_mul_mat` + custom gate | 🟡 Medium |
| PMB | Custom C struct + kernels (~200 lines) | 🟡 Medium |
| ADL loops | C for-loop + halting head | 🟡 Medium |

**Total estimated code:** ~2000 lines of C/C++
**Time estimate:** 4-8 weeks for experienced C++ dev

### 4.3 Mobile Deployment

```bash
# Build for Android ARM64
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26
make -j8

# Run on phone
adb push continuum-mobile /data/local/tmp/
adb push model.q4_k_m.gguf /data/local/tmp/
adb shell "/data/local/tmp/continuum-mobile -m model.q4_k_m.gguf -p 'Hello'"
```

### 🎯 Phase 4 Result: 20-50 tok/s on Android phone

---

## 📊 SUMMARY: Complete Timeline

| Phase | What | Tok/sec | Time | Status |
|---|---|---|---|---|
| **Phase 1** | Fix tokenizer + retrain 5 epochs | 0.3 | 6-8 hrs on GPU | 🔴 MUST DO |
| **Phase 2** | INT4 + Spec Decode + compile | **2-8** | 0 (already coded!) | ✅ Ready |
| **Phase 3** | Mobile-specific Python opts | **5-20** | 2-3 days | 🔧 New |  
| **Phase 4** | C++/GGUF port | **20-50** 🎯 | 4-8 weeks | 🔧 New |

---

## 🎯 RECOMMENDED PATH (Fastest to 20 tok/s):

1. **This week:** Phase 1 (fix tokenizer, retrain 5 epochs on Kaggle GPU)
2. **Immediately after:** Phase 2 (enable INT4 + Spec Decode) → **5-8 tok/s** ✅
3. **Next 2 weeks:** Phase 3 (mobile Python optimizations) → **10-20 tok/s** ✅
4. **Long-term:** Phase 4 (C++/GGUF port) → **20-50 tok/s** 🎯

---

## 📱 Phone Target Specs:

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| CPU | ARM Cortex-A73 | ARM Cortex-A78/X1 |
| Storage | 200 MB free | 500 MB free |
| Python | Termux + PyTorch 2.0+ | Or C++ binary (Phase 4) |
