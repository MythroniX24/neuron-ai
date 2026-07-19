"""Speed test: optimized inference engine with INT8."""
import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

print('=' * 55)
print('🧠 CONTINUUM-MAX: OPTIMIZED INFERENCE TEST')
print('=' * 55)

# 1. Create model + load weights
print('\n1️⃣ Loading model + trained weights...')
from continuum.model.model import create_continuum_max
model = create_continuum_max()
model.eval()
ckpt = torch.load('checkpoints/continuum_max_for_mobile.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'   ✅ global_step={ckpt.get("global_step","?")}')

# 2. Tokenizer
print('\n2️⃣ Loading tokenizer...')
from continuum.tokenizer.bpe import ContinuumTokenizer
tokenizer = ContinuumTokenizer.load('checkpoints/tokenizer_16k.json')
print(f'   ✅ {tokenizer.vocab_size_actual} tokens')

# 3. Test INT8 quantized inference
print('\n3️⃣ Testing INT8 quantized inference...')
from continuum.inference.engine import ContinuumInference

engine = ContinuumInference(
    model=model,
    tokenizer=tokenizer,
    device='cpu',
    quantize=True,  # ⚡ INT8 enabled
)

# Warmup - first inference is slower (dequantization cache)
print('\n   Warming up (first inference populates INT8 cache)...')
response = engine.generate("Hello", max_new_tokens=5, stream=False)
print(f'   Warmup response: "{response[:50]}..."')

# Actual speed test
print('\n4️⃣ Speed test (5 tokens)...')
prompt = "What is AI?"
t0 = time.time()
response = engine.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=40, stream=False)
elapsed = time.time() - t0
tok_per_sec = 10 / elapsed if elapsed > 0 else 0
print(f'   Response: "{response}"')
print(f'   ⏱ {elapsed:.1f}s for 10 tokens = {tok_per_sec:.1f} tok/s')

# 5. Cleanup
del engine, model
print('\n✅ TEST COMPLETE')
