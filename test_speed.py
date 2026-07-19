"""Test trained model speed: FP32 vs INT8 (quantized)."""
import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

print('='*55)
print('🧠 CONTINUUM-MAX: SPEED TEST')
print('='*55)

# 1. Load model + weights
print('\n1️⃣ Loading model + trained weights...')
from continuum.model.model import create_continuum_max
model = create_continuum_max()
model.eval()
ckpt = torch.load('checkpoints/continuum_max_for_mobile.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'   ✅ global_step={ckpt.get("global_step","?")}')

# 2. Load tokenizer
print('\n2️⃣ Loading tokenizer...')
from continuum.tokenizer.bpe import ContinuumTokenizer
tokenizer = ContinuumTokenizer.load('checkpoints/tokenizer_16k.json')
print(f'   ✅ {tokenizer.vocab_size_actual} tokens')

# 3. Test FP32 (no quantization)
print('\n3️⃣ FP32 MODE (no quantize)')
from continuum.conversation.manager import ConversationManager
manager_fp32 = ConversationManager(model=model, tokenizer=tokenizer, device='cpu', quantize=False)

prompt = "What is artificial intelligence?"
t0 = time.time()
response_fp32 = manager_fp32.chat(prompt, max_new_tokens=20)
t_fp32 = time.time() - t0
print(f'   Response: "{response_fp32[:80]}..."')
print(f'   ⏱ {t_fp32:.1f}s | ~{20/t_fp32:.1f} tok/s')

# 4. Test INT8 (quantized) - create fresh model
print('\n4️⃣ INT8 QUANTIZED MODE')
model_int8 = create_continuum_max()
model_int8.eval()
model_int8.load_state_dict(torch.load('checkpoints/continuum_max_for_mobile.pt', map_location='cpu', weights_only=False)['model_state_dict'])

manager_int8 = ConversationManager(model=model_int8, tokenizer=tokenizer, device='cpu', quantize=True)

t0 = time.time()
response_int8 = manager_int8.chat(prompt, max_new_tokens=20)
t_int8 = time.time() - t0
print(f'   Response: "{response_int8[:80]}..."')
print(f'   ⏱ {t_int8:.1f}s | ~{20/t_int8:.1f} tok/s')

# 5. Summary
print('\n' + '='*55)
print('📊 SPEED COMPARISON')
print('='*55)
speedup = t_fp32 / t_int8 if t_int8 > 0 else 0
print(f'   FP32: {t_fp32:.1f}s ({20/t_fp32:.1f} tok/s)')
print(f'   INT8: {t_int8:.1f}s ({20/t_int8:.1f} tok/s)')
print(f'   🚀 INT8 is {speedup:.1f}x faster!')
print(f'   For phone: use INT8 mode (quantize=True)')
print()
