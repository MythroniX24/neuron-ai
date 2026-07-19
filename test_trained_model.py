"""
Test trained Continuum-Max model with checkpoint from Kaggle training.
Loads weights, runs inference, shows results.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'continuum_max_for_mobile.pt')
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, 'tokenizer_16k.json')

print('=' * 60)
print('🧠 CONTINUUM-MAX TRAINED MODEL TEST')
print('=' * 60)

# 1. Load checkpoint info first
print('\n1️⃣  Checking checkpoint...')
if not os.path.exists(CKPT_PATH):
    print(f'   ❌ Checkpoint not found at: {CKPT_PATH}')
    print('   Place continuum_max_for_mobile.pt in checkpoints/ folder')
    sys.exit(1)

size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
print(f'   ✅ Checkpoint found: {size_mb:.0f} MB')

# 2. Create model
print('\n2️⃣  Creating model...')
from continuum.model.model import create_continuum_max
model = create_continuum_max()
print(f'   ✅ Model created: {model.num_params:,} parameters')

# 3. Load trained weights
print('\n3️⃣  Loading trained weights (on CPU)...')
device = 'cpu'
checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
global_step = checkpoint.get('global_step', '?')
print(f'   ✅ Weights loaded! (global_step={global_step})')

# 4. Load tokenizer
print('\n4️⃣  Loading tokenizer...')
from continuum.tokenizer.bpe import ContinuumTokenizer
if os.path.exists(TOKENIZER_PATH):
    tokenizer = ContinuumTokenizer.load(TOKENIZER_PATH)
    print(f'   ✅ Tokenizer loaded: {tokenizer.vocab_size_actual} tokens')
else:
    tokenizer = ContinuumTokenizer(vocab_size=16000)
    print(f'   ⚠️ Created fresh tokenizer: {tokenizer.vocab_size_actual} tokens')

# 5. Chat test
print('\n5️⃣  Testing chat inference...')
from continuum.conversation.manager import ConversationManager

model.eval()
model.to(device)

manager = ConversationManager(
    model=model, tokenizer=tokenizer,
    device='cpu', quantize=False  # No INT8 for accuracy test
)

prompts = [
    "What is the capital of France?",
    "Write a short poem about AI.",
    "Explain machine learning in simple terms.",
]

for prompt in prompts:
    print(f'\n   👤 >>> {prompt}')
    try:
        response = manager.chat(prompt, max_new_tokens=60)
        print(f'   🤖 AI: {response}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
        import traceback
        traceback.print_exc()

print('\n' + '=' * 60)
print('✅ TEST COMPLETE')
print('=' * 60)
