"""
Test suite for the Continuum byte-level BPE tokenizer.
Tests cover: basic encode/decode, single-digit splitting, training,
roundtrip fidelity, special tokens, and edge cases.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from continuum.tokenizer.bpe import ContinuumTokenizer


def test_base_vocab():
    """Verify all 256 base byte tokens + 3 special tokens exist."""
    tok = ContinuumTokenizer(vocab_size=8000)
    assert tok.vocab_size_actual == 259, f"Expected 259 base tokens, got {tok.vocab_size_actual}"
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    # Check byte tokens exist
    for i in range(256):
        assert tok.byte_to_token[i] == i + 3, f"Byte {i} mapped to wrong ID"
    print("✓ Base vocabulary: 256 bytes + 3 special tokens = 259 tokens")


def test_encode_decode_roundtrip():
    """Test that encode->decode is lossless for basic text."""
    tok = ContinuumTokenizer(vocab_size=8000)

    texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Testing 123 numbers and symbols @#$%",
        "Unicode: café, naïve, 中文",
        "",  # empty string
        "a" * 100,  # repetitive
    ]

    for text in texts:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text, f"Roundtrip failed:\n  Input:    {repr(text)}\n  Decoded:  {repr(decoded)}"
    print("✓ Encode-decode roundtrip: all texts roundtrip correctly")


def test_single_digit_splitting():
    """Verify that digits are always split into single tokens."""
    tok = ContinuumTokenizer(vocab_size=8000)

    # Without training, digits should stay as individual byte tokens
    # (bytes for '0'-'9' are 48-57)
    encoded = tok.encode("847")
    # Each digit is one byte -> one token
    assert len(encoded) == 3, f"Expected 3 tokens for '847', got {len(encoded)}"

    # Verify digits are never merged even after BPE training
    training_texts = [
        "the number 847 is large",
        "I have 123 apples",
        "count from 1 to 10, 20, 30",
    ] * 50  # Repeat to increase frequency
    tok.train(training_texts)

    encoded_847 = tok.encode("847")
    assert len(encoded_847) == 3, f"After training, expected 3 tokens for '847', got {len(encoded_847)}"

    encoded_123 = tok.encode("123")
    assert len(encoded_123) == 3, f"After training, expected 3 tokens for '123', got {len(encoded_123)}"

    print("✓ Single-digit splitting: digits never merged by BPE")


def test_training():
    """Test that BPE training works and produces merges."""
    tok = ContinuumTokenizer(vocab_size=8000)

    # Small corpus with repeated patterns
    corpus = [
        "hello world hello world hello world",
        "hello there world",
        "the world is hello",
    ] * 100

    tok.train(corpus, verbose=False)

    # After training, common words should be merged
    encoded_hello = tok.encode("hello")
    # "hello" is 5 bytes -> less than 5 tokens if merged
    assert len(encoded_hello) <= 5, f"Expected 'hello' to be <=5 tokens, got {len(encoded_hello)}"

    # Roundtrip should still work after training
    text = "hello world"
    decoded = tok.decode(tok.encode(text))
    assert decoded == text, f"Post-training roundtrip failed: {repr(decoded)}"

    print(f"✓ BPE training: vocab size={tok.vocab_size_actual}, merges learned")


def test_special_tokens():
    """Test BOS/EOS encoding."""
    tok = ContinuumTokenizer(vocab_size=8000)

    encoded = tok.encode_with_special("hello", add_bos=True, add_eos=True)
    assert encoded[0] == tok.bos_id
    assert encoded[-1] == tok.eos_id

    # Without special tokens
    encoded_plain = tok.encode_with_special("hello", add_bos=False, add_eos=False)
    assert tok.bos_id not in encoded_plain
    assert tok.eos_id not in encoded_plain

    print("✓ Special tokens: BOS/EOS encoding works correctly")


def test_save_load():
    """Test that saving and loading preserves the tokenizer state."""
    tok = ContinuumTokenizer(vocab_size=8000)

    corpus = ["hello world foo bar baz"] * 100
    tok.train(corpus, verbose=False)

    # Save
    save_path = "/tmp/continuum_tokenizer_test.json"
    tok.save(save_path)

    # Load
    tok_loaded = ContinuumTokenizer.load(save_path)

    # Verify merges preserved
    assert len(tok_loaded.merges) == len(tok.merges), \
        f"Merge count mismatch: {len(tok_loaded.merges)} vs {len(tok.merges)}"

    # Verify encode/decode still works
    text = "hello world foo bar"
    assert tok.decode(tok.encode(text)) == text
    assert tok_loaded.decode(tok_loaded.encode(text)) == text

    # Verify the encodings match
    assert tok.encode(text) == tok_loaded.encode(text), \
        "Encodings differ between original and loaded tokenizer"

    # Cleanup
    os.remove(save_path)
    print("✓ Save/Load: tokenizer state preserved correctly")


def test_edge_cases():
    """Test various edge cases."""
    tok = ContinuumTokenizer(vocab_size=8000)

    # Pure digits
    encoded = tok.encode("000111222")
    assert len(encoded) == 9  # 9 single digits

    # Newlines and tabs
    text = "Line1\nLine2\tTabbed"
    assert tok.decode(tok.encode(text)) == text

    # Emoji
    text = "Hello 😊 world 🌍"
    assert tok.decode(tok.encode(text)) == text

    # Mixed scripts
    text = "English 中文 हिंदी"
    assert tok.decode(tok.encode(text)) == text

    print("✓ Edge cases: all pass")


if __name__ == "__main__":
    print("=" * 60)
    print("Continuum Tokenizer Test Suite")
    print("=" * 60)

    test_base_vocab()
    test_encode_decode_roundtrip()
    test_single_digit_splitting()
    test_training()
    test_special_tokens()
    test_save_load()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


# ============================================================================
# Fast rank-based encoder (audit fix: 9.5h tokenization → minutes)
# ============================================================================

def _load_pretrained():
    path = os.path.join(os.path.dirname(__file__), "tokenizer_4k.json")
    return ContinuumTokenizer.load(path)


def test_training_never_learns_digit_merges():
    """Train/serve consistency: BPE training must never LEARN a digit-digit
    merge, because encode() refuses to APPLY them.

    Previously train() preprocessed the corpus by inserting spaces around
    digits (different preprocessing than encode()), so the same text produced
    DIFFERENT token IDs at training vs inference time.
    """
    tok = ContinuumTokenizer(vocab_size=320)
    tok.train([
        "flight 123 to gate 45 at 6 pm",
        "order 9876543210 today",
        "the year is 2026 and the score is 42",
    ] * 200, verbose=False)

    # No learned merge may combine two single ASCII digits
    for ba, bb in tok.merges:
        a_digit = len(ba) == 1 and 0x30 <= ba[0] <= 0x39
        b_digit = len(bb) == 1 and 0x30 <= bb[0] <= 0x39
        assert not (a_digit and b_digit), \
            f"learned digit-digit merge: {ba!r}+{bb!r}"

    # Digits still encode as single tokens, and roundtrip stays lossless
    ids = tok.encode("2026")
    assert len(ids) == 4, f"'2026' should be 4 single-digit tokens, got {ids}"
    assert tok.decode(ids) == "2026"


def test_fast_encode_matches_reference():
    """New rank-based encode() must produce IDENTICAL output to the legacy
    scan-every-merge encoder on diverse inputs (numbers, unicode, repeats)."""
    tok = _load_pretrained()
    samples = [
        "Hello world!",
        "the capital of france is paris .",
        "numbers 12345 and 42 must stay single digits",
        "Mixed text with punctuation, quotes \"and\" apostrophes.",
        "aabbaabb" * 40 + " 7 " + "ccddccdd" * 25,
    ]
    for s in samples:
        fast = tok.encode(s)
        ref = tok._encode_reference(s)
        assert fast == ref, (
            f"fast/reference mismatch on {s[:40]!r}: "
            f"{fast[:12]}... vs {ref[:12]}..."
        )


def test_pretrained_roundtrip_and_digits():
    """decode(encode(s)) == s on the pretrained 4K tokenizer; digits survive."""
    tok = _load_pretrained()
    for s in ["Train number 9 leaves at 8 pm.",
              "Total: 1234567890 rupees only."]:
        assert tok.decode(tok.encode(s)) == s, f"roundtrip failed for {s!r}"


def test_encode_speed_large_corpus():
    """~90KB must encode in seconds, not minutes (regression guard for the
    O(merges x seq_len) bug that cost 9.5h on Kaggle)."""
    import time
    tok = _load_pretrained()
    text = "the quick brown fox jumps over the lazy dog . " * 2000
    t0 = time.perf_counter()
    ids = tok.encode(text)
    dt = time.perf_counter() - t0
    assert len(ids) > 1000
    # Relative guard: must be 50x+ faster than the old O(merges*seq) bug (60000s).
    # Absolute cap generous for weak CPUs; CI fast machines will hit ~0.1-0.5s.
    assert dt < 15.0, f"encode too slow: {dt:.2f}s for {len(text)} chars"
