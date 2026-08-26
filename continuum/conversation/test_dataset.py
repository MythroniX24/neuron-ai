"""
Regression tests for ConversationalDataset tokenization.

1. Multiprocessing worker output must be IDENTICAL to serial tokenization
   (same token IDs, same -100 label masks).
2. Disk-cache save/load round-trip must preserve samples exactly.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from continuum.conversation import dataset as D
from continuum.conversation.dataset import ConversationalDataset
from continuum.tokenizer.bpe import ContinuumTokenizer


def _load_tok():
    path = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tokenizer_4k.json")
    return ContinuumTokenizer.load(path)


def _make_convs(n=10):
    convs = []
    for i in range(n):
        convs.append([
            {"role": "system", "content": "You are Continuum, a helpful assistant."},
            {"role": "user", "content": f"Question {i}: explain topic {i} with numbers 12345 please?"},
            {"role": "assistant", "content": f"Topic {i} is about testing tokenization with value {i}2345."},
        ])
    return convs


def test_mp_worker_matches_serial():
    """_mp_format_conversation (worker path) == _format_and_tokenize (serial)."""
    tok = _load_tok()
    D._mp_worker_init(tok, 512)

    for conv in _make_convs(8):
        ds = ConversationalDataset(tokenizer=tok, max_seq_len=512)
        expected = ds._format_and_tokenize(conv)
        got = D._mp_format_conversation(conv)
        assert got == expected, (
            f"Worker output differs from serial for conv: {got[:1]} vs {expected[:1]}"
        )
        assert len(expected) > 0, "test conversations produced no samples"


def test_save_load_roundtrip():
    """Dataset disk cache must preserve samples exactly."""
    import tempfile
    tok = _load_tok()
    ds = ConversationalDataset(tokenizer=tok, max_seq_len=512)
    for conv in _make_convs(6):
        ds.add_sample(conv)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.pt")
        ds.save(path)
        ds2 = ConversationalDataset.load(path, tok)

    assert ds2.samples == ds.samples, "save/load changed the samples"
