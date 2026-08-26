"""
Regression tests for the audit fixes in model.py and trainer.py.

Covers:
1. Parallel forward path with sequence length > window size (dynamic ALiBi +
   current-chunk K/V fix) — used to misalign the causal mask / crash.
2. ADL looping must update each core anchor window cache exactly ONCE per
   generated token (used to duplicate the same token N_max times).
3. Trainer accepts plain dict batches (run.py path) and produces finite loss.
4. Validation perplexity sanity (evaluation regression hook).
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from continuum.model.model import create_continuum_nano
from continuum.training.trainer import ContinuumTrainer


def test_forward_parallel_exceeding_window_size():
    """Seq len > window_size must run cleanly with finite logits."""
    torch.manual_seed(0)
    model = create_continuum_nano()
    model.eval()
    assert model.config.window_size < 96

    ids = torch.randint(3, model.config.vocab_size, (1, 96))
    with torch.no_grad():
        out = model.forward_parallel(ids, core_max_loops=1)

    assert out["logits"].shape == (1, 96, model.config.vocab_size)
    assert torch.isfinite(out["logits"]).all(), "Non-finite logits in parallel path"


def test_forward_parallel_batched():
    """Batch > 1 through the parallel path."""
    torch.manual_seed(1)
    model = create_continuum_nano()
    model.eval()

    ids = torch.randint(3, model.config.vocab_size, (2, 32))
    with torch.no_grad():
        out = model.forward_parallel(ids, core_max_loops=1)

    assert out["logits"].shape == (2, 32, model.config.vocab_size)
    assert torch.isfinite(out["logits"]).all()


def test_adl_window_cache_single_update_per_token():
    """Each generated token must append exactly ONE row to every window cache.

    Shift-by-one check: after a second single-token step, new row j must equal
    old row j+1. The pre-fix ADL bug appended the same token once PER LOOP,
    producing a shift of n_loops rows instead.
    """
    torch.manual_seed(2)
    model = create_continuum_nano()
    model.eval()  # eval + default max_loops exercises the full ADL path

    glt_states, window_caches = model.init_states(1)

    ids1 = torch.randint(3, model.config.vocab_size, (1, 1))
    with torch.no_grad():
        model.forward(ids1, glt_states, window_caches)
    snap = [(wk.clone(), wv.clone()) for wk, wv in window_caches]

    ids2 = torch.randint(3, model.config.vocab_size, (1, 1))
    with torch.no_grad():
        model.forward(ids2, glt_states, window_caches)

    for (wk_old, wv_old), (wk_new, wv_new) in zip(snap, window_caches):
        assert torch.allclose(wk_new[0, :-1], wk_old[0, 1:], atol=1e-5), \
            "Window K shifted by more than one row — ADL duplicated cache updates"
        assert torch.allclose(wv_new[0, :-1], wv_old[0, 1:], atol=1e-5), \
            "Window V shifted by more than one row — ADL duplicated cache updates"


def _make_trainer(tmpdir):
    model = create_continuum_nano()
    return ContinuumTrainer(
        model,
        learning_rate=1e-4,
        checkpoint_dir=tmpdir,
        device="cpu",
        use_amp=False,
        use_parallel_forward=True,
    )


def _dict_batch(model, batch_size=2, seq_len=16):
    vocab = model.config.vocab_size
    input_ids = torch.randint(3, vocab, (batch_size, seq_len))
    labels = input_ids.clone()
    # Simulate right-padding; padded labels must be ignored by the loss.
    input_ids[:, -2:] = 0
    labels[:, -2:] = -100
    return {"input_ids": input_ids, "labels": labels}


def test_train_step_accepts_dict_batches():
    """train_step must consume {"input_ids","labels"} dicts (run.py path).

    Before the fix, TensorDataset-style tuple batches crashed instantly with
    TypeError, breaking `python run.py train` entirely.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        batch = _dict_batch(trainer.model)

        metrics = trainer.train_step(
            batch, seq_len_curriculum=16, accumulation_step=1, total_accumulation_steps=1
        )
        assert math.isfinite(metrics["loss"]), "Loss is NaN/Inf on first step"

        metrics = trainer.train_step(
            batch, seq_len_curriculum=16, accumulation_step=1, total_accumulation_steps=1
        )
        assert math.isfinite(metrics["loss"]), "Loss is NaN/Inf on optimizer step"


def test_validation_perplexity_sanity():
    """val_ppl must be a positive finite number (evaluation baseline hook)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        batch = _dict_batch(trainer.model, batch_size=1, seq_len=12)

        metrics = trainer.validate([batch], max_batches=1)
        ppl = metrics["val_ppl"]
        assert math.isfinite(ppl) and ppl > 0, f"Invalid val_ppl: {ppl}"
        # Random-init nano over ~10 positions: bounded well below pathological values.
        assert ppl < 1_000_000, f"Pathological val_ppl: {ppl}"
