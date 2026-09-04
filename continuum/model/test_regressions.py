"""
Regression tests for bugs found in the full-project audit.

1. GLT state slot indexing: glt_states is indexed by BLOCK POSITION (anchors
   hold None), but the stage runners used to address it by GLT-count. Whenever
   an anchor precedes a GLT in a stage (every tier: nano core, max
   perception/core/output), layers read/wrote the WRONG slot — and in ADL
   inference mode the first-pass restore then wiped the core's recurrent
   memory to None EVERY token, so the reasoning core had no cross-token memory.

2. forward_parallel future leakage: the parallel perception stage wrote the
   chunk's LAST ws tokens into the core anchors' window caches, so the
   per-token core loop attended to FUTURE tokens for the first ws positions
   of every training sample.

3. Layer interleaving contract: the C++ port re-derives anchor positions from
   config — this test pins the exact Python interleaving it must match.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from continuum.model.model import create_continuum_nano, create_continuum_max


def test_core_glt_state_survives_adl_forward():
    """ADL-mode forward must leave the core GLT recurrent state intact.

    Bug: glt_states is indexed by BLOCK POSITION (anchors hold None slots),
    but the stage runners used to address it by GLT-count. Whenever an anchor
    precedes a GLT inside a stage (e.g. max-tier core [GLT, Anchor, GLT] or
    perception [GLT, GLT, Anchor, GLT]) the GLTs after the anchor read the
    wrong slot — and in ADL inference the first-pass restore then wiped the
    core's recurrent memory to None every token, so the reasoning core had
    no cross-token memory.
    """
    model = create_continuum_nano()
    model.eval()

    core_start = len(model.perception_blocks)
    core_glt_positions = [
        core_start + j for j, b in enumerate(model.core_blocks) if b.is_glt
    ]
    assert core_glt_positions, "test assumes a GLT layer inside the core stage"

    with torch.no_grad():
        # Default core_max_loops=None -> full ADL path (n_max_loops=3)
        r1 = model.forward(torch.tensor([[1234]]))

    for pos in core_glt_positions:
        s = r1["glt_states"][pos]
        assert s is not None, f"core GLT state at block {pos} is None after forward"
        assert s.abs().sum() > 0, (
            f"core GLT state at block {pos} was wiped — recurrent memory "
            f"never carries across tokens in ADL mode"
        )


def test_forward_parallel_core_attends_no_future_tokens():
    """Core anchors must start from EMPTY window caches in forward_parallel.

    Bug: the parallel perception stage populated the core anchors' window
    caches with the chunk's LAST ws tokens, so the per-token core loop
    attended to FUTURE tokens at early positions (training leak).
    """
    import continuum.model.model as mm

    model = create_continuum_nano()
    model.eval()

    core_anchor_mixers = [b.mixer for b in model.core_blocks if b.is_anchor]
    assert core_anchor_mixers, (
        "test assumes an anchor layer inside the core stage "
        "(nano core is [GLT, Anchor])"
    )

    first_call_window_norm = {"value": None}

    orig = mm.TransformerBlock.forward_anchor

    def spy(self, x, wk, wv, pmb_readouts=None):
        if self in model.core_blocks and first_call_window_norm["value"] is None:
            first_call_window_norm["value"] = wk.abs().sum().item()
        return orig(self, x, wk, wv, pmb_readouts)

    mm.TransformerBlock.forward_anchor = spy
    try:
        with torch.no_grad():
            # NOTE: core_max_loops=None (full ADL) — the leak lives in the
            # PER-TOKEN core loop; with core_max_loops=1 the core runs a fully
            # parallel stage that never reads the window caches at all.
            model.forward_parallel(torch.randint(0, 8000, (1, 64)))
    finally:
        mm.TransformerBlock.forward_anchor = orig

    assert first_call_window_norm["value"] is not None, "core anchor never ran"
    assert first_call_window_norm["value"] == 0.0, (
        f"core anchor's first window cache holds future tokens "
        f"(norm={first_call_window_norm['value']:.3f}) — training leak"
    )


def test_anchor_positions_match_build_contract():
    """Pin the exact Python layer interleaving the C++ port must replicate.

    Python's _build_stage(): anchors where (abs_idx+1) % interval == 0 with
    interval 3 in perception/output and 2 in the core, plus a final-layer
    fallback. The C++ port re-derives these positions from config.
    """
    model = create_continuum_max()
    cfg = model.config
    blocks = (list(model.perception_blocks) +
              list(model.core_blocks) +
              list(model.output_blocks))

    expected = []
    glt_used = anchor_used = 0
    for abs_idx in range(cfg.n_layers):
        if abs_idx < cfg.perception_layers:
            interval = 3
        elif abs_idx < cfg.perception_layers + cfg.core_layers:
            interval = 2
        else:
            interval = 3
        is_anchor = False
        if anchor_used < cfg.anchor_layers:
            if glt_used >= cfg.glt_layers:
                is_anchor = True
            elif (abs_idx + 1) % interval == 0:
                is_anchor = True
            elif abs_idx == cfg.n_layers - 1:
                is_anchor = True
        if is_anchor:
            anchor_used += 1
            expected.append(abs_idx)
        else:
            glt_used += 1

    actual = [i for i, b in enumerate(blocks) if b.is_anchor]
    assert actual == expected, f"anchor positions {actual} != expected {expected}"
    assert actual == [2, 5, 8], f"unexpected max-tier interleaving: {actual}"