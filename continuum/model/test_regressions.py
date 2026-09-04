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

def test_scan_gradient_checkpointing_matches_plain():
    """Checkpointing the GLT scan branch must reproduce the plain full-scan
    path EXACTLY (logits AND parameter gradients). The T4 OOM fix auto-enables
    checkpointing at large batch x seq; any recompute drift would corrupt
    training silently, so this pins bit-level parity on the CPU path."""
    model = create_continuum_nano()
    model.train()
    ids = torch.randint(0, 8000, (2, 24))

    def run(with_ckpt):
        model.zero_grad(set_to_none=True)
        model._ckpt_scan = with_ckpt
        out = model.forward_parallel(ids, core_max_loops=1)
        out["logits"].float().pow(2).mean().backward()
        # Some params (e.g. FFN gate_head) don't feed a pure-logits loss and
        # keep grad=None — collect clones-or-None and compare below.
        return out["logits"], [
            None if p.grad is None else p.grad.clone() for p in model.parameters()
        ]

    logits_ckpt, grads_ckpt = run(True)
    logits_plain, grads_plain = run(False)

    max_fwd = (logits_ckpt - logits_plain).abs().max().item()
    assert torch.allclose(logits_ckpt, logits_plain, atol=1e-5, rtol=1e-4), (
        f"checkpointed forward drifted from plain scan (max diff {max_fwd:.6f})"
    )
    names = [n for n, _ in model.named_parameters()]
    for name, gc, gp in zip(names, grads_ckpt, grads_plain):
        if gc is None or gp is None:
            # Params untouched by this input keep grad=None (set_to_none) in
            # BOTH runs — that must agree.
            assert gc is None and gp is None, (
                f"gradient presence differs on {name} "
                f"(ckpt={gc is not None}, plain={gp is not None})"
            )
            continue
        assert torch.allclose(gc, gp, atol=1e-5, rtol=1e-4), (
            f"checkpointed gradient drift on {name}: "
            f"max diff {(gc - gp).abs().max().item():.6f}"
        )


def test_parallel_scan_matches_sequential_reference():
    """The O(L^2) einsum scan must reproduce the sequential GLT recurrence
    EXACTLY (outputs AND gradients) — it replaced the Kogge-Stone
    implementation that dominated T4 training steps (~850 ms/step)."""
    import torch as _t
    from continuum.training.parallel_scan import (
        glt_parallel_forward_with_state,
        glt_sequential_forward,
    )
    _t.manual_seed(0)
    B, L, D = 2, 33, 24  # odd L exercises arbitrary-length behavior
    r = _t.sigmoid(_t.randn(B, L, D))
    Wo = _t.randn(D, D)

    def make_params():
        return [
            _t.randn(B, L, D, requires_grad=True),        # k
            _t.randn(B, L, D, requires_grad=True),        # v
            _t.randn(B, L, D, requires_grad=True),        # q
            _t.sigmoid(_t.randn(B, L, D, requires_grad=True)),  # gamma
            _t.sigmoid(_t.randn(B, L, D, requires_grad=True)),  # iota
        ]

    def fresh(vals):
        """Fresh leaf tensors with the SAME values (both paths must consume
        identical inputs; each path needs its own leaves for backward)."""
        return [v.detach().clone().requires_grad_(True) for v in vals]

    base = make_params()  # shared VALUES — not tensors fed to both graphs

    p_p = fresh(base)
    o_p, fs_p = glt_parallel_forward_with_state(*p_p, r, Wo)
    (o_p.sum() + fs_p.sum()).backward()
    g_p = [None if pp.grad is None else pp.grad.clone() for pp in p_p]

    p_s = fresh(base)
    o_s = glt_sequential_forward(*p_s, r, Wo)
    o_s.sum().backward()
    g_s = [None if pp.grad is None else pp.grad.clone() for pp in p_s]

    max_o = (o_p - o_s).abs().max().item()
    assert _t.allclose(o_p, o_s, atol=1e-5, rtol=1e-4), (
        f"einsum scan output drift vs sequential: max diff {max_o:.6f}"
    )
    for name, gc, gs in zip(("k", "v", "q", "gamma", "iota"), g_p, g_s):
        assert gc is not None and gs is not None, f"missing grad on {name}"
        assert _t.allclose(gc, gs, atol=1e-5, rtol=1e-4), (
            f"einsum scan grad drift on {name}: "
            f"max diff {(gc - gs).abs().max().item():.6f}"
        )

    # Final state parity: recompute the recurrence manually to completion.
    # (k, v, gamma, iota below are the *base* tensors via p_p/p_s values.)
    with _t.no_grad():
        kb, vb, qb, gb, ib = base
        S = _t.zeros(B, D, D)
        for t in range(L):
            S = (gb[:, t, :].unsqueeze(2) * S
                 + ib[:, t, :].unsqueeze(2)
                 * (kb[:, t, :].unsqueeze(2) @ vb[:, t, :].unsqueeze(1)))
    assert _t.allclose(fs_p, S, atol=1e-5, rtol=1e-4), (
        f"einsum final_state drift: max diff {(fs_p - S).abs().max().item():.6f}"
    )

    # Initial-state parity vs the (independent, fp32) chunked path — both
    # consume the SAME tensors (no backward on this section).
    p_c = fresh(base)
    S0 = _t.randn(B, D, D)
    o_c, fs_c = glt_parallel_forward_with_state(*p_c, r, Wo,
                                                initial_state=S0, chunk_size=16)
    o_f, fs_f = glt_parallel_forward_with_state(*p_c, r, Wo, initial_state=S0)
    assert _t.allclose(o_f, o_c, atol=1e-5, rtol=1e-4), (
        f"einsum+initial_state output drift vs chunked: "
        f"max diff {(o_f - o_c).abs().max().item():.6f}"
    )
    assert _t.allclose(fs_f, fs_c, atol=1e-5, rtol=1e-4), (
        f"einsum+initial_state final_state drift vs chunked: "
        f"max diff {(fs_f - fs_c).abs().max().item():.6f}"
    )
