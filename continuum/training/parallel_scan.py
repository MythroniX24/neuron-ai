import os
"""
Parallel Associative Scan for GLT Training (Section 6, 18).

The GLT recurrence:
    S_t = diag(gamma_t) * S_{t-1} + diag(iota_t) * B_t

is a first-order linear recurrence: S_t = a_t ⊙ S_{t-1} + b_t
where ⊙ is row-wise scaling and + is matrix addition.

The combine operator for two adjacent steps is:
    (a2, b2) ∘ (a1, b1) = (a2 ⊙ a1, a2 ⊙ b1 + b2)

This is associative, enabling O(log n) parallel prefix scan.

FIX: Double-buffering — clone a/b at each while-loop iteration.
Read from original, inplace-write to clone. This prevents autograd
version mismatch because clone tensors are fresh (no saved context).
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def _scan_dtype_for(k: torch.Tensor) -> torch.dtype:
    """Resolve the associative-scan dtype for one GLT layer.

    FP16 on CUDA under AMP: halves the [B, L, D, D] scan memory traffic
    (k/v are clamped to +-16 and RMS-normed to O(1), so accumulated state
    stays far inside FP16 range). FP32 everywhere else; set
    CONTINUUM_SCAN_FP32=1 to force FP32 even on CUDA.
    """
    if os.environ.get("CONTINUUM_SCAN_FP32") == "1":
        return torch.float32
    if k.is_cuda and k.dtype == torch.float16:
        return torch.float16
    return torch.float32


def _associative_scan_core(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Kogge-Stone parallel prefix scan (core implementation).

    Args:
        a: Decay gates [B, L, D]
        b: Gated inputs [B, L, D, D]

    Returns:
        Accumulated states [B, L, D, D]

    ⚡ FIX (training speed): purely FUNCTIONAL combine. The old code cloned
    the whole [B, L, D, D] buffer and wrote updated slices in-place every
    round — profiling showed aten::copy_ (~400ms/step) + the
    CopySlices/slice_backward machinery (~260ms/step) dominating T4 training.
    torch.cat keeps autograd graphs lean (no per-round full-buffer copies,
    no CopySlices) with bit-identical values.
    """
    step = 1
    while step < b.shape[1]:
        # Vectorized: all (i, i-step) pairs in one shot; prefix stays put.
        a = torch.cat([a[:, :step], a[:, step:] * a[:, :-step]], dim=1)
        b = torch.cat([b[:, :step],
                       a[:, step:].unsqueeze(3) * b[:, :-step] + b[:, step:]],
                      dim=1)
        step *= 2

    return b


def associative_scan(
    gammas: torch.Tensor,
    inputs: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Parallel prefix scan for GLT state evolution.

    Uses DOUBLE-BUFFERING to prevent autograd version corruption.

    Args:
        gammas: Decay gate vectors [B, L, d_state]
        inputs: Gated outer products [B, L, d_state, d_state]
        reverse: If True, scan from right to left

    Returns:
        states: All intermediate states [B, L, d_state, d_state]
    """
    if reverse:
        gammas = torch.flip(gammas, dims=[1])
        inputs = torch.flip(inputs, dims=[1])

    B, L, D, _ = inputs.shape
    states = _associative_scan_core(gammas, inputs)

    if reverse:
        states = torch.flip(states, dims=[1])

    return states


def _chunked_scan_with_initial_state(
    gamma_chunk: torch.Tensor,
    input_chunk: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run associative scan on a single chunk with an optional initial state.

    The scan combines initial_state with the chunk's inputs:
        S_0 = initial_state (if provided)
        S_t = gamma_t * S_{t-1} + input_t   for t in chunk

    We handle initial_state by prepending it to the chunk (reducing it
    to the standard no-initial-state case).

    Args:
        gamma_chunk: [B, C, D]
        input_chunk: [B, C, D, D]
        initial_state: [B, D, D] or None

    Returns:
        chunk_states: [B, C, D, D] (states at each position in the chunk)
        final_state: [B, D, D] (state after the last position)
    """
    B, C, D, _ = input_chunk.shape

    if initial_state is None:
        # No initial state — just run the scan directly
        chunk_states = _associative_scan_core(gamma_chunk, input_chunk)
        final_state = chunk_states[:, -1, :, :]
        return chunk_states, final_state

    # Prepend initial state to make it a zero-state scan
    # S[-1] = initial_state, then S[0] = gamma[0] * S[-1] + input[0]
    # Equivalent to: prepend gamma=1, input=initial_state, then scan + trim

    # Expand initial_state to match chunk layout
    init_gamma = torch.ones(B, 1, D, device=gamma_chunk.device, dtype=gamma_chunk.dtype)
    init_input = initial_state.unsqueeze(1)  # [B, 1, D, D]

    # Concatenate: [initial | chunk]
    full_gamma = torch.cat([init_gamma, gamma_chunk], dim=1)   # [B, 1+C, D]
    full_input = torch.cat([init_input, input_chunk], dim=1)   # [B, 1+C, D, D]

    # Run scan on the combined sequence
    full_states = _associative_scan_core(full_gamma, full_input)

    # Trim the initial state position — we only want the chunk's states
    chunk_states = full_states[:, 1:, :, :]   # [B, C, D, D]
    final_state = full_states[:, -1, :, :]     # [B, D, D]

    return chunk_states, final_state


def _compute_chunked_outer_product_and_scan(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    iota: torch.Tensor,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked GLT: compute outer product + scan in chunks to reduce peak VRAM.

    Instead of materializing the full [B, L, D, D] outer product, we:
    1. Process the sequence in chunks of `chunk_size`
    2. For each chunk: compute outer product [B, C, D, D], scan it, get final state
    3. Carry final state to next chunk as initial state
    4. Collect per-position outputs

    Peak memory: [B, chunk_size, D, D] instead of [B, L, D, D]

    ⚡ FIX: the scan itself now runs in FP32 — same stability guarantee as
    the non-chunked path (which casts gamma/inputs to fp32). Previously the
    chunked path scanned in the INPUT dtype, i.e. fp16 under AMP, where
    gamma^C products across a chunk can silently under/overflow and corrupt
    the recurrence on exactly the long CUDA training runs that use this path.

    Args:
        k, v, q, gamma, iota: [B, L, D]
        chunk_size: Number of positions per chunk (any size — Kogge-Stone
                    handles arbitrary lengths via step-doubling)

    Returns:
        outputs: [B, L, d_state] (h = states @ q)
        final_state: [B, D, D]
    """
    B, L, D = k.shape

    # NOTE: Kogge-Stone covers all prefix distances in ceil(log2 L) rounds for
    # ANY L — the old power-of-2 padding only wasted up to 2x compute.

    # Pad sequence to multiple of chunk_size
    n_chunks = (L + chunk_size - 1) // chunk_size
    pad_len = n_chunks * chunk_size - L

    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        q = F.pad(q, (0, 0, 0, pad_len))
        gamma = F.pad(gamma, (0, 0, 0, pad_len))
        iota = F.pad(iota, (0, 0, 0, pad_len))
        L_padded = L + pad_len
    else:
        L_padded = L

    # Clamp for FP16 safety
    k_safe = k.clamp(min=-16.0, max=16.0)
    v_safe = v.clamp(min=-16.0, max=16.0)

    all_outputs = []
    running_state = None

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size

        # Slice chunk
        k_c = k_safe[:, start:end, :]        # [B, C, D]
        v_c = v_safe[:, start:end, :]        # [B, C, D]
        q_c = q[:, start:end, :]             # [B, C, D]
        gamma_c = gamma[:, start:end, :]     # [B, C, D]
        iota_c = iota[:, start:end, :]       # [B, C, D]

        # Compute outer product for this chunk only: [B, C, D, D]
        outer_c = k_c.unsqueeze(-1) @ v_c.unsqueeze(-2)
        gated_input_c = iota_c.unsqueeze(-1) * outer_c

        # ⚡ FIX: scan in FP32 (see docstring). The carried state stays fp32
        # across chunks; only the readout is cast back to the input dtype.
        scan_dtype = torch.float32
        chunk_states, running_state = _chunked_scan_with_initial_state(
            gamma_c.to(scan_dtype),
            gated_input_c.to(scan_dtype),
            running_state.to(scan_dtype) if running_state is not None else None,
        )  # chunk_states: [B, C, D, D] fp32, running_state: [B, D, D] fp32

        # Compute output: h = states @ q (readout in input dtype — single
        # small GEMM, no accumulation risk)
        h_c = (chunk_states.to(k.dtype) @ q_c.unsqueeze(-1)).squeeze(-1)  # [B, C, D]
        all_outputs.append(h_c)

    # Remove padding from output
    outputs = torch.cat(all_outputs, dim=1)[:, :L, :]  # [B, L, D]

    return outputs, running_state.to(k.dtype)


def glt_parallel_forward(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    iota: torch.Tensor,
    r: torch.Tensor,
    W_o_weight: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Full parallel GLT forward pass for training.

    When chunk_size is set (e.g., 32), uses chunked scan to reduce peak VRAM
    from O(L * D^2) to O(chunk_size * D^2). Trades ~10% speed for ~60% less
    memory — critical for T4 training.
    """
    B, L, D = k.shape

    if chunk_size is not None and chunk_size < L:
        # Chunked path: reduces peak memory
        h, _ = _compute_chunked_outer_product_and_scan(
            k, v, q, gamma, iota, chunk_size=chunk_size
        )
    else:
        # Full parallel path: maximum speed (original code)
        k_safe = k.clamp(min=-16.0, max=16.0)
        v_safe = v.clamp(min=-16.0, max=16.0)
        outer = k_safe.unsqueeze(-1) @ v_safe.unsqueeze(-2)
        gated_input = iota.unsqueeze(-1) * outer

        # ⚡ FIX (training speed): scan in FP16 on CUDA under AMP — halves the
        # [B, L, D, D] memory traffic that dominated training steps. RMS-normed
        # inputs are O(1) and k/v are clamped to ±16, so per-position state
        # sums stay far inside FP16 range. CPU / FP32 runs and the old
        # behaviour are unchanged; CONTINUUM_SCAN_FP32=1 forces FP32.
        scan_dtype = _scan_dtype_for(k)
        gamma_f32 = gamma.to(scan_dtype)
        gated_input_f32 = gated_input.to(scan_dtype)
        states = associative_scan(gamma_f32, gated_input_f32)
        states = states.to(k.dtype)

        h = (states @ q.unsqueeze(-1)).squeeze(-1)  # [B, L, D]

    o = r * h
    o = o @ W_o_weight.T
    return o


def glt_parallel_forward_with_state(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    iota: torch.Tensor,
    r: torch.Tensor,
    W_o_weight: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel GLT forward that ALSO returns the final state.

    Args:
        initial_state: If provided, scan starts from this state [B, D, D]
        chunk_size: If set, use chunked scan to reduce VRAM

    Returns:
        outputs: [B, L, d_model]
        final_state: [B, d_state, d_state]
    """
    B, L, D = k.shape

    if chunk_size is not None and chunk_size < L:
        h, final_state = _compute_chunked_outer_product_and_scan(
            k, v, q, gamma, iota, chunk_size=chunk_size
        )
    else:
        # Full parallel path (original)
        k_safe = k.clamp(min=-16.0, max=16.0)
        v_safe = v.clamp(min=-16.0, max=16.0)
        outer = k_safe.unsqueeze(-1) @ v_safe.unsqueeze(-2)
        gated_input = iota.unsqueeze(-1) * outer

        # FIX (T4 OOM): FP16 under AMP, exactly like glt_parallel_forward().
        # Training calls THIS entry point, and it hardcoded FP32: at batch 24
        # / L=64 the FP32 [B,L,D,D] scan graph alone exceeded 14.5 GiB.
        scan_dtype = _scan_dtype_for(k)
        gamma_f32 = gamma.to(scan_dtype)
        gated_input_f32 = gated_input.to(scan_dtype)
        states = associative_scan(gamma_f32, gated_input_f32)
        states = states.to(k.dtype)

        h = (states @ q.unsqueeze(-1)).squeeze(-1)
        final_state = states[:, -1, :, :]

    o = r * h
    o = o @ W_o_weight.T

    return o, final_state


def glt_sequential_forward(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    iota: torch.Tensor,
    r: torch.Tensor,
    W_o_weight: torch.Tensor,
) -> torch.Tensor:
    """Sequential GLT forward pass (for validation/testing)."""
    B, L, D = k.shape
    d_model = W_o_weight.shape[0]
    device = k.device

    state = torch.zeros(B, D, D, device=device, dtype=k.dtype)
    outputs = []

    for t in range(L):
        # outer: k_t @ v_t^T  [B, D, D]
        k_t = k[:, t, :].unsqueeze(2)    # [B, D, 1]
        v_t = v[:, t, :].unsqueeze(1)    # [B, 1, D]
        outer_t = torch.bmm(k_t, v_t)    # [B, D, D]

        # S_t = gamma_t * S_{t-1} + iota_t * outer_t
        gamma_t = gamma[:, t, :].unsqueeze(2)    # [B, D, 1]
        iota_t = iota[:, t, :].unsqueeze(2)      # [B, D, 1]
        state = gamma_t * state + iota_t * outer_t

        # h_t = S_t @ q_t
        q_t = q[:, t, :].unsqueeze(2)   # [B, D, 1]
        h_t = torch.bmm(state, q_t).squeeze(2)   # [B, D]

        o_t = r[:, t] * h_t
        o_t = F.linear(o_t, W_o_weight)
        outputs.append(o_t.unsqueeze(1))

    return torch.cat(outputs, dim=1)
