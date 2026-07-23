# Continuum: An Adaptive Recurrent Architecture for On-Device Small Language Models

**A ground-up SLM design for CPU-only Android inference, scalable from 5M to 100M parameters.**

---

## Table of Contents

1. Design Philosophy — First-Principles Constraint Analysis
2. Architecture at a Glance
3. The Two-Tier Memory Philosophy
4. Tokenizer
5. Embedding Layer
6. Core Reasoning Block — Gated Linear Trace (GLT)
7. Precision Recall — Anchor Attention
8. Layer Composition & Interleaving
9. Adaptive Depth Looping — The Reasoning Core
10. Feed-Forward Mechanism — Gated Shard FFN
11. Gating Mechanisms — Consolidated
12. Persistent Memory Bank
13. Context Management
14. Memory Compression
15. Output Layer
16. End-to-End Data Flow
17. Parameter Scaling — 5M to 100M
18. Training Strategy
19. Inference Strategy
20. Self-Critique — Known Limitations
21. Alternative Designs Considered and Rejected
22. Goal Traceability & Closing Summary
23. Continuum Vision (ViGLT) — Multimodal Extension

---

## Executive Summary

Continuum is built on three original, cooperating mechanisms, plus a persistent memory module:

1. **Gated Linear Trace (GLT)** — replaces self-attention as the *default* sequence-mixer. It is a gated, matrix-valued associative recurrence with two **decoupled** data-dependent gates (decay and input). It runs in **O(1) memory per step** and **O(n) total compute**, with no growing cache, ever.
2. **Anchor Attention** — a deliberately small, size-bounded *real* softmax-attention mechanism (a short sliding window + a handful of persistent anchor tokens), used sparingly (roughly 1 layer in 3–4) for the one thing recurrence is structurally bad at: precise pairwise recall.
3. **Adaptive Depth Looping (ADL)** — a shared-weight "Reasoning Core" segment that loops itself a variable number of times per token (1 to N_max), controlled by a lightweight halting head. This buys *test-time reasoning depth* without buying *stored parameters* — the central trick for making a ≤100M-parameter model reason above its weight class.
4. **Persistent Memory Bank (PMB)** — a small, fixed-size set of addressable memory slots that store compressed, chunk-level summaries across a conversation (and optionally across app sessions), separate from GLT's fast-decaying working state.

Headline properties: no growing KV-cache under any circumstance; streaming, single-token decode; bounded worst-case per-token latency; instant conversation resume via state checkpointing (a few kilobytes, not megabytes); a single consistent set of scaling knobs from 5M to 100M parameters.

None of GLT, Anchor Attention, ADL, or PMB is a copy of any existing published architecture. Each is a first-principles response to a specific mobile constraint, built using well-established primitives (gated recurrence, sliding-window attention, adaptive computation, content-addressed memory) as raw material, combined in a way that — to the best of my knowledge — has not been proposed as a single system elsewhere. Section 21 makes the lineage and the departures explicit.

---

## 1. Design Philosophy — First-Principles Constraint Analysis

### 1.1 The real constraint is not FLOPs — it's bytes moved per token

The instinctive way to evaluate a language model architecture is FLOPs: how much arithmetic does it do. On a phone CPU running a small model at batch size 1 (a single user, single stream — the only case that matters here), this is the wrong metric. Modern ARM cores can execute billions of multiply-adds per second, but they can only pull a few gigabytes per second from RAM. For a small model doing simple projections and elementwise gates, the arithmetic finishes almost instantly — the CPU then sits idle waiting for the next chunk of weights (or cache) to arrive from memory. This is **memory-bandwidth-bound execution**, and it is the dominant regime for on-device LLM inference at batch size 1. It is *the* reason weight quantization (Section 19) helps so much even without specialized low-precision arithmetic units: shrinking the bytes that must move is a direct, almost 1:1 speedup, independent of FLOPs.

This reframing is the foundation of every decision below. The question asked of every component in this document is not "how many multiplications does this do" but **"how many bytes does this force through memory per generated token, and does that number grow with conversation length?"**

### 1.2 Critiquing existing architecture families against this lens

| Architecture family | Compute / token | Cache/state growth | Precise long-range recall | From-scratch training stability | RAM at a fixed 100M-param budget | Mobile verdict |
|---|---|---|---|---|---|---|
| **Full self-attention (Transformer)** | O(n) — attends over all past tokens | **O(n), unbounded** — KV-cache grows every turn | Strong (any token, one hop) | Mature, well understood | Same as any dense model | Gets slower and hungrier for RAM the longer the chat runs. Directly hostile to "smooth conversation." |
| **Pure linear-recurrent (Mamba/RWKV-style)** | O(1) fixed per step | **O(1), fixed** | Weak — history is compressed lossily into a fixed state | Needs careful gate parametrization, but mature by now | Same as any dense model | Excellent memory profile; recall is a genuine, documented weak point when used alone. |
| **Sparse Mixture-of-Experts** | Sub-linear in *total* params (only active experts compute) | N/A (a compute technique, not a memory one) | Depends entirely on the backbone it wraps | Fragile in practice — load-balancing losses, router collapse | **No savings** — all experts must stay resident in RAM regardless of routing | A real compute win, but at a *fixed* 100M ceiling the RAM benefit MoE is famous for does not apply, and the training fragility is a bad trade for a solo from-scratch run. |
| **Naively shrunk dense Transformer** ("just make GPT smaller") | O(n) growing | O(n) growing | Strong locally, but reasoning does not improve just because it's small | Mature | Same as any dense model | Inherits every mobile weakness of full attention, at a scale too small to compensate with brute-force capacity. |
| **Continuum (this design)** | O(1) backbone + a small, fixed-size attention dose | **O(1)** — fixed state + bounded window/anchors | Weak from the backbone alone; strong where it matters via Anchor Attention + PMB | Designed for stability from the ground up (Section 18) | Fixed by design — nothing grows silently | Purpose-built against exactly this list. |

No single existing family clears every constraint. The instinct to "just pick the best one" is itself the mistake — Transformers, SSMs, and MoE are each excellent at *one specific property* and mediocre-to-bad at the others. The correct move is not selection but **decomposition**: identify which property each family is uniquely good at, take only that property, and pay for it only when it's actually needed.

### 1.3 Design principles derived from this analysis

Every subsequent section is a direct consequence of one or more of these:

1. **The default sequence-mixer must have O(1) memory and O(n) total compute.** This rules out full self-attention as the backbone by construction, not by assumption (Section 6 — Gated Linear Trace).
2. **Real softmax attention is allowed, but only in small, structurally bounded doses**, reserved for the one job recurrence cannot do well: precise pairwise recall (Section 7 — Anchor Attention).
3. **Reasoning depth should come from reused compute, not from stored parameters.** Parameters are the tightest budget in a ≤100M model; test-time compute is comparatively cheap and controllable (Section 9 — Adaptive Depth Looping).
4. **Every expensive operation must be conditional.** Nothing runs at full cost unconditionally if a cheaper default path exists — this is applied at the token level (ADL), the channel level (GLT gates), and the FFN level (Gated Shard FFN, Section 10).
5. **Nothing is allowed to grow unboundedly with context length**, by design, not by convention. Every cache-like structure in this document (the Anchor window, the anchor set, the memory bank) has a hard, fixed size stated up front.
6. **Training must be parallelizable** despite the model being recurrent at inference — a naive token-by-token training unroll is too slow to be trainable on the kind of hardware a solo developer actually has access to (Section 18).
7. **The model is a mobile app citizen, not a research artifact ported to mobile as an afterthought.** Process lifecycle (backgrounding, being killed, resuming), battery, and thread topology are first-class design inputs, not implementation details bolted on later.

The rest of this document builds up from these seven principles, module by module.

---

## 2. Architecture at a Glance

Continuum is organized into three macro-stages that a token passes through in sequence, plus one side-channel (the Persistent Memory Bank) that every stage can read from and that gets written periodically.

```
                                CONTINUUM — MACRO DATA FLOW
                                ============================

  Input text
      |
      v
 +-----------+      +--------------+
 | Tokenizer |----->| Embedding    |    byte-level BPE, small vocab,
 | (BPE)     |      | (factorized) |    tied input/output table
 +-----------+      +--------------+
                           |
                           v
   ======================================================
   |              STAGE 1 : PERCEPTION                   |
   |     [ GLT ] -> [ GLT ] -> [ Anchor Attention ]       |
   |            single fixed-depth pass                  |
   ======================================================
                           |
                           v
   ======================================================
   |         STAGE 2 : REASONING CORE  (looped)           |
   |                                                       |
   |   +-------------------------------------+             |
   |   |  [ GLT ] -> [Anchor Attn] -> [ GLT ] | <----+      |
   |   +------------------+--------------------+     |      |
   |                      |                          |      |
   |                      v                          |      |
   |               [ Halting Head ]                  |      |
   |               /              \                  |      |
   |         confident?        not yet?               |      |
   |             |             (loop again,           |      |
   |             |              up to N_max times) ---+      |
   |             v                                          |
   |     (weights are SHARED across every loop iteration —   |
   |      this is the only place in the network that runs    |
   |      more than once per token, and it costs zero extra   |
   |      stored parameters to do so)                         |
   ======================================================
                           |
                           v
   ======================================================
   |               STAGE 3 : OUTPUT                       |
   |   [GLT] -> [GLT] -> [GLT] -> [ Anchor Attention ]     |
   ======================================================
                           |
                           v
                  +-------------------+
                  | Output projection |   tied embedding,
                  | + softmax         |   small vocab -> cheap
                  +-------------------+
                           |
                           v
                 next-token probability distribution


   Side-channel, crosses every stage:
   +--------------------------------------------------------------+
   |                 PERSISTENT MEMORY BANK (PMB)                  |
   |  read by every Anchor Attention layer through its anchor      |
   |  tokens  --  written once per K-token chunk via a gated,      |
   |  content-addressed update (fixed slot count, never grows)     |
   +--------------------------------------------------------------+
```

**Why three stages, not one uniform stack:** a plain stack of identical layers forces every token through the same amount of computation, which directly violates principle 4 (minimize unnecessary computation). Splitting into Perception → Reasoning Core → Output lets exactly one segment — the one in the middle, where the representation is already reasonably well-formed but the final answer isn't yet committed — carry the *variable*, token-dependent cost. Perception and Output stay fixed-cost and fast, because "turn tokens into a representation" and "turn a representation into a decision" don't generally benefit from looping the way "work through a hard step" does.

**Why GLT is the majority layer type and Anchor Attention is the minority:** GLT is the O(1) workhorse; Anchor Attention is real softmax attention with real (if bounded) cost. The ratio (roughly 1 Anchor layer per 3 GLT layers, tuned per tier in Section 17) is chosen so that the model gets a periodic "precise lookup" checkpoint without paying attention's cost on every layer.

---

## 3. The Two-Tier Memory Philosophy

Before going module-by-module, it's worth naming the memory model explicitly, because three different mechanisms in this document are all "memory" in some sense, and conflating them is a common source of confusion in recurrent-architecture write-ups.

| Tier | Mechanism | Analogy | Lifespan | Size | Section |
|---|---|---|---|---|---|
| **Working memory** | GLT's recurrent state `S_t` | What you're holding in your head *right now* while mid-sentence | Continuously decaying — old content fades exponentially unless refreshed | Fixed per layer (`d_state × d_state`) | 6 |
| **Precision window** | Anchor Attention's local window + anchors | Glancing back at the last paragraph you wrote | Fixed lookback window (tokens fall out once past it) | Fixed (`w + m` tokens) | 7 |
| **Long-term memory** | Persistent Memory Bank | Notes you deliberately wrote down because they mattered | Survives indefinitely (can even persist across app restarts) | Fixed slot count | 12 |

No tier alone is sufficient. Working memory alone forgets everything gradually and uniformly — it can't tell "this fact matters, keep it" from "this was filler." The precision window alone can't see past a few dozen–hundred tokens. Long-term memory alone (with no working memory) would need to write on every single token, which is neither affordable nor useful (most tokens aren't worth committing to permanent storage). Together, they form a strictly more capable and more mobile-appropriate memory system than any single mechanism, and each is cheap precisely because it isn't trying to do the other two tiers' job.

---

## 4. Tokenizer

**Purpose.** Convert raw UTF-8 text into a sequence of discrete integer IDs the model can embed, and convert generated IDs back into text. This is the entry and exit point of the whole system, and at small parameter budgets it has an outsized effect on total model size (Section 5), so it is treated as a first-class design decision, not a default import.

**Input.** Raw text (any UTF-8 string — user messages, system prompts, generated continuations).

**Output.** A sequence of integer token IDs (encode direction) or reconstructed text (decode direction).

**Internal working.**
- **Base mechanism: byte-level BPE.** Every possible byte (0–255) is a guaranteed base token, and BPE merges are learned on top of that byte alphabet. This is a well-established choice (used in the GPT-2/GPT-4/Llama tokenizer lineage) and is adopted here as a standard building block, not claimed as novel — its value is that **there is no out-of-vocabulary case, ever.** Emoji, code symbols, Hinglish code-switching, rare Unicode — everything falls back gracefully to bytes if no merge applies, instead of an `<unk>` token that silently destroys information.
- **Deliberately small vocabulary: 8,000–16,000 tokens depending on tier** (Section 17), versus the 32,000–128,000+ vocabularies common in larger models. This is the one place where Continuum makes a choice a larger model wouldn't: at 100M total parameters, a 128k-entry embedding table would be a large fraction of the entire budget before a single reasoning layer is written (worked out precisely in Section 5). A small vocabulary trades a *known, bounded* cost (slightly more tokens per sentence — quantified in Section 20) for a *large, direct* parameter saving.
- **Single-digit number tokenization.** Digits 0–9 are never merged into multi-digit tokens (e.g. "847" is always three tokens: `8`,`4`,`7`, never one `847` token). This is a small, well-established, cheap intervention specifically aimed at arithmetic reasoning: multi-digit tokenization forces the model to memorize an enormous, inconsistent lookup table of number chunks rather than learning positional (place-value) structure, which is a documented cause of weak arithmetic in small models. Enforcing single-digit splitting costs nothing in vocabulary size and directly helps the "strong reasoning ability" goal.
- **Domain-trainable, not fixed.** The BPE merge table is learned from a training corpus chosen per deployment (general conversational text, code, or a narrower domain) — the mechanism is fixed, the vocabulary content is a training-time artifact that can be regenerated for a different target distribution without touching the architecture.

**Advantages.** No OOV failures; small embedding footprint; better arithmetic behavior for free; fully retrainable/swappable vocabulary without any architectural change.

**Limitations.** A smaller vocabulary means more tokens are needed to encode the same text compared to a large-vocabulary tokenizer — roughly 20–40% more tokens for typical English prose versus a 32k+ vocabulary (this is a real, honest tradeoff, quantified further in Section 20, not eliminated by any trick here). Single-digit splitting also means numeric strings are longer in token count, trading a small amount of sequence-length efficiency for a meaningful gain in arithmetic reliability.

**Why it improves the overall model.** Every other component's parameter budget is calculated *after* the tokenizer's footprint is subtracted (Section 5). Getting the vocabulary size right is the single highest-leverage decision for a small model's usable-capacity-per-parameter, because it's pure overhead that competes directly with the layers that actually do reasoning.

---

## 5. Embedding Layer

**Purpose.** Map each token ID to a dense vector the network can compute with, and (in reverse, tied) map the final hidden state back to a distribution over the vocabulary.

**Input.** A token ID (integer in `[0, vocab_size)`).

**Output.** A vector of dimension `d_model` (the network's working width).

**Internal working.**
A naive embedding table of shape `[vocab_size, d_model]` is often the single largest parameter consumer in a small model. Concretely: at `vocab_size = 12,000` and `d_model = 384`, a naive table costs 4.6M parameters — untied, input *and* output tables would cost 9.2M, i.e. nearly half of a 20M-parameter budget spent before any reasoning layer exists. Continuum addresses this with two standard, well-established techniques used together:

1. **Weight tying.** The same table serves as both the input lookup and (transposed) the output projection. This is standard practice and simply halves the naive embedding cost outright.
2. **Factorization (ALBERT-style).** The table itself is kept at a *smaller* dimension `d_embed < d_model` (e.g. `d_embed = d_model / 4` to `d_model / 6`), and two small linear maps handle the conversion: an up-projection `U : d_embed -> d_model` right after the lookup, and a down-projection `D : d_model -> d_embed` right before the final (tied) output matmul. Table cost becomes `vocab_size × d_embed` instead of `vocab_size × d_model`, at the cost of two small `d_embed × d_model` matrices.

For the example above (`vocab=12,000, d_model=384, d_embed=80`): table cost drops to 0.96M, plus two projections of `80×384=30,720` each (~0.06M) — total ≈ 1.02M, versus 4.6M naive-tied or 9.2M naive-untied. That difference (roughly 3.5M parameters) is worth more than an entire extra layer at this scale.

**Advantages.** Order-of-magnitude reduction in embedding-related parameters at small `vocab`/`d_model` ratios; the saved budget is reallocated to layers that actually compute reasoning; tying also acts as a mild regularizer (the model can't develop wildly different geometries for "reading" versus "writing" a token).

**Limitations.** Factorization adds two matrix multiplies (up-projection once per input token, down-projection once per output step) — negligible in absolute cost given how small `d_embed` is, but not literally free. Very aggressive factorization (`d_embed` too small relative to `vocab_size`) can bottleneck how many distinct tokens the model can represent cleanly; the ratios above are chosen to stay well clear of that regime.

**Why it improves the overall model.** This is the clearest, least controversial parameter-efficiency win in the entire design, and it is taken in full at every tier (Section 17) precisely because it has essentially no downside at this model scale — it converts "wasted" embedding-table capacity directly into reasoning-layer capacity.

---

## 6. Core Reasoning Block — Gated Linear Trace (GLT)

This is the backbone sequence-mixer: the layer type that appears most often in the stack (Section 8) and does the bulk of the "read the conversation so far" work.

**Purpose.** Carry forward a compressed, continuously-updated summary of everything seen so far, in a fixed amount of memory, updated in a fixed amount of compute per token — the direct architectural answer to principle 1 (Section 1.3).

**Input.** The current token's hidden state `x_t` (dimension `d_model`), plus the previous timestep's recurrent state `S_(t-1)` (a `d_state × d_state` matrix, `d_state < d_model`, carried forward internally — not something the rest of the network sees directly).

**Output.** A new hidden state `o_t` (dimension `d_model`) passed to the next layer, plus the updated state `S_t`, carried forward to the next token.

**Internal working.**

```
                    GATED LINEAR TRACE (GLT) — one timestep
                    ========================================

        x_t   (current token's hidden state, dimension d_model)
         |
         +----------+----------+----------+-------------+-------------+
         |          |          |          |             |
         v          v          v          v             v
       W_k x_t    W_v x_t    W_q x_t  sigma(W_g x_t) sigma(W_i x_t)
         |          |          |          |             |
         v          v          v          v             v
        k_t        v_t        q_t     gamma_t         iota_t
     (d_state)  (d_state)  (d_state)  (decay gate,   (input gate,
                                        d_state,       d_state,
                                        range 0..1)    range 0..1)
         |          |                     |              |
         +----+-----+                     |              |
              |                           |              |
              v                           |              |
        outer product                     |              |
         k_t (x) v_t                      |              |
      [ d_state x d_state ]               |              |
              |                           |              |
              +---------------+-----------+--------------+
                              |
                              v
     S_t  =  diag(gamma_t) . S_(t-1)  +  diag(iota_t) . (k_t (x) v_t)
                              |
                              |   <-- carried forward as the recurrent
                              |       state into timestep t+1
                              v
               read:  h_t = S_t . q_t          (matrix-vector product)
                              |
                              v
               output gate:  r_t = sigma(W_r x_t)
                              |
                              v
               o_t = W_o ( r_t * h_t )    -->  passed to the next layer
```

Reading the mechanism in words: each token produces a key `k_t` and value `v_t` (what to remember), and two independent per-channel gates — a **decay gate** `gamma_t` (how much of the *old* state to keep) and an **input gate** `iota_t` (how much of the *new* content to write in). The state update decays the previous matrix state and adds a freshly gated outer product of the new key and value. A query `q_t` then reads out of the current state via a matrix-vector product, an output gate `r_t` decides how much of that recalled content is actually relevant to pass on, and a final projection returns the result to `d_model`.

**The one deliberate, explicit design departure worth flagging:** most published gated-recurrent and state-space designs couple decay and input into a *single* data-dependent parameter (for instance, a single "step size" that simultaneously controls how much old state decays and how much new content is admitted, which has an elegant continuous-time interpretation). GLT instead uses **two independently-learned gates**. This costs a modest, cheap amount of extra parameters (one more `d_model × d_state` matrix) and gives up that continuous-time elegance, in exchange for the ability to represent combinations a coupled gate structurally cannot — e.g. "hold onto old context firmly *and* be slow to accept new information" (useful mid-explanation, when a digression shouldn't overwrite the running thread) versus "let go of old context quickly *and* also be slow to accept new information" (useful right after a topic change, before the model has decided the new topic is worth committing to). This is a genuine tradeoff, not a strict improvement, and is presented as such — Section 21 discusses the coupled alternative directly.

**Training vs. inference — two different modes of the same math.** Because `gamma_t`, `iota_t`, `k_t`, `v_t`, `q_t` are all computed independently per position (from `x_t` alone, not from `S_t`), the recurrence `S_t = diag(gamma_t) S_(t-1) + diag(iota_t)(k_t v_t^T)` is a **linear** recurrence in `S`. This means it admits a **parallel (associative) scan**: during training, an entire sequence's worth of state updates can be computed in parallel chunks on a GPU, rather than one slow sequential step at a time (this is the same mathematical trick that makes S4/Mamba/linear-attention families trainable at all on modest hardware — it is adopted here as a systems technique, not claimed as novel). During inference, the model simply runs the true sequential recurrence one token at a time, which is exactly the streaming behavior mobile decode wants. **Training is parallel; inference is sequential — same formula, two execution modes.** This directly satisfies design principle 6.

**Numerical stability details.** `gamma_t` and `iota_t` are sigmoid-bounded (strictly in `(0,1)`), which guarantees the state can't blow up purely from the recurrence itself. `k_t` and `v_t` are additionally passed through a lightweight RMSNorm before the outer product, preventing the state from drifting to large magnitudes over very long sequences (a practical safeguard commonly needed in linear-attention-style state updates). The decay gate's bias is initialized with a *spread* of starting values across channels — some channels start with a strong bias toward slow decay (long memory), others toward fast decay (short memory) — a simplified, practical borrowing of the general insight behind HiPPO-style initialization (different channels should specialize to different timescales from the start) without adopting the fuller continuous-time HiPPO machinery.

**Advantages.** O(1) memory per layer regardless of conversation length — the state is a fixed `d_state × d_state` matrix whether the conversation is 10 tokens or 10,000. O(n) total compute. Small state size (e.g. 64×64 = 4,096 floats) fits comfortably in L1/L2 cache, so each step's read/write pattern stays cache-resident — a concrete, favorable memory-bandwidth profile, in contrast to a growing KV-cache that eventually has to spill out of cache into slower RAM. No positional encoding needed at all for this layer type — order is inherent to the recurrence itself, unlike attention, which is permutation-invariant and must be told the order externally (this removes both parameters and compute that attention-based designs spend on positional encoding).

**Limitations.** The state is a **lossy** compression of history by construction — it has a bounded information-theoretic capacity (roughly proportional to `d_state²`), shared across everything the model has seen so far. Precise verbatim recall of something from far back in the conversation is not something this layer type can guarantee on its own; that job is explicitly handed to Anchor Attention and the Persistent Memory Bank (Sections 7, 12). The decoupled-gate design costs slightly more parameters than a coupled alternative for a benefit that is genuinely useful but harder to prove decisively superior in the abstract — it is a considered bet, not a certainty (Section 20 revisits this honestly).

**Why it improves the overall model.** GLT is what makes "efficient long-context handling" and "smooth conversation" (Section 1) architecturally guaranteed rather than best-effort: there is no code path in which a long conversation causes this layer type to use more memory or get slower per token. It is also what frees up the model's compute budget to be spent where it actually helps reasoning (Section 9) instead of on an ever-growing attention computation.

---

## 7. Precision Recall — Anchor Attention

Section 1.3's second principle states that real attention is allowed, in small bounded doses, for the one job recurrence structurally can't guarantee: exact pairwise recall. This section is Continuum's direct answer to the prompt's "attention or its alternative" requirement — GLT (Section 6) is the *alternative* used as the default; Anchor Attention is the *real attention*, used sparingly and only where its specific strength is worth its specific cost.

**Purpose.** Give the model a periodic, precise lookup capability — both over the immediate recent context (exact local recall) and over a small set of persistent/salient reference points (near-exact global recall) — without ever letting the attended set size, and therefore the compute and cache cost, grow with conversation length.

**Input.** The current token's query, plus (a) a short window of the most recent `w` tokens' keys/values and (b) a fixed set of `m` "anchors."

**Output.** A `d_model`-dimensional vector, the weighted combination of whatever in the attended set was most relevant to the current token.

**Internal working.**

```
             ANCHOR ATTENTION — bounded-size real attention
             ================================================

   [ Persistent anchors: a_1 .. a_m ]      [ Local window: x_(t-w) .. x_t ]
     (m tokens total, split into                (w tokens, most recent
      static "registers" + dynamic                 only, ALiBi-style
      top-k Persistent Memory Bank                  distance bias)
      readouts — see Section 12)
                    \                                    /
                     \                                  /
                      v                                v
              +----------------------------------------+
              |   Keys / Values: anchors + window        |
              |            ( m + w total, FIXED )         |
              +----------------------------------------+
                                 |
                   query q_t from current token
                                 |
                                 v
                 softmax attention over (m + w) keys
                                 |
                                 v
                        weighted sum of values
                                 |
                                 v
                        output, dimension d_model

   Total attended set size is FIXED at (m + w), independent of how
   long the conversation has run  ->  bounded compute AND bounded
   cache for this layer type, no matter the context length.
```

Two design choices worth spelling out:

- **The anchor set `a_1..a_m` is composed of two parts:** a small number of *static, learned* "register" vectors (always the same, functioning as a stable, always-available attention target — this specifically mitigates a well-documented pathology where attention mechanisms dump disproportionate weight onto arbitrary early tokens as a kind of "no-op" target; giving the model dedicated slots for that behavior is cheaper and more controllable than letting it happen accidentally), and a small number of *dynamically retrieved* readouts from the Persistent Memory Bank (the top few slots most relevant to the current query, fetched by content similarity — Section 12). This means Anchor Attention is simultaneously the network's "read the last paragraph precisely" mechanism and its "read interface" into long-term memory.
- **Positional handling is asymmetric on purpose.** The local window uses a simple distance-based bias added to the attention scores before softmax (an ALiBi-style linear penalty for distance — chosen because with a bounded window, there is no extrapolation problem: the model only ever needs biases for distances `0..w`, and that range is identical at training and inference time). The anchors — both the static registers and the PMB readouts — carry **no** positional bias; they are treated as content-addressable reference points that aren't "at" any particular sequence position, which is the correct semantics for a persistent register or a compressed long-range summary.
- **Grouped-query attention is used internally** (fewer key/value heads than query heads, a standard, well-established efficiency technique) to further reduce this layer type's parameter and compute cost — an orthogonal optimization stacked on top of the bounded-size design, not itself a novel contribution of this document.

**Advantages.** Precise, near-lossless recall over the recent past and over whatever the model has chosen to keep in long-term memory — the specific capability GLT cannot guarantee. Fixed, bounded cost regardless of how long the conversation has run: `w` and `m` are hyperparameters chosen once per tier (Section 17), never a function of conversation length.

**Limitations.** This is still *not* full attention — anything outside the local window that was *not* deemed important enough to be written to the Persistent Memory Bank is genuinely unreachable by this layer type. It is a deliberate accuracy/efficiency tradeoff, not a free approximation of full attention (Section 20 is explicit about this). Real softmax attention, even bounded, costs more per layer than GLT — this is exactly why it is used sparingly (Section 8) rather than as the default.

**Why it improves the overall model.** Anchor Attention is what prevents GLT's lossy compression from being a hard ceiling on model quality. It gives the network a periodic, cheap "double-check" against the raw recent tokens and against anything it decided was worth remembering — the combination is what makes the two-tier memory philosophy (Section 3) actually work in practice rather than just in theory.

---

## 8. Layer Composition & Interleaving

**The general rule.** Anchor Attention appears roughly every second-to-third layer *within* the Reasoning Core (where precise recall during active "thinking" is disproportionately valuable), and roughly every third-to-fourth layer in the Perception and Output stages (where it mainly serves as a periodic precision checkpoint around a majority-GLT stack). GLT remains the majority layer type throughout, consistent with it being the default per Section 1.3.

**Worked example — the "Small" tier** (`d_model = 384`, 8 total physical layers; full tier table in Section 17):

```
   Layer:      1      2      3   |    4       5      |    6      7      8
   Stage:   [------ Perception ------] [--- Reasoning ---] [------ Output ------]
                                        [--- Core (looped) ---]
   Type:      GLT    GLT   Anchor  |   GLT    Anchor      |  GLT    GLT   Anchor

   Physical layers = 8 (3 Perception + 2 Core + 3 Output)
   GLT layers  = 5   Anchor layers = 3   (ratio ~ 1.7 : 1)
   The 2 Core layers are the ONLY ones that execute more than once per
   token (Section 9) — their weights are stored once, used 1..N_max times.
```

**Why interleaving (alternating layer types sequentially down the stack) rather than a parallel dual-path design.** An alternative worth naming explicitly: every layer could run *both* GLT and Anchor Attention in parallel and merge their outputs with a learned gate. This would be more expressive in principle, but it means paying for real softmax attention at *every* layer, every token — directly violating design principle 4 (minimize unnecessary computation) for a benefit that a periodic, cheaper checkpoint captures most of anyway. Interleaving was chosen specifically because it lets each layer be either cheap (GLT) or precise (Anchor), never both-and-therefore-neither-cheap, which keeps the aggregate compute close to "mostly O(1) layers, occasionally a bounded-attention layer" rather than "every layer pays attention's cost."

---

## 9. Adaptive Depth Looping — The Reasoning Core

This is the mechanism most directly responsible for "strong reasoning ability" and "high intelligence for its parameter size" — the two goals hardest to satisfy honestly at ≤100M parameters, since brute-force scale (the usual way large models get "smarter") is simply not available.

**Purpose.** Give the model **variable, token-dependent computational depth at inference time, without variable stored parameters.** Standard wisdom holds that reasoning-heavy tasks benefit from more sequential computation (more layers, more steps of refinement) — but more *stored* layers means more *parameters*, which is the one resource this design cannot spend freely. Adaptive Depth Looping resolves the tension by reusing the *same* weights multiple times per token, so effective computational depth and stored parameter depth become two independent numbers instead of one.

**Input.** The hidden state as it exits the Perception stage (Section 8).

**Output.** A hidden state that has been refined by 1 to `N_max` passes through the Reasoning Core, ready for the Output stage.

**Internal working.**

```
        ADAPTIVE DEPTH LOOPING — variable compute per token
        ======================================================

   from Perception stage
           |
           v
   +-----------------------------------------------+
   |  Reasoning Core:  [GLT] -> [Anchor] -> [GLT]    |   <-----+
   +-----------------------------------------------+          |
           |                                                  |
           v                                                  |
   +--------------------+                                       |
   |   Halting Head      |  (tiny linear + sigmoid,               |
   |   reads pooled       |   d_model -> 1)                        |
   |   hidden state,      |                                        |
   |   outputs p_i         |                                       |
   +--------------------+                                        |
           |                                                    |
   cumulative(p_1 .. p_i)  >=  threshold ?                        |
           |                                                    |
        no |                                                    |
           +----------------------------------------------------+
           |    (loop again, SAME weights, up to N_max times total)
        yes|
           v
    weighted-combine the per-iteration states (ACT-style, weighted
    by each iteration's halting probability) and pass the result
    to the Output stage

   Easy token   ("the", punctuation)          -> typically 1 iteration
   Hard token   (an inference step, a          -> up to N_max iterations,
    disambiguation, an arithmetic step)           more compute exactly
                                                    where it's needed
```

Only **one physical copy** of the Reasoning Core's weights exists in memory. Looping it `N` times means executing the same matrices `N` times, not loading `N` different sets. Using the worked example from Section 8 (Perception = 3 layers, Core = 2 layers, Output = 3 layers, `N_max = 4`): a hard token can receive `3 + (2 × 4) + 3 = 14` layers' worth of *computation*, while the model only ever *stores* `3 + 2 + 3 = 8` layers' worth of *parameters*. This 8-stored / up-to-14-effective ratio is the concrete mechanism behind "high intelligence for its parameter size" — it is not a metaphor, it is a direct accounting identity of the design.

**Training the halting decision.** A hard threshold/argmax stop is not differentiable, so training uses the standard adaptive-computation-time approach: accumulate a weighted combination of the state produced at *every* loop iteration, weighted by that iteration's halting probability (so gradient flows through all iterations during training, even though only a variable number actually "count" at inference), plus an explicit **ponder-cost** term added to the loss that penalizes excessive looping. This balances two failure modes directly: with no ponder cost, the model has no incentive to ever stop early (it will happily spend `N_max` iterations on every token, defeating the point); with too strong a ponder cost from the very start of training, the model may never discover that looping helps at all, and will converge to always halting after one iteration. The practical mitigation is to **anneal the ponder-cost weight in gradually** — near zero for the first portion of training (let the model discover that looping is useful before penalizing it), then ramped up to its target strength (push it toward using looping efficiently once it already knows looping helps). A hard cap `N_max` (chosen per tier, Section 17) bounds worst-case latency unconditionally, regardless of how training turns out — this is a deliberate safety net for principle 7 (mobile UX cannot tolerate an unbounded worst case, even a rare one).

**Advantages.** Reasoning-relevant compute is spent adaptively, per token, rather than uniformly — directly satisfying principle 4. Effective depth scales independently of stored parameter count, which is the single highest-leverage lever available for improving a small model's reasoning without growing its file size. Worst-case latency is bounded by design (`N_max`), so this never turns into unpredictable jank on-device.

**Limitations.** The ponder-cost balance is a genuinely finicky training hyperparameter — get it wrong and the mechanism either never activates or always maxes out; annealing helps but does not eliminate the need for care and monitoring during training (Section 18 gives concrete signals to watch for). This design also makes a specific structural bet: that "the part of computation worth repeating" is well-localized to one designated segment of the network. If that assumption is wrong — if useful iterative refinement actually needs to happen throughout the stack rather than in one middle segment — a more distributed adaptive-compute scheme (Section 21) might do better; this is flagged honestly rather than presented as settled.

**Why it improves the overall model.** This is the component doing the most direct work against the hardest constraint in the whole brief: getting real reasoning ability out of a parameter budget too small for scale to do it alone. Every other component in this document is primarily about efficiency; this one is primarily about capability.

---

## 10. Feed-Forward Mechanism — Gated Shard FFN

**Purpose.** Give the model's feed-forward computation the same "spend compute only where needed" property GLT and ADL already have, without importing the training fragility that comes with classic discrete-routing Mixture-of-Experts (Section 1.2) — directly serving principle 4 and the "easy to train from scratch" goal simultaneously.

**Input.** The hidden state coming out of a layer's mixing block (GLT or Anchor Attention output).

**Output.** A transformed hidden state of the same dimension (`d_model`), ready for the residual connection into the next layer.

**Internal working.**

```
                GATED SHARD FFN — one layer
                ============================

    hidden state h   (dimension d_model)
           |
           v
    +----------------------------------------------+
    |  cheap gate head:  g = sigma(W_gate h)          |
    |  g is a vector in (0,1)^K   (K = shard count)    |
    +----------------------------------------------+
           |
           v
     for each shard k = 1..K:
        shard_k(h)      = SwiGLU_k(h)     (a standard gated FFN
                                            block; each shard holds
                                            a 1/K fraction of the
                                            total intermediate width)
        contribution_k  = g_k * shard_k(h)
           |
           v
     output = down-project( sum_k contribution_k )   -->  dimension d_model

     At inference: any shard whose gate g_k falls below a small
     threshold is SKIPPED ENTIRELY (its matmuls are not executed).
     A sparsity-inducing regularizer during training pushes most
     (token, shard) gate values toward 0 or 1, so this skip is a
     real, usable speedup rather than a theoretical one.
```

Each individual shard is built from a standard SwiGLU-style gated block (gate-projection and up-projection multiplied elementwise, then down-projected) — this specific primitive is adopted as a well-established, efficient building block, not claimed as novel. The novel part is the **soft, continuous per-shard gate with a sparsity regularizer**, used instead of discrete top-k routing: because `g_k` is a smooth sigmoid rather than an argmax, there is no routing collapse to guard against, no load-balancing auxiliary loss to tune, and no discrete decision that can lock in badly early in training and never recover — all genuine risks of classic MoE (Section 1.2) that this design is explicitly trying to avoid.

**Advantages.** Average compute cost tracks the number of *actively gated-open* shards rather than the total shard count `K`, without any of classic MoE's training instability. Degrades gracefully: even a poorly-tuned sparsity regularizer just yields less sparsity (more shards active, more compute, but a model that still trains and works), rather than a routing collapse that produces a broken model.

**Limitations.** Unlike true MoE (which can decouple *total* capacity from active compute, letting total parameters exceed what a dense model of the same active-compute budget could afford), total FFN parameters here are fixed by `K` regardless of how sparse the gates become — the win from sharding is purely a **compute** saving, not a **capacity** increase, and this document does not claim otherwise. Achieving good sparsity also depends on the regularizer's strength being tuned reasonably (Section 18) — an untuned model simply gets less of the compute benefit, not a broken one, but it is still a knob that needs attention.

**Why it improves the overall model.** This applies the "activate expensive computation only when required" principle at the sub-layer level (individual FFN shards), complementing GLT's per-channel gating and ADL's per-token looping — the three together mean the model spends compute conditionally at three different granularities: channel, shard, and token/depth.

---

## 11. Gating Mechanisms — Consolidated

Every gate in the architecture, in one place, since gating is the mechanism principle 4 is implemented *through* at every level of the design:

| Gate | Location | Computed from | Controls | Approx. parameter cost |
|---|---|---|---|---|
| Decay gate `gamma_t` | Inside every GLT layer | `x_t` | How much of the old recurrent state survives this step | One `d_model × d_state` matrix |
| Input gate `iota_t` | Inside every GLT layer, decoupled from `gamma_t` | `x_t` | How much new content gets written into the state | One `d_model × d_state` matrix |
| Output gate `r_t` | Inside every GLT layer | `x_t` | How much of the recalled state content passes onward | One `d_model × d_state` matrix |
| Shard gate `g_k` | Inside every Gated Shard FFN | hidden state `h` | Which FFN shards activate for this token | `K × d_model`, small |
| Halting gate `p_i` | After each Reasoning Core loop iteration | pooled post-loop hidden state | Whether to loop again (up to `N_max`) | `d_model × 1`, negligible |
| PMB write gate | Persistent Memory Bank, once per chunk | chunk summary + existing slot content | How much of a memory slot gets overwritten vs. preserved | Small, one shared matrix |
| PMB addressing weights | Persistent Memory Bank, once per chunk | similarity between chunk summary and every slot | Which slot(s) receive the write | No extra parameters (a similarity computation) |

Two design notes that apply across the whole table: every gate here is a **cheap, small, per-token or per-chunk linear projection followed by a sigmoid or softmax** — none of them is itself an expensive operation. And every gate is genuinely **decoupled from its neighbors** where it matters (decay vs. input in GLT, per-shard in the FFN) rather than sharing a single control signal for multiple jobs, which is the specific, repeated design choice this document makes in favor of expressiveness over parameter-minimalism at these particular points (each costs very little in absolute terms, so the tradeoff is favorable even though it is a tradeoff).

---

## 12. Persistent Memory Bank

**Purpose.** Provide genuine long-term memory — information that should survive far longer than GLT's continuously-decaying working state, and that can, as a bonus specific to this design, survive across app sessions entirely. This is the "long-term" tier of the two-tier memory philosophy (Section 3).

**Input.** A periodic "chunk summary" — a pooled representation of the last `K` tokens' worth of GLT states and Anchor outputs (`K` a tunable chunk size, e.g. every 64–128 tokens, not every token — writing on every token would be both unaffordable and counterproductive, since most individual tokens aren't worth committing to permanent storage).

**Output.** A fixed-size set of `S` slot vectors (`S` a small constant per tier, Section 17), read by every Anchor Attention layer as part of its anchor set (Section 7), via a top-k content-similarity lookup against the current query.

**Internal working.**

```
              PERSISTENT MEMORY BANK — write path (every K tokens)
              ======================================================

     tokens t-K+1 .. t   (one "chunk")
              |
              v
      pool the chunk's GLT states / Anchor outputs
              |
              v
       chunk summary vector  c   (dimension d_mem)
              |
              v
      similarity( c, slot_1 ), similarity( c, slot_2 ), ... , similarity( c, slot_S )
              |
              v
        softmax over slots  -->  addressing weights  w_1 .. w_S
              |
              v
     for the most strongly addressed slot(s):
        update_gate = sigma( W_u [ c ; slot_i ] )
        slot_i_new  = (1 - update_gate) * slot_i_old  +  update_gate * c
              |
              v
     Persistent Memory Bank   (S slots, FIXED count, never grows)
              |
              v
     read at every Anchor Attention layer, as part of its anchor
     set (Section 7), via a top-k content-similarity lookup
```

There is a **single** Persistent Memory Bank shared across the entire network (not one per layer), which keeps its footprint small and constant regardless of network depth. The addressing mechanism (similarity-based softmax over slots, followed by a gated update) is a simplified, direct descendant of content-addressable memory ideas from the Neural Turing Machine / Differentiable Neural Computer lineage — adopted here in a deliberately minimal form (no separate read/write heads, no learned addressing controller beyond the similarity computation itself) appropriate to a small-model budget.

**A concrete, mobile-specific advantage:** because the bank is a small, fixed-size tensor (e.g. 64 slots × 128 dimensions at fp16 is 16 kilobytes), it can be serialized to disk trivially. This means a conversation's *salient long-term content* can survive the app being killed and relaunched by Android — not just the immediate working state (Section 13 covers this operationally).

**Advantages.** Bounded size regardless of conversation length. Captures information that mattered early in a long conversation, which GLT's recency-biased decaying state would have long since diluted. Cheap enough to persist across app sessions, giving a lightweight, fully on-device analogue of durable context — without any server-side storage.

**Limitations.** External memory modules have a well-documented failure mode: the network can learn to simply ignore them if there's no training pressure forcing genuine use, since attending to the (initially useless) memory bank offers no early benefit for the loss to reward. This is addressed directly in the training strategy (Section 18) via an auxiliary self-supervised objective, not left to hope. Even when used well, the bank's fixed slot count is a real capacity ceiling — very long conversations with many genuinely distinct salient facts will eventually force older slots to be overwritten in favor of newer ones.

**Why it improves the overall model.** This is what turns "efficient long-context handling" from "doesn't fall over" (GLT's job) into "actually retains what mattered" — and it is the component that makes cross-session continuity possible on-device without any additional infrastructure, which is a genuinely mobile-specific capability most architectures never have reason to consider.

---

## 13. Context Management

**No architectural context limit.** Because GLT's state is a fixed-size matrix rather than a growing cache, there is no hard maximum sequence length the way a Transformer has a fixed context window — the model can, in principle, keep processing tokens indefinitely without running out of memory. This needs an immediate, honest caveat, given in full below, but the *architectural* claim (no code path that runs out of RAM purely because a conversation got long) holds unconditionally.

**The honest caveat: unbounded length is not the same as unbounded fidelity.** GLT's state has a bounded information-theoretic capacity (roughly tied to `d_state²`, shared across everything seen so far) — "unbounded context" describes memory *usage*, not recall *quality*. In practice, expect three different fidelity zones layered on top of each other: strong, near-exact fidelity over the most recent tokens via the Anchor Attention window; reliable retention of whatever was salient enough to be written to the Persistent Memory Bank, effectively indefinitely; and gradually softening general context beyond that, mediated entirely by GLT's decaying state. This is presented as a real property of the design, not a limitation hidden behind the phrase "unbounded context."

**State checkpointing — a mobile-specific capability this architecture gets almost for free.** The complete recurrent state of a conversation is small and *constant-sized*: every GLT layer's `d_state × d_state` matrix, plus the Persistent Memory Bank's `S` slots. For the Small tier worked example (5 GLT layers at `d_state = 96`, plus a 32-slot PMB at `d_mem ≈ 96`, all at fp16): roughly `5 × 96 × 96 × 2 bytes ≈ 88 KB` for the GLT states, plus `32 × 96 × 2 bytes ≈ 6 KB` for the PMB — on the order of **100 KB total**, regardless of whether the conversation is 50 tokens or 50,000 tokens long. Android routinely backgrounds and kills apps to reclaim memory; a Transformer-based app would either have to reprocess the entire conversation history on resume (slow, and getting slower the longer the chat has run) or serialize a KV-cache that itself grows with conversation length. Here, the checkpoint is serialize-on-pause, reload-on-resume, and generation continues from the exact token it left off at — with **zero recomputation**, and a checkpoint size that never grows no matter how long the conversation gets.

**Chunk boundaries as natural checkpoints.** The same `K`-token chunking used for Persistent Memory Bank writes (Section 12) is a convenient, already-existing hook for other context-management behaviors an application layer might want — for example, triggering an explicit summarization pass, or flagging a natural point to prune very old, low-salience content from being considered for future writes — without requiring any new architectural mechanism beyond what chunking already provides.

---

## 14. Memory Compression

Three genuinely distinct kinds of compression happen in this architecture, and they are worth clearly separating because they compress different things, for different reasons, at different times:

| Level | What is compressed | Mechanism | When | Purpose |
|---|---|---|---|---|
| **Working-state compression** | The entire conversation's recent history | GLT's bounded `d_state × d_state` recurrent matrix (Section 6) | Continuously, every token | Keep runtime conversation memory O(1) regardless of length |
| **Long-term / salience compression** | Whatever the model judges worth keeping | Persistent Memory Bank's chunk-summary + gated slot writes (Section 12) | Once per `K`-token chunk | Retain important content far longer than the working state would, in a still-bounded footprint |
| **Weight compression** | The model's learned parameters (not conversation state at all) | Post-training or quantization-aware INT4/INT8 quantization (Section 19) | Once, at deployment/export time | Shrink the static model file and, on a memory-bandwidth-bound CPU, directly speed up every inference step |

The first two are about **runtime conversation memory** and are architectural; the third is about **static deployment size** and is a training/export-time decision layered on top. All three matter for a mobile deployment, but conflating them (a common imprecision in casual architecture discussions) obscures that they solve different problems and are tuned independently.

**An optional, defensive addition worth naming even though it isn't part of the core spec:** for extremely long-running conversations, periodically passing the GLT state through a small, dedicated "state refresh" sub-network (essentially a tiny auto-encoder trained to re-express the current state in a cleaner, less drift-prone form) can guard against slow numerical drift accumulating over tens of thousands of steps. This is not required for the architecture to function, and adds a small amount of extra compute at chunk boundaries only — it is flagged here as a robustness option to consider empirically, not a committed component.

---

## 15. Output Layer

**Purpose.** Convert the final hidden state (after the Output stage, Section 8) into a probability distribution over the vocabulary, from which the next token is sampled or chosen.

**Input.** The final hidden state, dimension `d_model`.

**Output.** A probability distribution over `vocab_size` tokens.

**Internal working.** The hidden state is down-projected through `W_out` (`d_model -> d_embed`, the same factorized dimension used by the input embedding, Section 5), then multiplied against the **transpose of the same tied embedding table** `E` (shape `[vocab_size, d_embed]`) to produce logits, followed by a standard softmax.

Because the vocabulary is deliberately small (8,000–16,000 tokens, Section 4), this final softmax is computationally trivial — a few thousand logits is a negligible cost on any mobile CPU, unlike the 100,000+ vocabularies common in larger models. This is an explicit **non-choice worth justifying**: large-vocabulary models often need hierarchical or adaptive softmax variants to keep this step affordable; Continuum deliberately avoids that entire category of complexity by keeping the vocabulary small in the first place (Section 4), rather than solving a problem it chose not to have.

**Advantages.** Tying halves the parameter cost relative to a separate output projection (already the largest single saving in the embedding budget, Section 5); the small vocabulary keeps this step's compute negligible without any specialized softmax machinery.

**Limitations.** Weight tying means the input ("reading") and output ("writing") representations for every token necessarily share the same underlying geometry — a mild constraint, though one with a long, well-tested track record across many architectures at this scale, and one this design accepts deliberately given the parameter savings involved.

**Why it improves the overall model.** This is the last of several places (alongside the embedding table, Section 5) where a well-established, low-risk efficiency technique is applied in full, because at this parameter scale, saving a few million parameters on "plumbing" like embeddings and output projection is worth more than spending them on marginal plumbing sophistication — every saved parameter here is a parameter available for the layers that actually reason.

---

## 16. End-to-End Data Flow

Section 2 gave the macro, stage-level picture. This section traces one concrete token through the entire system, step by step, to make the interaction between every module explicit.

```
   ONE TOKEN'S JOURNEY, START TO FINISH
   =======================================

   [1] Raw text  --tokenizer (byte-level BPE, single-digit numbers)-->  token ID

   [2] token ID  --factorized embedding lookup + up-projection-->  x  (dim d_model)

   [3] PERCEPTION:  x flows through GLT -> GLT -> Anchor Attention
                    (each GLT layer updates its own S_t; the Anchor
                     layer reads the local window + static registers
                     + top-k Persistent Memory Bank readouts)

   [4] REASONING CORE:  x flows through GLT -> Anchor -> GLT (one pass)
                        -> Halting Head checks confidence
                        -> loop again with the SAME weights if not
                           confident yet (up to N_max times)

   [5] OUTPUT STAGE:  x flows through GLT -> GLT -> GLT -> Anchor Attention

   [6] Output projection:  x --down-project--> matmul with tied embedding
                           table transpose --> softmax --> distribution
                           over the vocabulary

   [7] A token is sampled/chosen from that distribution -- this becomes
       both the visible output AND, in generation mode, the next [1]-[6]
       input, with every GLT state carried forward unchanged

   [8] (separately, every K tokens): the chunk's states are pooled into
       a summary and written into the Persistent Memory Bank via its
       gated, content-addressed update (Section 12) -- independent of
       the per-token flow above

   [9] (on app background/kill): the current GLT states + PMB slots
       (a small, FIXED-size object, Section 13) are serialized to disk;
       on relaunch, they are reloaded and generation resumes from
       exactly this point, with no reprocessing of anything before it
```

Every arrow in this trace costs either O(1) (GLT layers, the FFN, the output projection) or a small, fixed O(w+m) (Anchor Attention layers) — nothing in this entire path grows with how long the conversation has already run, except the loop count in step [4], which is itself capped at `N_max` and decided per-token by the halting head, never by conversation length.

---

## 17. Parameter Scaling — 5M to 100M

**The estimation method, stated up front.** Every number in this section comes from the following back-of-envelope formulas, applied consistently across tiers. These are approximate — real counts depend on implementation details (bias terms, norm parameters, exact GQA head-reduction ratio) not fully pinned down at the design-document stage — and are presented as a starting point for empirical tuning, not a guaranteed exact count.

```
   embedding_params   ~=  vocab x d_embed  +  2 x (d_embed x d_model)     [tied, factorized]
   GLT_layer_params   ~=  7 x d_model x d_state                          [k,v,q,gamma,iota,r,o]
   anchor_layer_params ~= 4 x d_model^2                                  [q,k,v,o -- before GQA reduction]
   ffn_layer_params   ~=  3 x r x d_model^2                              [SwiGLU-style, r = expansion ratio;
                                                                          shard count changes ACTIVE compute,
                                                                          not this total]
   PMB_mechanism      ~=  small, roughly constant (~4 x d_model x d_mem),
                          independent of slot count S (S affects STATE
                          size, not stored PARAMETER count)
   halting_head       ~=  negligible (d_model x 1)

   total  ~=  embedding_params
            + (n_GLT_layers  x GLT_layer_params)
            + (n_anchor_layers x anchor_layer_params)
            + (n_layers x ffn_layer_params)         [every layer gets one FFN]
            + PMB_mechanism + halting_head
```

**The four tiers.**

| Tier | `d_model` | `n_layers` | GLT : Anchor split | `d_state` | vocab | `d_embed` | FFN expansion `r` | Shards `K` | PMB slots `S` | Anchor window `w` / anchors `m` | ADL `N_max` | **~Total params** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Continuum-Nano** | 192 | 6 | 4 : 2 | 48 | 8,000 | 48 | 3x | 2 | 16 | 48 / 8 | 3 | **~5M** |
| **Continuum-Small** | 384 | 8 | 5 : 3 | 96 | 12,000 | 80 | 4x | 4 | 32 | 96 / 12 | 4 | **~20M** |
| **Continuum-Base** | 576 | 10 | 7 : 3 | 144 | 16,000 | 128 | 4x | 4 | 48 | 128 / 16 | 4 | **~50M** |
| **Continuum-Max** | 768 | 12 | 9 : 3 | 192 | 16,000 | 160 | 4x | 6 | 64 | 128 / 24 | 5 | **~100M** |

**What scales, what doesn't, and why.**

- **`d_model`, `d_state`, `d_embed` scale together** — this is the primary "width" lever, and it's where most of the capacity comes from (the FFN term grows quadratically in `d_model`, which is why it dominates total parameters at every tier, matching the well-established pattern in dense transformer-family models generally).
- **`n_layers` grows only modestly (6 → 12)**, deliberately. Because Adaptive Depth Looping already supplies "virtual depth" independent of stored layers (Section 9), this design doesn't need aggressive physical-depth scaling the way a pure feed-forward stack would — a large part of the "more capable at larger sizes" story is width and looping budget, not raw layer count.
- **Vocabulary grows mildly then plateaus (8k → 12k → 16k → 16k)** — a bigger model can afford a somewhat richer vocabulary, but this was never the primary capacity lever (Section 4), so it isn't pushed further once it stops being the bottleneck.
- **Anchor window `w` plateaus around 128** rather than growing with model size — a larger model doesn't need a dramatically larger local window; the Persistent Memory Bank and the GLT state are the mechanisms responsible for handling scale in the *length* dimension, not the window.
- **PMB slot count and ADL max-loop count grow only slightly** — bigger tiers can afford to remember a few more distinct things and spend a couple more reasoning iterations, but neither is scaled aggressively, since both are already doing their job at the smaller sizes; over-scaling them would spend parameter/compute budget with diminishing returns relative to spending it on width.

---

## 18. Training Strategy

**Parallelizable despite being recurrent.** As established in Section 6, GLT's linear recurrence admits a parallel (associative) scan, so training processes whole sequences in parallel chunks rather than one slow sequential step at a time — this is what makes "easy training from scratch" a realistic claim rather than a hope; a naive sequential unroll would be far too slow to be practical on the kind of hardware a solo developer actually has access to.

**Initialization.** The GLT decay gate's bias is initialized with a deliberate *spread* across channels — some channels start biased toward slow decay (long memory), others toward fast decay (short memory) — a simplified, practical borrowing of the general insight behind HiPPO-style initialization (different channels should specialize to different timescales from the outset) without adopting the fuller continuous-time HiPPO formalism. All gates are sigmoid-bounded by construction (Section 6), which removes an entire category of the exploding/vanishing-state failure modes that less-constrained recurrent parametrizations have to guard against separately.

**Sequence-length curriculum.** Training begins on short sequences and progressively increases sequence length. This reduces early-training compute cost and, empirically in related recurrent-model literature, tends to stabilize the learned dynamics by not exposing the model to very long dependencies before it has learned basic local patterns.

**Loss composition.** The total training loss combines:
1. Standard next-token cross-entropy (the primary signal).
2. The ADL ponder-cost term (Section 9), **annealed in gradually** — near zero early in training so the model can first discover that looping helps, then ramped up to its target strength so the model learns to loop *efficiently* rather than never or always.
3. The Gated Shard FFN sparsity regularizer (Section 10), encouraging shard gates toward 0 or 1 rather than staying uniformly diffuse, so the inference-time skip becomes real rather than nominal.
4. An auxiliary **"memory-only prediction"** objective for the Persistent Memory Bank: at a sampled fraction of training positions, the GLT state and Anchor local-window contributions are masked out, and the model is asked to predict the target token using **only** the Persistent Memory Bank readouts. This directly counters the well-documented tendency of external-memory modules to be learned but never actually relied upon — if the bank isn't storing anything useful, this auxiliary loss stays high and provides a genuine gradient pushing the write mechanism toward storing genuinely predictive content, rather than leaving PMB utility to chance.

**Data strategy.** Small models benefit disproportionately from dense, high-quality training data relative to raw web-scale scrape — a well-established finding in the small-model literature generally (in the spirit of the "quality over volume" philosophy behind several recent small-model efforts, referenced here as general precedent rather than a specific method being copied). Concretely: favor curated conversational text, code, and explicit reasoning/word-problem data over noisy unfiltered scrape, especially at the Nano and Small tiers where every parameter has to work harder. Knowledge distillation from a larger teacher model (matching soft output distributions rather than only hard labels), if a suitable teacher is available, is a strong optional lever for boosting small-model quality further — the architecture works with plain from-scratch training too, but distillation is worth using when feasible.

**Regularization specific to this architecture's failure modes.** Standard dropout and weight decay apply throughout. In addition, a modest **state dropout** — randomly zeroing or perturbing the GLT recurrent state during training — is a targeted defense against a failure mode specific to gated-recurrent architectures: the model discovering it can "shortcut" by relying mostly on very recent tokens (which are cheap and always available) rather than genuinely using the longer-range compressed state. Perturbing the state during training removes that shortcut's reliability, pushing the model to actually use its memory system rather than a local-recency heuristic that happens to work most of the time.

**Stability monitoring.** Standard gradient clipping applies. In addition, monitor the **distribution of decay-gate (`gamma_t`) values** across channels during training — if this distribution collapses toward all-0 (the state constantly resets, no long-range memory forms) or all-1 (the state never updates, new information is never admitted), that is a concrete, actionable early signal of a training pathology worth investigating before continuing a long run.

**A practical implementation roadmap — introduce complexity in stages.** This architecture has five interacting, partly-novel components (GLT, Anchor Attention, Adaptive Depth Looping, Gated Shard FFN, Persistent Memory Bank). Implementing and training all five simultaneously invites a tangled debugging process where a failure could originate almost anywhere. A more tractable path for a solo from-scratch implementation:

1. **Stage 1:** GLT + Gated Shard FFN + tied factorized embedding only (no Anchor Attention, no ADL, no PMB). Verify this alone trains to sane loss curves and produces coherent short-range text.
2. **Stage 2:** Add Anchor Attention (fixed single-pass depth, no looping yet). Verify recall-sensitive behavior improves over Stage 1.
3. **Stage 3:** Add Adaptive Depth Looping to the designated Reasoning Core segment. Verify the halting-gate distribution is sane (not collapsed to always-1 or always-`N_max`) before trusting the mechanism.
4. **Stage 4:** Add the Persistent Memory Bank and its auxiliary loss. Verify (via the memory-only-prediction auxiliary loss trending down over training) that the bank is actually being used, not merely present.

Each stage should be independently validated before the next is added — this is the practical answer to the complexity this design honestly carries (Section 20 revisits this directly).

---

## 19. Inference Strategy

**Weight-only quantization, and why it works especially well here.** Per Section 1.1, batch-size-1 CPU inference is memory-bandwidth-bound: the bottleneck is bytes moved from RAM, not raw arithmetic. Quantizing weights to INT4 or INT8 (group-wise, either post-training or quantization-aware) directly reduces bytes moved per token, which speeds up inference roughly in proportion to the size reduction — even without specialized low-precision compute kernels. Activations are kept at higher precision (fp16), since activation quantization is comparatively fragile and most of the RAM/speed benefit already comes from shrinking the *weights*. This mirrors common practice in mobile LLM deployment (e.g. the GGUF/llama.cpp ecosystem's weight-quantization approach), adopted here as a systems technique, not claimed as an architectural novelty.

**Cache-friendly by construction.** The GLT recurrent state (`d_state × d_state`, e.g. 96×96 = 9,216 floats at the Small tier) is small enough to stay resident in L1/L2 cache for the duration of a layer's per-token update — a concrete, favorable memory-access pattern, in direct contrast to a Transformer's KV-cache, which for long contexts eventually exceeds cache size and spills into slower RAM, with per-token latency creeping upward as the conversation grows. Continuum's per-token cost, and cache behavior, stays flat regardless of how long the conversation has run.

**No growing cache anywhere in the model.** GLT layers carry O(1) state. Anchor Attention layers carry a small, hard-bounded cache of exactly `w + m` tokens' worth of keys/values (Section 7) — this never grows, no matter how long the conversation runs; once the window slides forward, the oldest local token is simply dropped (its relevant content, if it was salient, has already had the opportunity to be written to the Persistent Memory Bank).

**Streaming, single-token decode.** Generation processes one token at a time, updating every layer's state in place — the natural execution mode for an interactive chat experience, with no batching or re-processing of history required at any point.

**Dual-mode execution: parallel prefill, sequential decode.** The initial prompt (a user's message, a system prompt) does not need to be processed one token at a time — the same parallel-scan formulation used for training (Section 6, Section 18) applies equally at inference time for the *prefill* phase, letting an entire prompt be processed as a single parallel pass rather than a sequential unroll. Only the token-by-token *generation* phase runs the true sequential recurrence. This two-mode execution (parallel prefill / sequential decode) mirrors how Mamba- and RWKV-family models are deployed in practice today, and is essential for keeping time-to-first-token low.

**Adaptive compute means data-dependent speed.** Because Adaptive Depth Looping (Section 9) varies iteration count per token, actual decode speed is workload-dependent: routine conversational turns run close to the `N=1` cost, while a turn requiring more inference-heavy computation runs closer to the `N_max` cost — bounded, and known in advance, but genuinely variable. This should be surfaced honestly (e.g. as an expected-latency range rather than a single fixed tokens/second figure) rather than quoted as one static number.

**Threading.** Android SoCs are typically big.LITTLE: latency-sensitive single-token decode benefits from being pinned to the "big" cores, while the parallel-scan prefill phase (Section 6) and Anchor Attention's multi-head computation can make good use of multiple threads across the full core set. Batch size is assumed to be 1 throughout (a single user, a single stream) — this simplifies several tradeoffs relative to datacenter-style serving, since large-batch-friendliness was never a design requirement here.

**Operational note on state persistence.** As covered in Section 13, the entire runtime state (GLT matrices + PMB slots) is small and fixed-size, making serialize-on-pause / reload-on-resume a cheap, constant-cost operation regardless of conversation length — worth implementing as a first-class part of the inference runtime, not an afterthought, given how directly it serves the "smooth conversation" and "mobile-first" goals.

---

## 20. Self-Critique — Known Limitations

Presented honestly, as requested, rather than glossed over:

1. **Long-range recall is genuinely lossy, not just "efficiently approximated."** GLT's fixed-size state cannot guarantee exact recall of something far outside the Anchor window unless it was salient enough to be written to the Persistent Memory Bank. Tasks requiring precise verbatim retrieval of arbitrary, non-salient information from deep in a long context (e.g. "repeat back the exact 500th word") remain a real weak point, only partially mitigated, not solved.
2. **The Adaptive Depth Looping training balance is genuinely finicky.** Get the ponder-cost weighting or its annealing schedule wrong, and the mechanism can collapse to always-minimum-depth (never engaging the reasoning benefit) or always-maximum-depth (defeating the efficiency point). Section 18's annealing and monitoring guidance mitigates this but does not eliminate the need for active attention during training.
3. **The Persistent Memory Bank's usefulness depends on a training signal that has to be gotten right.** The memory-only-prediction auxiliary loss (Section 18) is a genuine mitigation for the known "unused external memory" failure mode, but it is a mitigation, not a proof — a poorly-weighted auxiliary loss could still leave the bank underutilized in practice.
4. **The small vocabulary is a real, quantified tradeoff, not a free efficiency win.** Expect roughly 20–40% more tokens to encode the same English text compared to a 32k+-vocabulary tokenizer, which partially offsets the per-token compute savings with more decode steps — the net effect is very likely still favorable given how much cheaper each individual step is, but this is a genuine cost, not a rounding error.
5. **GLT's state has a hard information-theoretic ceiling.** No amount of architectural cleverness changes the fact that a bounded matrix has bounded capacity. Extremely long, uniformly high-information-density conversations (not just long in token count, but dense in genuinely novel content throughout) will eventually see general (non-PMB-salient, non-recent) context quality degrade as the state saturates.
6. **The three-stage macro-structure is a specific, falsifiable bet.** Localizing "reasoning-relevant" adaptive computation to one designated middle segment assumes that useful iterative refinement is well-localized rather than needing to happen throughout the network. If that assumption turns out to be wrong empirically, a more distributed adaptive-compute scheme (looping applied more broadly, or at multiple points) might outperform this specific structural choice — this document commits to one hypothesis and states it as such, not as settled fact.
7. **This is a more complex system than a plain small Transformer or a plain small Mamba**, by design — five interacting, partly-original components (GLT, Anchor Attention, ADL, Gated Shard FFN, PMB) means more hyperparameters, more potential failure points, and a genuinely steeper implementation and debugging burden than a single-mechanism architecture. The staged implementation roadmap (Section 18) is the direct, practical answer to this tension, but it does not make the tension disappear — "easy to train from scratch" and "five cooperating novel-ish mechanisms" are in real tension with each other, and that tension is acknowledged here rather than argued away.

---

## 21. Alternative Designs Considered and Rejected

**Whole existing architectures, rejected wholesale** (full analysis in Section 1.2): a standard full-self-attention Transformer (rejected — unbounded KV-cache growth is directly hostile to "smooth conversation" and "low RAM usage"); a direct Mamba- or RWKV-style clone with no hybridization (rejected both because the prompt explicitly calls for genuine originality, and because pure linear-recurrent designs are documented to underperform hybrids specifically on associative recall); a heavy discrete-routing Mixture-of-Experts backbone (rejected primarily for from-scratch training fragility, and because at a fixed ≤100M-parameter ceiling, MoE's signature RAM-for-compute tradeoff doesn't apply — all experts stay resident regardless of routing).

**Internal alternatives, considered within the chosen hybrid design space:**

- **Vector-valued state (a single decaying vector per channel, like a simple gated RNN) instead of GLT's matrix-valued associative state.** Cheaper (`O(d_state)` instead of `O(d_state²)`), but structurally unable to perform the associative, content-based recall a key-value outer-product state supports — more like a running exponential average than a queryable memory. Rejected because, at the small `d_state` values actually used here (48–192), the quadratic cost is still tiny in absolute terms, making the extra recall capability essentially free to acquire.
- **A single coupled decay/input gate (a Δ-style continuous-time discretization) instead of GLT's two independent gates.** More elegant, with a clean continuous-time interpretation, and marginally cheaper. Rejected in favor of the decoupled version specifically because independent control over "how much to forget" versus "how much to accept" can represent gate combinations a single coupled parameter cannot — accepted as a genuine tradeoff (Section 6), not a strict win, since the coupled alternative's elegance and slightly lower parameter count are real, legitimate advantages of the road not taken.
- **Discrete top-k expert routing (classic MoE-style)** instead of the Gated Shard FFN's soft, continuous per-shard gating. Rejected because discrete routing brings load-balancing losses and a real risk of routing collapse — exactly the from-scratch training fragility this design is trying to avoid (Section 1.2, Section 10) — in exchange for a capability (true capacity/compute decoupling) that doesn't apply at a fixed total-parameter ceiling anyway.
- **A parallel dual-path layer (every layer runs both GLT and Anchor Attention, merged by a learned gate)** instead of interleaving layer types sequentially down the stack. Rejected because it means paying real attention's cost at every single layer, which directly conflicts with "minimize unnecessary computation" for a benefit a periodic, cheaper checkpoint (the interleaved design, Section 8) captures most of anyway.
- **Recency-based (FIFO) slot replacement** instead of content-based addressing for the Persistent Memory Bank. Simpler to implement and train, but would lose precisely the property that justifies having a separate long-term tier at all — retaining what's *salient* regardless of *recency*, as distinct from GLT's already recency-biased working state. Rejected as the primary mechanism, but noted as a reasonable fallback if content-based addressing proves difficult to train reliably in practice.

---

## 22. Goal Traceability & Closing Summary

Every goal stated at the outset, traced to the specific component(s) responsible for it:

| Stated goal | Primary component(s) responsible | Section |
|---|---|---|
| Extremely fast CPU inference | GLT's O(1)-per-step cost, bounded-size Anchor Attention, weight quantization | 6, 7, 19 |
| Low RAM usage | Factorized/tied embedding, small vocabulary, fixed-size state (no growing cache), weight quantization | 5, 6, 19 |
| Mobile-first design | State checkpointing across app lifecycle, bounded worst-case latency (ADL cap), cache-resident small matmuls | 13, 9, 19 |
| High intelligence for its parameter size | Weight-shared Adaptive Depth Looping (effective depth exceeds stored depth) | 9 |
| Strong reasoning ability | Adaptive Depth Looping's variable test-time compute | 9 |
| Efficient long-context handling | GLT's O(1) state + Persistent Memory Bank + Anchor Attention's bounded window, working together (Section 3) | 3, 6, 7, 12 |
| Smooth conversation | No cache growth (flat per-token cost over time), streaming decode, bounded worst-case latency | 6, 9, 19 |
| Efficient memory usage | Two-tier memory system (working state + long-term bank); weight quantization for the static model | 3, 6, 12, 19 |
| Easy training from scratch | Parallel-scan formulation, bounded/sigmoid gates, staged implementation roadmap | 6, 18 |
| Scalable 5M to 100M | A consistent set of knobs (`d_model`, `n_layers`, `d_state`, shard count, PMB slots) that scale together predictably | 17 |

**Closing summary.** Continuum's central bet is that a small on-device model should not try to be a shrunk-down version of a datacenter architecture — it should be designed around the specific thing mobile CPUs are actually bad at (moving bytes through memory quickly, especially in a growing pattern) and the specific thing small parameter budgets are actually bad at (reasoning depth). Gated Linear Trace answers the first problem directly, by construction, not by approximation. Adaptive Depth Looping answers the second by decoupling *effective* depth from *stored* depth, which is the only lever available once scale itself is off the table. Anchor Attention and the Persistent Memory Bank exist because neither of those two solutions is free — recurrence compresses lossily, and looping helps depth but not recall — so the design pays, deliberately and in small bounded doses, for the two capabilities recurrence alone cannot provide. Nothing here is claimed to be a solved problem; Sections 20 and 21 name the real tradeoffs and the roads not taken as plainly as the design itself is presented.
