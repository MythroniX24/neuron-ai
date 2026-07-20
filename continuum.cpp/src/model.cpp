/*
 * model.cpp — Layer forward passes for Continuum SLM C++ engine.
 *
 * Ports the custom architecture from Python to C++:
 * - GLT (Gated Linear Trace): matrix recurrence with decoupled gates
 * - Anchor Attention: bounded softmax over window + static anchors + PMB
 * - GatedShardFFN: soft-gated per-shard SwiGLU
 * - HaltingHead, PMB, etc.
 */

#include "model.h"
#include <cmath>
#include <cstring>

namespace continuum {

// ============================================================================
// RMSNorm forward (reusable helper)
// ============================================================================
void rms_norm_forward(Tensor& output, const Tensor& x, const Tensor& scale, float eps) {
    tensor_rms_norm(output, x, scale, eps);
}

// ============================================================================
// FactorizedEmbedding: token_id → embedding [d_model]
// ============================================================================
void embed_forward(Tensor& output, int32_t token_id, const EmbedWeights& w, Arena& arena) {
    int32_t d_embed = w.up_proj.shape.ne[0];    // d_embed (ne[0]=cols, ne[1]=rows → d_embed, d_model)
    int32_t d_model = w.up_proj.shape.ne[1];     // d_model

    // ⚡ Phase 9: FP16 wiring — dequantize into arena if use_fp16
    const float* emb_ptr = fp16_wire(w.embed_table.data, w.embed_table_fp16, w.use_fp16, arena, (size_t)w.embed_table.shape.ne[0] * d_embed);
    const float* up_ptr  = fp16_wire(w.up_proj.data, w.up_proj_fp16, w.use_fp16, arena, (size_t)d_model * d_embed);

    for (int32_t i = 0; i < d_model; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < d_embed; j++) {
            float emb = emb_ptr[token_id * d_embed + j];
            float up = up_ptr[i * d_embed + j];
            sum += up * emb;
        }
        output.data[i] = sum;
    }
}

// ============================================================================
// Output projection: hidden[d_model] → logits[vocab_size]
// ============================================================================
void output_projection(Tensor& logits, const Tensor& hidden, const EmbedWeights& w, Arena& arena) {
    int32_t d_model = w.down_proj.shape.ne[1];
    int32_t d_embed = w.down_proj.shape.ne[0];
    int32_t vocab_size = w.embed_table.shape.ne[0];

    // ⚡ Phase 9: FP16 wiring
    const float* down_ptr = fp16_wire(w.down_proj.data, w.down_proj_fp16, w.use_fp16, arena, (size_t)d_embed * d_model);
    const float* emb_ptr  = fp16_wire(w.embed_table.data, w.embed_table_fp16, w.use_fp16, arena, (size_t)vocab_size * d_embed);

    auto interm = arena.alloc_tensor(TensorShape(d_embed));
    for (int32_t e = 0; e < d_embed; e++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < d_model; j++) {
            sum += down_ptr[e * d_model + j] * hidden.data[j];
        }
        interm.data[e] = sum;
    }

    for (int32_t v = 0; v < vocab_size; v++) {
        float sum = 0.0f;
        for (int32_t e = 0; e < d_embed; e++) {
            sum += emb_ptr[v * d_embed + e] * interm.data[e];
        }
        logits.data[v] = sum;
    }
}

// ============================================================================
// GLT forward: x[d_model] + state[d_state,d_state] → output[d_model] + new_state
//
// Algorithm:
//   1. Pre-norm: xn = RMSNorm(x)
//   2. Project: k=W_k(xn), v=W_v(xn), q=W_q(xn)  [all d_state]
//   3. KV norm: k = RMSNorm(k), v = RMSNorm(v)
//   4. Gates: gamma=sigmoid(W_gamma*xn+b_gamma), iota=sigmoid(W_iota*xn+b_iota),
//             r=sigmoid(W_r*xn+b_r)  [all d_state]
//   5. Outer: outer[i,j] = k[i] * v[j]  [d_state, d_state]
//   6. Update: S_new[i,j] = gamma[i]*S_old[i,j] + iota[i]*outer[i,j]
//   7. Read: h = S_new * q  [d_state]
//   8. Output: o = x + W_o * (r * h)  [d_model]
// ============================================================================
void glt_forward(Tensor& output, Tensor& new_state,
                 const Tensor& x, const Tensor& state,
                 const GLTWeights& w, Arena& arena) {
    int32_t d_model = w.W_k.shape.ne[1];   // W_k: [d_state, d_model]
    int32_t d_state = w.W_k.shape.ne[0];   // d_state

    // ⚡ Phase 9: FP16 wiring — dequantize all weight matrices once into arena
    const float* Wk = fp16_wire(w.W_k.data, w.W_k_fp16, w.use_fp16, arena, (size_t)d_state * d_model);
    const float* Wv = fp16_wire(w.W_v.data, w.W_v_fp16, w.use_fp16, arena, (size_t)d_state * d_model);
    const float* Wq = fp16_wire(w.W_q.data, w.W_q_fp16, w.use_fp16, arena, (size_t)d_state * d_model);
    const float* Wg = fp16_wire(w.W_gamma.data, w.W_gamma_fp16, w.use_fp16, arena, (size_t)d_state * d_model);
    const float* Wi = fp16_wire(w.W_iota.data, w.W_iota_fp16, w.use_fp16, arena, (size_t)d_state * d_model);
    const float* Wr = fp16_wire(w.W_r.data, w.W_r_fp16, w.use_fp16, arena, (size_t)d_state * d_model);
    const float* Wo = fp16_wire(w.W_o.data, w.W_o_fp16, w.use_fp16, arena, (size_t)d_model * d_state);

    auto xn  = arena.alloc_tensor(TensorShape(d_model));
    auto k   = arena.alloc_tensor(TensorShape(d_state));
    auto v   = arena.alloc_tensor(TensorShape(d_state));
    auto q   = arena.alloc_tensor(TensorShape(d_state));
    auto gamma = arena.alloc_tensor(TensorShape(d_state));
    auto iota  = arena.alloc_tensor(TensorShape(d_state));
    auto r     = arena.alloc_tensor(TensorShape(d_state));
    auto h     = arena.alloc_tensor(TensorShape(d_state));
    auto rh    = arena.alloc_tensor(TensorShape(d_state));

    // 1. Pre-norm
    rms_norm_forward(xn, x, w.norm_scale, 1e-6f);

    // 2. Project k, v, q
    for (int32_t i = 0; i < d_state; i++) {
        float ks = 0, vs = 0, qs = 0;
        for (int32_t j = 0; j < d_model; j++) {
            float xn_j = xn.data[j];
            ks += Wk[i * d_model + j] * xn_j;
            vs += Wv[i * d_model + j] * xn_j;
            qs += Wq[i * d_model + j] * xn_j;
        }
        k.data[i] = ks;
        v.data[i] = vs;
        q.data[i] = qs;
    }

    // 3. KV norm (in-place)
    rms_norm_forward(k, k, w.kv_norm_scale, 1e-6f);
    rms_norm_forward(v, v, w.kv_norm_scale, 1e-6f);

    // 4. Gates: sigmoid(W * xn + bias)
    for (int32_t i = 0; i < d_state; i++) {
        float gs = w.gamma_bias.data[i];
        float is_ = w.iota_bias.data[i];
        float rs = w.r_bias.data[i];
        for (int32_t j = 0; j < d_model; j++) {
            float xn_j = xn.data[j];
            gs += Wg[i * d_model + j] * xn_j;
            is_ += Wi[i * d_model + j] * xn_j;
            rs += Wr[i * d_model + j] * xn_j;
        }
        gamma.data[i] = 1.0f / (1.0f + std::exp(-gs));
        iota.data[i]  = 1.0f / (1.0f + std::exp(-is_));
        r.data[i]     = 1.0f / (1.0f + std::exp(-rs));
    }

    // 5+6. Outer product + state update
    for (int32_t i = 0; i < d_state; i++) {
        float gi = gamma.data[i];
        float ii = iota.data[i];
        float ki = k.data[i];
        for (int32_t j = 0; j < d_state; j++) {
            float outer_ij = ki * v.data[j];
            new_state.data[i * d_state + j] =
                gi * state.data[i * d_state + j] + ii * outer_ij;
        }
    }

    // 7. Read: h = S_new @ q
    for (int32_t i = 0; i < d_state; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < d_state; j++) {
            sum += new_state.data[i * d_state + j] * q.data[j];
        }
        h.data[i] = sum;
    }

    // 8. Output gate + projection + residual
    for (int32_t i = 0; i < d_state; i++) {
        rh.data[i] = r.data[i] * h.data[i];
    }

    for (int32_t i = 0; i < d_model; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < d_state; j++) {
            sum += Wo[i * d_state + j] * rh.data[j];
        }
        output.data[i] = sum + x.data[i];  // residual
    }
}

// ============================================================================
// Anchor Attention forward
// ============================================================================
void anchor_forward(Tensor& output, Tensor& new_wk, Tensor& new_wv,
                    const Tensor& x, const Tensor& window_k, const Tensor& window_v,
                    const Tensor& pmb_readouts, const AnchorWeights& w,
                    const Tensor& static_k, const Tensor& static_v,
                    bool causal_mask, Arena& arena) {
    int32_t d_model = w.norm_scale.shape.ne[0];
    int32_t n_heads = w.alibi_slopes.shape.ne[0];
    int32_t n_kv = window_k.shape.ne[1];
    int32_t head_dim = d_model / n_heads;
    int32_t q_dim = n_heads * head_dim;
    int32_t kv_dim = n_kv * head_dim;
    int32_t window_size = window_k.shape.ne[0];
    int32_t n_static = static_k.shape.ne[0];
    int32_t n_pmb = pmb_readouts.shape.ne[0];
    int32_t total_kv = n_static + n_pmb + window_size;
    int32_t n_groups = n_heads / n_kv;

    (void)causal_mask;

    // ⚡ Phase 9: FP16 wiring
    int32_t qkv_dim = q_dim + 2 * kv_dim;
    const float* Wqkv = fp16_wire(w.W_qkv.data, w.W_qkv_fp16, w.use_fp16, arena, (size_t)qkv_dim * d_model);

    // 1. Pre-norm
    auto xn = arena.alloc_tensor(TensorShape(d_model));
    rms_norm_forward(xn, x, w.norm_scale, 1e-6f);

    // 2. Fused QKV projection → qkv [q_dim + 2*kv_dim]
    auto qkv = arena.alloc_tensor(TensorShape(qkv_dim));
    for (int32_t i = 0; i < qkv_dim; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < d_model; j++) {
            sum += Wqkv[i * d_model + j] * xn.data[j];
        }
        qkv.data[i] = sum;
    }

    // Extract Q, new K, new V slices
    float* q_ptr = qkv.data;
    float* new_k_ptr = qkv.data + q_dim;
    float* new_v_ptr = qkv.data + q_dim + kv_dim;

    // 3. Attention: per-head
    memset(output.data, 0, d_model * sizeof(float));
    auto scores = arena.alloc_tensor(TensorShape(total_kv));

    for (int32_t h = 0; h < n_heads; h++) {
        int32_t kv_h = h / n_groups;

        // Compute scores = Q_h @ K^T / sqrt(head_dim) + ALiBi
        float max_score = -1e30f;
        for (int32_t t = 0; t < total_kv; t++) {
            float s = 0.0f;
            for (int32_t d = 0; d < head_dim; d++) {
                float q_val = q_ptr[h * head_dim + d];
                float k_val = 0.0f;
                if (t < n_static) {
                    k_val = static_k.data[t * n_kv * head_dim + kv_h * head_dim + d];
                } else if (t < n_static + n_pmb) {
                    k_val = pmb_readouts.data[(t - n_static) * d_model + kv_h * head_dim + d];
                } else {
                    int32_t wp = t - n_static - n_pmb;
                    k_val = window_k.data[wp * n_kv * head_dim + kv_h * head_dim + d];
                }
                s += q_val * k_val;
            }
            s /= std::sqrt((float)head_dim);

            // ALiBi for window portion
            if (t >= n_static + n_pmb) {
                int32_t wp = t - n_static - n_pmb;
                s += w.alibi_slopes.data[h] * (float)(-(window_size - wp));
            }
            scores.data[t] = s;
            if (s > max_score) max_score = s;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int32_t t = 0; t < total_kv; t++) {
            float v = std::exp(scores.data[t] - max_score);
            scores.data[t] = v;
            sum_exp += v;
        }
        float inv = 1.0f / (sum_exp + 1e-10f);

        // Weighted sum of V
        for (int32_t d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int32_t t = 0; t < total_kv; t++) {
                float v_val = 0.0f;
                if (t < n_static) {
                    v_val = static_v.data[t * n_kv * head_dim + kv_h * head_dim + d];
                } else if (t < n_static + n_pmb) {
                    v_val = pmb_readouts.data[(t - n_static) * d_model + kv_h * head_dim + d];
                } else {
                    int32_t wp = t - n_static - n_pmb;
                    v_val = window_v.data[wp * n_kv * head_dim + kv_h * head_dim + d];
                }
                sum += scores.data[t] * inv * v_val;
            }
            output.data[h * head_dim + d] += sum;
        }
    }

    // 4. Output projection: W_o @ attention_output
    const float* Wo_anchor = fp16_wire(w.W_o.data, w.W_o_fp16, w.use_fp16, arena, (size_t)d_model * q_dim);
    auto attn_out = arena.alloc_tensor(TensorShape(d_model));
    for (int32_t i = 0; i < d_model; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < q_dim; j++) {
            sum += Wo_anchor[i * q_dim + j] * output.data[j];
        }
        attn_out.data[i] = sum;
    }
    memcpy(output.data, attn_out.data, d_model * sizeof(float));

    // 5. Update window cache: shift left + append new K/V
    // new_wk/new_wv shape: [window_size, n_kv, head_dim] → contiguous
    int32_t kv_stride = n_kv * head_dim;
    for (int32_t p = 0; p < window_size - 1; p++) {
        memcpy(new_wk.data + p * kv_stride,
               window_k.data + (p + 1) * kv_stride,
               kv_stride * sizeof(float));
        memcpy(new_wv.data + p * kv_stride,
               window_v.data + (p + 1) * kv_stride,
               kv_stride * sizeof(float));
    }
    memcpy(new_wk.data + (window_size - 1) * kv_stride, new_k_ptr, kv_dim * sizeof(float));
    memcpy(new_wv.data + (window_size - 1) * kv_stride, new_v_ptr, kv_dim * sizeof(float));
}

// ============================================================================
// GatedShardFFN forward
// ============================================================================
void ffn_forward(Tensor& output, const Tensor& x, const FFNWeights& w, Arena& arena) {
    int32_t d_model = w.gate_proj_fused.shape.ne[1];
    int32_t total_inter = w.gate_proj_fused.shape.ne[0];
    int32_t n_shards = w.gate_head.shape.ne[0];
    int32_t shard_inter = total_inter / n_shards;

    // ⚡ Phase 9: FP16 wiring
    const float* Gp = fp16_wire(w.gate_proj_fused.data, w.gate_proj_fp16, w.use_fp16, arena, (size_t)total_inter * d_model);
    const float* Up = fp16_wire(w.up_proj_fused.data, w.up_proj_fp16, w.use_fp16, arena, (size_t)total_inter * d_model);
    const float* Dp = fp16_wire(w.down_proj_fused.data, w.down_proj_fp16, w.use_fp16, arena, (size_t)d_model * total_inter);
    const float* Gh = fp16_wire(w.gate_head.data, w.gate_head_fp16, w.use_fp16, arena, (size_t)n_shards * d_model);

    // 1. Pre-norm
    auto xn = arena.alloc_tensor(TensorShape(d_model));
    rms_norm_forward(xn, x, w.norm_scale, 1e-6f);

    // 2. Gate head: sigmoid(gate_head * xn + bias) → [n_shards]
    auto gates = arena.alloc_tensor(TensorShape(n_shards));
    for (int32_t s = 0; s < n_shards; s++) {
        float g = w.gate_head_bias.data[s];
        for (int32_t j = 0; j < d_model; j++)
            g += Gh[s * d_model + j] * xn.data[j];
        gates.data[s] = 1.0f / (1.0f + std::exp(-g));
    }

    // 3. SwiGLU: gate_proj ⊙ SiLU ⊗ up_proj, gated per shard
    auto swiglu = arena.alloc_tensor(TensorShape(total_inter));
    for (int32_t i = 0; i < total_inter; i++) {
        float gp = 0, up = 0;
        for (int32_t j = 0; j < d_model; j++) {
            gp += Gp[i * d_model + j] * xn.data[j];
            up += Up[i * d_model + j] * xn.data[j];
        }
        float silu_gp = gp / (1.0f + std::exp(-gp));
        float gated = gates.data[i / shard_inter] * silu_gp * up;
        swiglu.data[i] = gated;
    }

    // 4. Down projection
    memset(output.data, 0, d_model * sizeof(float));
    for (int32_t i = 0; i < d_model; i++) {
        float sum = 0;
        for (int32_t j = 0; j < total_inter; j++) {
            sum += Dp[i * total_inter + j] * swiglu.data[j];
        }
        output.data[i] = sum;
    }

    // 5. Residual
    for (int32_t i = 0; i < d_model; i++)
        output.data[i] += x.data[i];
}

// ============================================================================
// Halting head: hidden[d_model] → halting probability [1]
// ============================================================================
float halting_forward(const Tensor& hidden, const HaltingWeights& w, Arena& arena) {
    int32_t d_model = w.pool_proj.shape.ne[1];
    int32_t d_hidden = w.pool_proj.shape.ne[0];

    auto pooled = arena.alloc_tensor(TensorShape(d_hidden));

    for (int32_t i = 0; i < d_hidden; i++) {
        float sum = 0;
        for (int32_t j = 0; j < d_model; j++)
            sum += w.pool_proj.data[i * d_model + j] * hidden.data[j];
        pooled.data[i] = sum / (1.0f + std::exp(-sum));  // SiLU
    }

    float p = w.halt_bias.data[0];
    for (int32_t i = 0; i < d_hidden; i++)
        p += w.halt_proj.data[i] * pooled.data[i];

    return 1.0f / (1.0f + std::exp(-p));
}

// ============================================================================
// PMB read: query[d_model] → top-k readouts[n_readout, d_model]
// ============================================================================
void pmb_read(Tensor& readouts, const Tensor& query, const PMBWeights& w, Arena& arena) {
    int32_t n_slots = w.slots.shape.ne[0];
    int32_t d_model = w.slots.shape.ne[1];
    int32_t n_readout = readouts.shape.ne[0];

    auto scores = arena.alloc_tensor(TensorShape(n_slots));

    for (int32_t s = 0; s < n_slots; s++) {
        float sim = 0;
        for (int32_t d = 0; d < d_model; d++)
            sim += w.slots.data[s * d_model + d] * query.data[d];
        scores.data[s] = sim * w.write_scale;
    }

    // Take first n_readout slots (placeholder for actual top-k selection)
    for (int32_t k = 0; k < n_readout && k < n_slots; k++) {
        memcpy(readouts.data + k * d_model,
               w.slots.data + k * d_model,
               d_model * sizeof(float));
    }
}

// ============================================================================
// PMB write: chunk_summary → update slots (placeholder)
// ============================================================================
void pmb_write(const Tensor& chunk_summary, PMBWeights& w, Arena& arena) {
    (void)chunk_summary;
    (void)w;
    (void)arena;
}

// ============================================================================
// Initialize runtime state
// ============================================================================
void RuntimeState::init(const ModelConfig& cfg, Arena& arena) {
    glt_states.clear();
    window_k_caches.clear();
    window_v_caches.clear();

    for (int32_t i = 0; i < cfg.glt_layers; i++) {
        auto s = arena.alloc_tensor(TensorShape(cfg.d_state, cfg.d_state));
        s.fill(0.0f);
        glt_states.push_back(std::move(s));
    }
    for (int32_t i = 0; i < cfg.anchor_layers; i++) {
        auto wk = arena.alloc_tensor(TensorShape(cfg.window_size, cfg.n_kv_heads, cfg.head_dim));
        auto wv = arena.alloc_tensor(TensorShape(cfg.window_size, cfg.n_kv_heads, cfg.head_dim));
        wk.fill(0.0f);
        wv.fill(0.0f);
        window_k_caches.push_back(std::move(wk));
        window_v_caches.push_back(std::move(wv));
    }

    pmb_slots = arena.alloc_tensor(TensorShape(cfg.pmb_slots, cfg.d_model));
    pmb_slots.fill(0.0f);
}

void RuntimeState::reset() {
    for (auto& s : glt_states) s.fill(0.0f);
    for (auto& w : window_k_caches) w.fill(0.0f);
    for (auto& w : window_v_caches) w.fill(0.0f);
    pmb_slots.fill(0.0f);
}

// ============================================================================
// Continuum full forward: one token through all three stages
//
// Uses arena-allocated temps + move semantics to avoid Tensor copies.
// ============================================================================
void continuum_forward(
    Tensor& logits, RuntimeState& state, const Tensor& token_embed,
    const ModelWeights& weights, const ModelConfig& cfg, Arena& arena) {

    // ─── Store hidden state in an arena tensor ───
    auto x = arena.alloc_tensor(TensorShape(cfg.d_model));
    memcpy(x.data, token_embed.data, cfg.d_model * sizeof(float));

    int32_t glt_idx = 0, anchor_idx = 0;

    // ─── Stage 1: Perception ───
    for (int32_t l = 0; l < cfg.perception_layers; l++) {
        bool is_anchor = (anchor_idx < cfg.anchor_layers) &&
                         ((l + 1) % 3 == 0 || glt_idx >= cfg.glt_layers);

        auto tmp = arena.alloc_tensor(TensorShape(cfg.d_model));

        if (is_anchor) {
            anchor_forward(tmp, state.window_k_caches[anchor_idx],
                          state.window_v_caches[anchor_idx],
                          x, state.window_k_caches[anchor_idx],
                          state.window_v_caches[anchor_idx],
                          state.pmb_slots,
                          weights.anchor_layers[anchor_idx],
                          weights.anchor_layers[anchor_idx].static_anchors,
                          weights.anchor_layers[anchor_idx].static_anchors,
                          false, arena);
            anchor_idx++;
        } else {
            auto new_s = arena.alloc_tensor(TensorShape(cfg.d_state, cfg.d_state));
            glt_forward(tmp, new_s, x, state.glt_states[glt_idx],
                       weights.glt_layers[glt_idx], arena);
            state.glt_states[glt_idx] = std::move(new_s);
            glt_idx++;
        }

        auto tmp2 = arena.alloc_tensor(TensorShape(cfg.d_model));
        ffn_forward(tmp2, tmp, weights.ffn_layers[l], arena);
        memcpy(x.data, tmp2.data, cfg.d_model * sizeof(float));
    }

    // ─── Stage 2: Reasoning Core ───
    for (int32_t l = 0; l < cfg.core_layers; l++) {
        int32_t abs_l = cfg.perception_layers + l;
        bool is_anchor = (anchor_idx < cfg.anchor_layers) &&
                         (abs_l % 3 == 0 || glt_idx >= cfg.glt_layers);

        auto tmp = arena.alloc_tensor(TensorShape(cfg.d_model));

        if (is_anchor) {
            anchor_forward(tmp, state.window_k_caches[anchor_idx],
                          state.window_v_caches[anchor_idx],
                          x, state.window_k_caches[anchor_idx],
                          state.window_v_caches[anchor_idx],
                          state.pmb_slots,
                          weights.anchor_layers[anchor_idx],
                          weights.anchor_layers[anchor_idx].static_anchors,
                          weights.anchor_layers[anchor_idx].static_anchors,
                          false, arena);
            anchor_idx++;
        } else {
            auto new_s = arena.alloc_tensor(TensorShape(cfg.d_state, cfg.d_state));
            glt_forward(tmp, new_s, x, state.glt_states[glt_idx],
                       weights.glt_layers[glt_idx], arena);
            state.glt_states[glt_idx] = std::move(new_s);
            glt_idx++;
        }

        auto tmp2 = arena.alloc_tensor(TensorShape(cfg.d_model));
        ffn_forward(tmp2, tmp, weights.ffn_layers[abs_l], arena);
        memcpy(x.data, tmp2.data, cfg.d_model * sizeof(float));
    }

    // ─── Stage 3: Output ───
    for (int32_t l = 0; l < cfg.output_layers; l++) {
        int32_t abs_l = cfg.perception_layers + cfg.core_layers + l;
        bool is_anchor = (anchor_idx < cfg.anchor_layers) &&
                         (abs_l % 3 == 0 || glt_idx >= cfg.glt_layers);

        auto tmp = arena.alloc_tensor(TensorShape(cfg.d_model));

        if (is_anchor) {
            anchor_forward(tmp, state.window_k_caches[anchor_idx],
                          state.window_v_caches[anchor_idx],
                          x, state.window_k_caches[anchor_idx],
                          state.window_v_caches[anchor_idx],
                          state.pmb_slots,
                          weights.anchor_layers[anchor_idx],
                          weights.anchor_layers[anchor_idx].static_anchors,
                          weights.anchor_layers[anchor_idx].static_anchors,
                          false, arena);
            anchor_idx++;
        } else {
            auto new_s = arena.alloc_tensor(TensorShape(cfg.d_state, cfg.d_state));
            glt_forward(tmp, new_s, x, state.glt_states[glt_idx],
                       weights.glt_layers[glt_idx], arena);
            state.glt_states[glt_idx] = std::move(new_s);
            glt_idx++;
        }

        auto tmp2 = arena.alloc_tensor(TensorShape(cfg.d_model));
        ffn_forward(tmp2, tmp, weights.ffn_layers[abs_l], arena);
        memcpy(x.data, tmp2.data, cfg.d_model * sizeof(float));
    }

    // ─── Final norm + output projection ───
    auto xn = arena.alloc_tensor(TensorShape(cfg.d_model));
    rms_norm_forward(xn, x, weights.embed.final_norm_scale, 1e-6f);
    output_projection(logits, xn, weights.embed, arena);
}

// ============================================================================
// Weight loading from binary file (placeholder — real impl in continuum.cpp)
// ============================================================================
bool load_weights(ModelWeights& weights, const std::string& path, Arena& arena) {
    (void)weights;
    (void)path;
    (void)arena;
    return false;
}

} // namespace continuum
