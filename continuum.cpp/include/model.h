/*
 * model.h — Continuum SLM model architecture in C++.
 *
 * Ports the custom architecture:
 * - FactorizedEmbedding (16000 → 160 → 768)
 * - GLTLayer (Gated Linear Trace — matrix recurrence)
 * - AnchorAttention (window + static anchors + PMB, GQA, ALiBi)
 * - GatedShardFFN (soft-gated per-shard SwiGLU)
 * - RMSNorm
 * - HaltingHead (adaptive depth looping)
 * - PersistentMemoryBank (content-addressed long-term memory)
 */

#ifndef CONTINUUM_MODEL_H
#define CONTINUUM_MODEL_H

#include "tensor.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>

namespace continuum {

// ============================================================================
// Model Configuration (matches ContinuumConfig in Python)
// ============================================================================
struct ModelConfig {
    int32_t d_model = 768;
    int32_t d_state = 192;
    int32_t d_embed = 160;
    int32_t vocab_size = 16000;

    int32_t n_layers = 12;
    int32_t glt_layers = 9;
    int32_t anchor_layers = 3;
    int32_t perception_layers = 4;
    int32_t core_layers = 3;
    int32_t output_layers = 5;

    int32_t ffn_expansion = 4;
    int32_t ffn_shards = 6;

    int32_t n_heads = 12;
    int32_t n_kv_heads = 4;
    int32_t window_size = 128;
    int32_t n_anchors = 24;
    int32_t n_static_anchors = 8;

    int32_t n_max_loops = 5;
    float halt_threshold = 0.95f;

    int32_t pmb_slots = 64;
    int32_t pmb_readout = 16;
    int32_t chunk_size = 64;

    int32_t eos_token_id = 2;

    // Derived values (computed from config)
    int32_t head_dim = 0;
    int32_t kv_dim = 0;
    int32_t q_dim = 0;
    int32_t n_pmb_anchors = 0;
    int32_t ffn_shard_intermediate = 0;
    int32_t ffn_total_intermediate = 0;

    void init_derived() {
        head_dim = d_model / n_heads;
        kv_dim = n_kv_heads * head_dim;
        q_dim = n_heads * head_dim;
        n_pmb_anchors = n_anchors - n_static_anchors;
        ffn_total_intermediate = ffn_expansion * d_model;
        ffn_shard_intermediate = ffn_total_intermediate / ffn_shards;
        ffn_shard_intermediate = (ffn_shard_intermediate / 8) * 8;
        ffn_total_intermediate = ffn_shard_intermediate * ffn_shards;
    }
};

// ============================================================================
// Weights for one GLT Layer
// ============================================================================
struct GLTWeights {
    Tensor W_k;        // [d_state, d_model]
    Tensor W_v;        // [d_state, d_model]
    Tensor W_q;        // [d_state, d_model]
    Tensor W_gamma;    // [d_state, d_model]
    Tensor gamma_bias; // [d_state]
    Tensor W_iota;     // [d_state, d_model]
    Tensor iota_bias;  // [d_state]
    Tensor W_r;        // [d_state, d_model]
    Tensor r_bias;     // [d_state]
    Tensor W_o;        // [d_model, d_state]
    Tensor norm_scale; // [d_model]
    Tensor kv_norm_scale; // [d_state]
};

// ============================================================================
// Weights for one Anchor Attention Layer
// ============================================================================
struct AnchorWeights {
    Tensor W_qkv;      // [q_dim + 2*kv_dim, d_model]  -- fused QKV
    Tensor W_o;        // [d_model, q_dim]
    Tensor static_anchors; // [n_static_anchors, d_model]
    Tensor alibi_slopes;   // [n_heads]
    Tensor norm_scale;     // [d_model]
};

// ============================================================================
// Weights for one GatedShardFFN
// ============================================================================
struct FFNWeights {
    Tensor gate_proj_fused;  // [ffn_total_intermediate, d_model]
    Tensor up_proj_fused;    // [ffn_total_intermediate, d_model]
    Tensor down_proj_fused;  // [d_model, ffn_total_intermediate]
    Tensor gate_head;        // [n_shards, d_model]
    Tensor gate_head_bias;   // [n_shards]
    Tensor norm_scale;       // [d_model]
};

// ============================================================================
// Weights for FactorizedEmbedding + Output
// ============================================================================
struct EmbedWeights {
    Tensor embed_table;   // [vocab_size, d_embed]
    Tensor up_proj;       // [d_model, d_embed]
    Tensor down_proj;     // [d_embed, d_model]
    Tensor final_norm_scale; // [d_model]
};

// ============================================================================
// Weights for HaltingHead
// ============================================================================
struct HaltingWeights {
    Tensor pool_proj;     // [d_model/4, d_model]
    Tensor halt_proj;     // [1, d_model/4]
    Tensor halt_bias;     // [1]
};

// ============================================================================
// Weights for PersistentMemoryBank
// ============================================================================
struct PMBWeights {
    Tensor slots;         // [n_slots, d_model]
    Tensor W_update;      // [1, 2*d_model]
    Tensor update_bias;   // [1]
    float write_scale;
};

// ============================================================================
// Full model weights
// ============================================================================
struct ModelWeights {
    EmbedWeights embed;
    std::vector<GLTWeights> glt_layers;
    std::vector<AnchorWeights> anchor_layers;
    std::vector<FFNWeights> ffn_layers;
    HaltingWeights halting;
    PMBWeights pmb;
    ModelConfig config;
};

// ============================================================================
// Runtime State
// ============================================================================
struct RuntimeState {
    // GLT states: one [d_state, d_state] matrix per GLT layer
    std::vector<Tensor> glt_states;

    // Window caches: one (window_k, window_v) pair per Anchor layer
    std::vector<Tensor> window_k_caches;
    std::vector<Tensor> window_v_caches;

    // PMB slots
    Tensor pmb_slots;

    void init(const ModelConfig& cfg, Arena& arena);
    void reset();
};

// ============================================================================
// Forward declarations — layer forward passes
// ============================================================================

// GLT forward: x[d_model] + state[d_state,d_state] → output[d_model] + new_state
void glt_forward(Tensor& output, Tensor& new_state,
                 const Tensor& x, const Tensor& state,
                 const GLTWeights& w, Arena& arena);

// Anchor Attention forward: x + window_kv + pmb → output
void anchor_forward(Tensor& output, Tensor& new_wk, Tensor& new_wv,
                    const Tensor& x, const Tensor& window_k, const Tensor& window_v,
                    const Tensor& pmb_readouts, const AnchorWeights& w,
                    const Tensor& static_kv_k, const Tensor& static_kv_v,
                    bool causal_mask, Arena& arena);

// GatedShardFFN forward: x → output (pre-norm + residual inside)
void ffn_forward(Tensor& output, const Tensor& x, const FFNWeights& w, Arena& arena);

// FactorizedEmbedding: token_id → embedding[d_model]
void embed_forward(Tensor& output, int32_t token_id, const EmbedWeights& w, Arena& arena);

// Output projection: hidden[d_model] → logits[vocab_size]
void output_projection(Tensor& logits, const Tensor& hidden, const EmbedWeights& w, Arena& arena);

// RMSNorm
void rms_norm_forward(Tensor& output, const Tensor& x, const Tensor& scale, float eps);

// Halting head: hidden → halting probability
float halting_forward(const Tensor& hidden, const HaltingWeights& w, Arena& arena);

// PMB read: query[d_model] → top-k readouts[n_readout, d_model]
void pmb_read(Tensor& readouts, const Tensor& query, const PMBWeights& w, Arena& arena);

// PMB write: chunk_summary[d_model] → update slots
void pmb_write(const Tensor& chunk_summary, PMBWeights& w, Arena& arena);

// ============================================================================
// Full model forward (one token)
// ============================================================================
void continuum_forward(
    Tensor& logits,         // [1, vocab_size] output
    RuntimeState& state,    // mutable state
    const Tensor& token,    // [1, d_model] input embedding
    const ModelWeights& weights,
    const ModelConfig& cfg,
    Arena& arena
);

// ============================================================================
// Weight loading
// ============================================================================
bool load_weights(ModelWeights& weights, const std::string& path, Arena& arena);

} // namespace continuum

#endif // CONTINUUM_MODEL_H
