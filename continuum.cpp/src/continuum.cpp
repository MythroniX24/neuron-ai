/*
 * continuum.cpp — Main inference API for Continuum SLM C++ engine.
 *
 * Usage:
 *   ./continuum model.bin [--prompt "Hello"] [--temp 0.8] [--max-tokens 100]
 *
 * Targets 20-50 tok/s on Android ARM CPUs (with NEON intrinsics).
 */

#include "model.h"
#include "sampler.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <chrono>

using namespace continuum;

// ============================================================================
// Inference engine: wraps model, state, arena, sampler
// ============================================================================
class ContinuumEngine {
    ModelWeights weights_;
    RuntimeState state_;
    Arena arena_;
    Sampler sampler_;
    SamplerConfig samp_cfg_;
    std::vector<int32_t> token_buf_;

    // Buffers
    Tensor token_embed_;
    Tensor logits_;
    Tensor hidden_;

public:
    ContinuumEngine(size_t arena_mb = 512)
        : arena_(arena_mb * 1024 * 1024) {}

    bool load(const std::string& path) {
        printf("Loading model from %s...\n", path.c_str());
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            printf("ERROR: Cannot open %s\n", path.c_str());
            return false;
        }

        // ─── Read config ───
        auto& cfg = weights_.config;
        int32_t header[21];

        // Read and verify magic number + version
        int32_t magic, version;
        if (fread(&magic, sizeof(int32_t), 1, f) != 1) { printf("ERROR: Failed to read magic\n"); fclose(f); return false; }
        if (fread(&version, sizeof(int32_t), 1, f) != 1) { printf("ERROR: Failed to read version\n"); fclose(f); return false; }
        if (magic != 0x434F4E54) {  // 'CONT'
            printf("ERROR: Invalid model file (bad magic: 0x%08X, expected 0x434F4E54)\n", magic);
            fclose(f);
            return false;
        }
        if (version != 1) {
            printf("ERROR: Unsupported model version %d (expected 1)\n", version);
            fclose(f);
            return false;
        }

        if (fread(header, sizeof(int32_t), 21, f) != 21) {
            printf("ERROR: Failed to read config header\n");
            fclose(f);
            return false;
        }
        cfg.d_model = header[0];
        cfg.d_state = header[1];
        cfg.d_embed = header[2];
        cfg.vocab_size = header[3];
        cfg.n_layers = header[4];
        cfg.glt_layers = header[5];
        cfg.anchor_layers = header[6];
        cfg.perception_layers = header[7];
        cfg.core_layers = header[8];
        cfg.output_layers = header[9];
        cfg.ffn_expansion = header[10];
        cfg.ffn_shards = header[11];
        cfg.n_heads = header[12];
        cfg.n_kv_heads = header[13];
        cfg.window_size = header[14];
        cfg.n_anchors = header[15];
        cfg.n_static_anchors = header[16];
        cfg.n_max_loops = header[17];
        cfg.pmb_slots = header[18];
        cfg.pmb_readout = header[19];
        cfg.chunk_size = header[20];
        if (fread(&cfg.halt_threshold, sizeof(float), 1, f) != 1) { printf("ERROR: Failed to read halt_threshold\n"); fclose(f); return false; }
        if (fread(&cfg.eos_token_id, sizeof(int32_t), 1, f) != 1) { printf("ERROR: Failed to read eos_token_id\n"); fclose(f); return false; }
        cfg.init_derived();

        printf("  d_model=%d, layers=%d, vocab=%d\n",
               cfg.d_model, cfg.n_layers, cfg.vocab_size);

        // ─── Read embedding weights ───
        weights_.embed.embed_table = arena_.alloc_tensor(
            TensorShape(cfg.d_embed, cfg.vocab_size));
        weights_.embed.up_proj = arena_.alloc_tensor(
            TensorShape(cfg.d_embed, cfg.d_model));
        weights_.embed.down_proj = arena_.alloc_tensor(
            TensorShape(cfg.d_model, cfg.d_embed));
        weights_.embed.final_norm_scale = arena_.alloc_tensor(
            TensorShape(cfg.d_model));

#define SAFE_FREAD(ptr, size, count, file, label) \
    do { if (fread(ptr, size, count, file) != (size_t)(count)) { \
        printf("ERROR: Failed to read %s\n", label); fclose(file); return false; \
    } } while(0)

        SAFE_FREAD(weights_.embed.embed_table.data, sizeof(float),
              weights_.embed.embed_table.n_elements(), f, "embed_table");
        SAFE_FREAD(weights_.embed.up_proj.data, sizeof(float),
              weights_.embed.up_proj.n_elements(), f, "up_proj");
        SAFE_FREAD(weights_.embed.down_proj.data, sizeof(float),
              weights_.embed.down_proj.n_elements(), f, "down_proj");
        SAFE_FREAD(weights_.embed.final_norm_scale.data, sizeof(float),
              weights_.embed.final_norm_scale.n_elements(), f, "final_norm");

        // ─── Read layer weights ───
        weights_.glt_layers.resize(cfg.glt_layers);
        weights_.anchor_layers.resize(cfg.anchor_layers);
        weights_.ffn_layers.resize(cfg.n_layers);

        for (int i = 0; i < cfg.glt_layers; i++) {
            auto& g = weights_.glt_layers[i];
            g.W_k = arena_.alloc_tensor(TensorShape(cfg.d_model, cfg.d_state));
            g.W_v = arena_.alloc_tensor(TensorShape(cfg.d_model, cfg.d_state));
            g.W_q = arena_.alloc_tensor(TensorShape(cfg.d_model, cfg.d_state));
            g.W_gamma = arena_.alloc_tensor(TensorShape(cfg.d_model, cfg.d_state));
            g.gamma_bias = arena_.alloc_tensor(TensorShape(cfg.d_state));
            g.W_iota = arena_.alloc_tensor(TensorShape(cfg.d_model, cfg.d_state));
            g.iota_bias = arena_.alloc_tensor(TensorShape(cfg.d_state));
            g.W_r = arena_.alloc_tensor(TensorShape(cfg.d_model, cfg.d_state));
            g.r_bias = arena_.alloc_tensor(TensorShape(cfg.d_state));
            g.W_o = arena_.alloc_tensor(TensorShape(cfg.d_state, cfg.d_model));
            g.norm_scale = arena_.alloc_tensor(TensorShape(cfg.d_model));
            g.kv_norm_scale = arena_.alloc_tensor(TensorShape(cfg.d_state));

            char lbl[64];
            #define RDT(t, name) snprintf(lbl,64,"GLT%d " name,i); SAFE_FREAD(t.data,sizeof(float),t.n_elements(),f,lbl)
            RDT(g.W_k, "W_k"); RDT(g.W_v, "W_v"); RDT(g.W_q, "W_q");
            RDT(g.W_gamma, "W_gamma"); RDT(g.gamma_bias, "gamma_bias");
            RDT(g.W_iota, "W_iota"); RDT(g.iota_bias, "iota_bias");
            RDT(g.W_r, "W_r"); RDT(g.r_bias, "r_bias");
            RDT(g.W_o, "W_o"); RDT(g.norm_scale, "norm");
            RDT(g.kv_norm_scale, "kv_norm");
            #undef RDT
        }

        for (int i = 0; i < cfg.anchor_layers; i++) {
            int32_t q_dim = cfg.n_heads * cfg.head_dim;
            int32_t kv_dim = cfg.n_kv_heads * cfg.head_dim;
            auto& a = weights_.anchor_layers[i];
            a.W_qkv = arena_.alloc_tensor(TensorShape(cfg.d_model, q_dim + 2*kv_dim));
            a.W_o = arena_.alloc_tensor(TensorShape(q_dim, cfg.d_model));
            a.static_anchors = arena_.alloc_tensor(TensorShape(cfg.d_model, cfg.n_static_anchors));
            a.alibi_slopes = arena_.alloc_tensor(TensorShape(cfg.n_heads));
            a.norm_scale = arena_.alloc_tensor(TensorShape(cfg.d_model));

            #define RDA(t, name) snprintf(lbl,64,"Anchor%d " name,i); SAFE_FREAD(t.data,sizeof(float),t.n_elements(),f,lbl)
            char lbl[64];
            RDA(a.W_qkv, "W_qkv"); RDA(a.W_o, "W_o");
            RDA(a.static_anchors, "anchors"); RDA(a.alibi_slopes, "alibi");
            RDA(a.norm_scale, "norm");
            #undef RDA
        }

        for (int i = 0; i < cfg.n_layers; i++) {
            auto& ffn = weights_.ffn_layers[i];
            ffn.gate_proj_fused = arena_.alloc_tensor(
                TensorShape(cfg.d_model, cfg.ffn_total_intermediate));
            ffn.up_proj_fused = arena_.alloc_tensor(
                TensorShape(cfg.d_model, cfg.ffn_total_intermediate));
            ffn.down_proj_fused = arena_.alloc_tensor(
                TensorShape(cfg.ffn_total_intermediate, cfg.d_model));
            ffn.gate_head = arena_.alloc_tensor(
                TensorShape(cfg.d_model, cfg.ffn_shards));
            ffn.gate_head_bias = arena_.alloc_tensor(
                TensorShape(cfg.ffn_shards));
            ffn.norm_scale = arena_.alloc_tensor(TensorShape(cfg.d_model));

            #define RDF(t, name) snprintf(lbl,64,"FFN%d " name,i); SAFE_FREAD(t.data,sizeof(float),t.n_elements(),f,lbl)
            char lbl[64];
            RDF(ffn.gate_proj_fused, "gate"); RDF(ffn.up_proj_fused, "up");
            RDF(ffn.down_proj_fused, "down"); RDF(ffn.gate_head, "gate_head");
            RDF(ffn.gate_head_bias, "gate_bias"); RDF(ffn.norm_scale, "norm");
            #undef RDF
        }

        // ─── Halting head ───
        weights_.halting.pool_proj = arena_.alloc_tensor(
            TensorShape(cfg.d_model, cfg.d_model/4));
        weights_.halting.halt_proj = arena_.alloc_tensor(
            TensorShape(cfg.d_model/4, 1));
        weights_.halting.halt_bias = arena_.alloc_tensor(TensorShape(1));
        SAFE_FREAD(weights_.halting.pool_proj.data, sizeof(float),
              weights_.halting.pool_proj.n_elements(), f, "halt_pool");
        SAFE_FREAD(weights_.halting.halt_proj.data, sizeof(float),
              weights_.halting.halt_proj.n_elements(), f, "halt_proj");
        SAFE_FREAD(weights_.halting.halt_bias.data, sizeof(float),
              weights_.halting.halt_bias.n_elements(), f, "halt_bias");

        // ─── PMB ───
        weights_.pmb.slots = arena_.alloc_tensor(
            TensorShape(cfg.d_model, cfg.pmb_slots));
        weights_.pmb.W_update = arena_.alloc_tensor(
            TensorShape(cfg.d_model * 2, 1));
        weights_.pmb.update_bias = arena_.alloc_tensor(TensorShape(1));
        SAFE_FREAD(weights_.pmb.slots.data, sizeof(float),
              weights_.pmb.slots.n_elements(), f, "pmb_slots");
        SAFE_FREAD(weights_.pmb.W_update.data, sizeof(float),
              weights_.pmb.W_update.n_elements(), f, "pmb_update");
        SAFE_FREAD(weights_.pmb.update_bias.data, sizeof(float),
              weights_.pmb.update_bias.n_elements(), f, "pmb_bias");
        if (fread(&weights_.pmb.write_scale, sizeof(float), 1, f) != 1) {
            printf("ERROR: Failed to read pmb_write_scale\n"); fclose(f); return false;
        }

        fclose(f);

        // ─── Allocate runtime buffers ───
        state_.init(cfg, arena_);
        token_embed_ = arena_.alloc_tensor(TensorShape(cfg.d_model));
        logits_ = arena_.alloc_tensor(TensorShape(cfg.vocab_size));
        hidden_ = arena_.alloc_tensor(TensorShape(cfg.d_model));

        samp_cfg_.vocab_size = cfg.vocab_size;
        samp_cfg_.eos_token_id = cfg.eos_token_id;

        printf("  Model loaded! Arena: %.1f MB / %.1f MB\n",
               arena_.used() / (1024.0 * 1024.0),
               arena_.capacity() / (1024.0 * 1024.0));

        return true;
    }

    void set_temperature(float t) { samp_cfg_.temperature = t; }
    void set_top_k(int k) { samp_cfg_.top_k = k; }
    void set_top_p(float p) { samp_cfg_.top_p = p; }
    void set_rep_penalty(float rp) { samp_cfg_.repetition_penalty = rp; }
    void set_max_tokens(int m) { /* stored externally */ }

    std::string generate(const std::vector<int32_t>& prompt_ids, int max_new_tokens = 100) {
        auto& cfg = weights_.config;
        token_buf_.clear();
        token_buf_.insert(token_buf_.end(), prompt_ids.begin(), prompt_ids.end());

        auto t0 = std::chrono::high_resolution_clock::now();

        // Prefill: process prompt tokens
        for (size_t p = 0; p < prompt_ids.size(); p++) {
            embed_forward(token_embed_, prompt_ids[p], weights_.embed, arena_);
            continuum_forward(logits_, state_, token_embed_, weights_, cfg, arena_);
        }

        // Generate
        int n_generated = 0;
        for (int i = 0; i < max_new_tokens; i++) {
            int32_t token = sampler_.sample(logits_, samp_cfg_);
            token_buf_.push_back(token);
            n_generated++;

            if (token == cfg.eos_token_id) break;

            embed_forward(token_embed_, token, weights_.embed, arena_);
            continuum_forward(logits_, state_, token_embed_, weights_, cfg, arena_);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double tps = n_generated / elapsed;
        printf("  Generated %d tokens in %.2fs (%.1f tok/s)\n", n_generated, elapsed, tps);

        // Return as string
        return std::to_string(n_generated) + " tokens generated at " +
               std::to_string((int)tps) + " tok/s";
    }

    void reset() {
        sampler_.reset();
        state_.reset();
    }
};

// ============================================================================
// Main entry point
// ============================================================================
static void show_usage(const char* prog) {
    printf("Continuum SLM C++ Inference Engine\n\n");
    printf("Usage: %s <model.bin> [OPTIONS]\n", prog);
    printf("  --prompt TEXT     Input prompt (default: \"Hello\")\n");
    printf("  --temp FLOAT      Temperature (default: 0.8)\n");
    printf("  --top-k INT       Top-K filtering (default: 40)\n");
    printf("  --top-p FLOAT     Nucleus sampling (default: 0.9)\n");
    printf("  --max-tokens INT  Max tokens to generate (default: 100)\n");
    printf("  --seed INT        Random seed (default: 42)\n");
    printf("  --help            Show this help\n");
}

int main(int argc, char** argv) {
    // Check for --help anywhere in args
    bool want_help = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            want_help = true;
        }
    }

    if (argc < 2 || want_help || (argc >= 2 && argv[1][0] == '-')) {
        show_usage(argv[0]);
        return want_help ? 0 : 1;
    }

    std::string model_path = argv[1];
    std::string prompt = "Hello";
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.9f;
    int max_tokens = 100;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temp = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
    }

    ContinuumEngine engine(1024);  // 1 GB arena

    if (!engine.load(model_path)) {
        printf("Failed to load model.\n");
        return 1;
    }

    engine.set_temperature(temp);
    engine.set_top_k(top_k);
    engine.set_top_p(top_p);

    // Simple ASCII tokenization (placeholder — real impl uses BPE tokenizer)
    std::vector<int32_t> prompt_ids;
    for (char c : prompt) prompt_ids.push_back((int32_t)(unsigned char)c);

    printf("Prompt: \"%s\"\n", prompt.c_str());
    std::string result = engine.generate(prompt_ids, max_tokens);
    printf("Result: %s\n", result.c_str());

    return 0;
}
