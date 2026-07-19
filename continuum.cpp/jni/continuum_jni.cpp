/*
 * continuum_jni.cpp — JNI bridge for Continuum SLM on Android.
 *
 * Exposes the C++ inference engine to Java/Kotlin via JNI.
 *
 * Kotlin usage:
 *   class ContinuumEngine {
 *       init { System.loadLibrary("continuum_jni") }
 *       external fun loadModel(path: String): Boolean
 *       external fun generate(prompt: String, maxTokens: Int, temp: Float): String
 *       external fun reset(): Void
 *       external fun getModelInfo(): String
 *   }
 */

#include <jni.h>
#include <string>
#include <cstring>
#include <cstdio>

// Include the Phase 3 C++ inference engine
#include "model.h"
#include "sampler.h"
#include "tensor.h"

using namespace continuum;

// ============================================================================
// Global engine instance (one per process)
// ============================================================================
static ContinuumEngine* g_engine = nullptr;
static bool g_loaded = false;

// ============================================================================
// Helper: convert jstring to std::string
// ============================================================================
static std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

// ============================================================================
// ContinuumEngine: wraps model + weights + state + sampler
// ============================================================================
class ContinuumEngine {
    ModelWeights weights_;
    RuntimeState state_;
    Arena arena_;
    Sampler sampler_;
    SamplerConfig samp_cfg_;
    std::vector<int32_t> token_buf_;

    Tensor token_embed_;
    Tensor logits_;
    Tensor hidden_;

public:
    ContinuumEngine(size_t arena_mb = 256)
        : arena_(arena_mb * 1024 * 1024) {}

    // Reads model in CONT binary format (from export_to_cpp.py).
    // For GGUF files, use export_to_cpp.py first: python export_to_cpp.py --checkpoint model.pt
    bool load_from_file(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "JNI: Cannot open %s\n", path.c_str());
            return false;
        }

        auto& cfg = weights_.config;

        // Read magic + version (CONT format: 0x434F4E54 version 1)
        int32_t magic, version;
        if (fread(&magic, sizeof(int32_t), 1, f) != 1 ||
            fread(&version, sizeof(int32_t), 1, f) != 1) {
            fprintf(stderr, "JNI: Failed to read header\n");
            fclose(f);
            return false;
        }

        if (magic == 0x47554746) {  // GGUF
            fprintf(stderr, "JNI: GGUF format detected. Convert with: python exports/export_to_cpp.py\n");
            fclose(f);
            return false;
        }

        if (magic != 0x434F4E54) {  // 'CONT'
            fprintf(stderr, "JNI: Bad magic: 0x%08X\n", magic);
            fclose(f);
            return false;
        }

        // Read config header
        int32_t header[21];
        if (fread(header, sizeof(int32_t), 21, f) != 21) {
            fprintf(stderr, "JNI: Failed to read config\n");
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

        fread(&cfg.halt_threshold, sizeof(float), 1, f);
        fread(&cfg.eos_token_id, sizeof(int32_t), 1, f);
        cfg.init_derived();

#define R(t, s) { t = arena_.alloc_tensor(TensorShape s); \
    fread(t.data, sizeof(float), t.n_elements(), f); }

        // Embedding
        auto& e = weights_.embed;
        R(e.embed_table, (cfg.d_embed, cfg.vocab_size));
        R(e.up_proj, (cfg.d_embed, cfg.d_model));
        R(e.down_proj, (cfg.d_model, cfg.d_embed));
        R(e.final_norm_scale, (cfg.d_model));

        // GLT layers
        weights_.glt_layers.resize(cfg.glt_layers);
        for (int i = 0; i < cfg.glt_layers; i++) {
            auto& g = weights_.glt_layers[i];
            R(g.W_k, (cfg.d_model, cfg.d_state));
            R(g.W_v, (cfg.d_model, cfg.d_state));
            R(g.W_q, (cfg.d_model, cfg.d_state));
            R(g.W_gamma, (cfg.d_model, cfg.d_state));
            R(g.gamma_bias, (cfg.d_state));
            R(g.W_iota, (cfg.d_model, cfg.d_state));
            R(g.iota_bias, (cfg.d_state));
            R(g.W_r, (cfg.d_model, cfg.d_state));
            R(g.r_bias, (cfg.d_state));
            R(g.W_o, (cfg.d_state, cfg.d_model));
            R(g.norm_scale, (cfg.d_model));
            R(g.kv_norm_scale, (cfg.d_state));
        }

        // Anchor layers
        weights_.anchor_layers.resize(cfg.anchor_layers);
        for (int i = 0; i < cfg.anchor_layers; i++) {
            auto& a = weights_.anchor_layers[i];
            int32_t q_dim = cfg.n_heads * cfg.head_dim;
            int32_t kv_dim = cfg.n_kv_heads * cfg.head_dim;
            R(a.W_qkv, (cfg.d_model, q_dim + 2*kv_dim));
            R(a.W_o, (q_dim, cfg.d_model));
            R(a.static_anchors, (cfg.d_model, cfg.n_static_anchors));
            R(a.alibi_slopes, (cfg.n_heads));
            R(a.norm_scale, (cfg.d_model));
        }

        // FFN layers
        weights_.ffn_layers.resize(cfg.n_layers);
        for (int i = 0; i < cfg.n_layers; i++) {
            auto& ffn = weights_.ffn_layers[i];
            R(ffn.gate_proj_fused, (cfg.d_model, cfg.ffn_total_intermediate));
            R(ffn.up_proj_fused, (cfg.d_model, cfg.ffn_total_intermediate));
            R(ffn.down_proj_fused, (cfg.ffn_total_intermediate, cfg.d_model));
            R(ffn.gate_head, (cfg.d_model, cfg.ffn_shards));
            R(ffn.gate_head_bias, (cfg.ffn_shards));
            R(ffn.norm_scale, (cfg.d_model));
        }

        // Halting
        R(weights_.halting.pool_proj, (cfg.d_model, cfg.d_model/4));
        R(weights_.halting.halt_proj, (cfg.d_model/4, 1));
        R(weights_.halting.halt_bias, (1));

        // PMB
        R(weights_.pmb.slots, (cfg.d_model, cfg.pmb_slots));
        R(weights_.pmb.W_update, (cfg.d_model * 2, 1));
        R(weights_.pmb.update_bias, (1));
        fread(&weights_.pmb.write_scale, sizeof(float), 1, f);

#undef R

        fclose(f);

        // Allocate runtime buffers
        state_.init(cfg, arena_);
        token_embed_ = arena_.alloc_tensor(TensorShape(cfg.d_model));
        logits_ = arena_.alloc_tensor(TensorShape(cfg.vocab_size));
        hidden_ = arena_.alloc_tensor(TensorShape(cfg.d_model));

        samp_cfg_.vocab_size = cfg.vocab_size;
        samp_cfg_.eos_token_id = cfg.eos_token_id;

        fprintf(stderr, "JNI: Model loaded (%d params)\\n",
                cfg.n_layers * cfg.d_model * cfg.d_model * 4 / 1000000);
        return true;
    }

    std::string generate(const std::string& prompt, int max_tokens, float temp) {
        auto& cfg = weights_.config;

        samp_cfg_.temperature = temp;

        // Simple ASCII tokenization
        token_buf_.clear();
        for (char c : prompt) {
            token_buf_.push_back((int32_t)(unsigned char)c);
        }

        // Prefill
        for (size_t p = 0; p < token_buf_.size(); p++) {
            embed_forward(token_embed_, token_buf_[p], weights_.embed, arena_);
            continuum_forward(logits_, state_, token_embed_, weights_, cfg, arena_);
        }

        // Generate
        std::string result;
        for (int i = 0; i < max_tokens; i++) {
            int32_t token = sampler_.sample(logits_, samp_cfg_);
            if (token == cfg.eos_token_id) break;
            result += (char)(token & 0xFF);

            embed_forward(token_embed_, token, weights_.embed, arena_);
            continuum_forward(logits_, state_, token_embed_, weights_, cfg, arena_);
        }

        return result;
    }

    void reset() {
        sampler_.reset();
        state_.reset();
    }

    std::string get_info() {
        auto& cfg = weights_.config;
        char buf[256];
        snprintf(buf, sizeof(buf),
            "Continuum SLM | d_model=%d, layers=%d, vocab=%d",
            cfg.d_model, cfg.n_layers, cfg.vocab_size);
        return std::string(buf);
    }
};

// ============================================================================
// JNI Exports
// ============================================================================

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_continuum_slm_ContinuumEngine_loadModel(
    JNIEnv* env, jobject /* this */, jstring path) {

    if (g_engine) {
        delete g_engine;
        g_engine = nullptr;
        g_loaded = false;
    }

    g_engine = new ContinuumEngine(512);  // 512 MB arena
    std::string p = jstring_to_string(env, path);
    g_loaded = g_engine->load_from_file(p);
    return g_loaded ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_continuum_slm_ContinuumEngine_generate(
    JNIEnv* env, jobject /* this */,
    jstring prompt, jint maxTokens, jfloat temperature) {

    if (!g_loaded || !g_engine) {
        return env->NewStringUTF("ERROR: Model not loaded");
    }

    std::string p = jstring_to_string(env, prompt);
    std::string result = g_engine->generate(p, maxTokens, temperature);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_continuum_slm_ContinuumEngine_reset(
    JNIEnv* /* env */, jobject /* this */) {
    if (g_engine) {
        g_engine->reset();
    }
}

JNIEXPORT jstring JNICALL
Java_com_continuum_slm_ContinuumEngine_getModelInfo(
    JNIEnv* env, jobject /* this */) {
    if (!g_engine) {
        return env->NewStringUTF("No model loaded");
    }
    return env->NewStringUTF(g_engine->get_info().c_str());
}

} // extern "C"
