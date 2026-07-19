/*
 * sampler.h — Token sampling for Continuum SLM C++ inference.
 *
 * Supports: temperature, top-k, top-p (nucleus), repetition penalty.
 */

#ifndef CONTINUUM_SAMPLER_H
#define CONTINUUM_SAMPLER_H

#include "tensor.h"
#include <vector>
#include <random>
#include <algorithm>

namespace continuum {

struct SamplerConfig {
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    int32_t vocab_size = 16000;
    int32_t eos_token_id = 2;
};

class Sampler {
    std::mt19937 rng;
    std::vector<int32_t> recent_tokens;   // last 64 tokens for repetition penalty
    std::vector<float> sorted_buf;        // reusable sort buffer
    std::vector<int32_t> sorted_idx;      // reusable index buffer
    std::vector<float> work_buf;          // mutable logits copy (Tensor cannot be copied)

public:
    Sampler(int seed = 42) : rng(seed) {
        recent_tokens.reserve(64);
        sorted_buf.reserve(16000);
        sorted_idx.reserve(16000);
        work_buf.reserve(16000);
    }

    // Sample one token from logits
    // logits: [1, vocab_size]
    int32_t sample(const Tensor& logits, const SamplerConfig& cfg);

    // Reset for new conversation
    void reset() {
        recent_tokens.clear();
    }

private:
    void apply_repetition_penalty(Tensor& logits, const SamplerConfig& cfg);
    void apply_temperature(Tensor& logits, float temp);
    void apply_top_k(Tensor& logits, int32_t k, int32_t vocab_size);
    void apply_top_p(Tensor& logits, float p, int32_t vocab_size);
    int32_t multinomial_sample(const Tensor& probs);
};

} // namespace continuum

#endif // CONTINUUM_SAMPLER_H
