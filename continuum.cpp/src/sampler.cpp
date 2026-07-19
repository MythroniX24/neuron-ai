/*
 * sampler.cpp — Token sampling for Continuum SLM C++ inference.
 */

#include "sampler.h"
#include <cmath>

namespace continuum {

void Sampler::apply_repetition_penalty(Tensor& logits, const SamplerConfig& cfg) {
    if (cfg.repetition_penalty <= 1.0f || recent_tokens.empty()) return;

    int32_t V = cfg.vocab_size;
    for (int32_t tid : recent_tokens) {
        if (tid >= 0 && tid < V) {
            if (logits.data[tid] > 0)
                logits.data[tid] /= cfg.repetition_penalty;
            else
                logits.data[tid] *= cfg.repetition_penalty;
        }
    }
}

void Sampler::apply_temperature(Tensor& logits, float temp) {
    if (temp <= 0.01f) temp = 0.01f;
    float inv = 1.0f / temp;
    for (size_t i = 0; i < logits.n_elements(); i++)
        logits.data[i] *= inv;
}

void Sampler::apply_top_k(Tensor& logits, int32_t k, int32_t vocab_size) {
    if (k <= 0 || k >= vocab_size) return;

    // Find k-th largest value
    sorted_buf.resize(vocab_size);
    std::copy(logits.data, logits.data + vocab_size, sorted_buf.begin());
    std::nth_element(sorted_buf.begin(), sorted_buf.begin() + vocab_size - k,
                     sorted_buf.end());
    float threshold = sorted_buf[vocab_size - k];

    for (int32_t i = 0; i < vocab_size; i++)
        if (logits.data[i] < threshold)
            logits.data[i] = -1e30f;
}

void Sampler::apply_top_p(Tensor& logits, float p, int32_t vocab_size) {
    if (p >= 1.0f) return;

    sorted_buf.resize(vocab_size);
    sorted_idx.resize(vocab_size);
    for (int32_t i = 0; i < vocab_size; i++) {
        sorted_buf[i] = logits.data[i];
        sorted_idx[i] = i;
    }

    // Sort descending
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [this](int32_t a, int32_t b) {
                  return sorted_buf[a] > sorted_buf[b];
              });

    // Softmax
    float max_val = sorted_buf[sorted_idx[0]];
    float sum_exp = 0;
    for (int32_t i = 0; i < vocab_size; i++) {
        float v = std::exp(sorted_buf[sorted_idx[i]] - max_val);
        sorted_buf[i] = v;
        sum_exp += v;
    }

    // Find cumulative threshold
    float cumsum = 0;
    int32_t cutoff = vocab_size;
    for (int32_t i = 0; i < vocab_size; i++) {
        cumsum += sorted_buf[i] / sum_exp;
        if (cumsum > p) { cutoff = i + 1; break; }
    }

    // Zero out everything beyond cutoff
    for (int32_t i = cutoff; i < vocab_size; i++)
        logits.data[sorted_idx[i]] = -1e30f;
}

int32_t Sampler::multinomial_sample(const Tensor& probs) {
    int32_t V = (int32_t)probs.n_elements();

    // Softmax
    float max_val = -1e30f;
    for (int32_t i = 0; i < V; i++) max_val = std::max(max_val, probs.data[i]);

    float sum = 0;
    for (int32_t i = 0; i < V; i++) {
        probs.data[i] = std::exp(probs.data[i] - max_val);
        sum += probs.data[i];
    }

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float cum = 0;
    for (int32_t i = 0; i < V; i++) {
        cum += probs.data[i];
        if (cum >= r) return i;
    }

    return V - 1;  // fallback
}

int32_t Sampler::sample(const Tensor& logits, const SamplerConfig& cfg) {
    int32_t V = cfg.vocab_size;

    // Copy logits to mutable working buffer (Tensor copy is deleted)
    work_buf.resize(V);
    std::copy(logits.data, logits.data + V, work_buf.begin());
    Tensor work = Tensor::view(work_buf.data(), TensorShape(V));

    // 1. Repetition penalty
    apply_repetition_penalty(work, cfg);

    // 2. Temperature
    apply_temperature(work, cfg.temperature);

    // 3. Top-K
    apply_top_k(work, cfg.top_k, V);

    // 4. Top-P
    apply_top_p(work, cfg.top_p, V);

    // 5. Sample
    int32_t token = multinomial_sample(work);

    // Track for repetition penalty
    recent_tokens.push_back(token);
    if ((int32_t)recent_tokens.size() > 64)
        recent_tokens.erase(recent_tokens.begin());

    return token;
}

} // namespace continuum
