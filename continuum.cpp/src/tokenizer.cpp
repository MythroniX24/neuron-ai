/*
 * tokenizer.cpp — BPE tokenizer in pure C++ for mobile inference.
 *
 * Phase D: Eliminates Python tokenizer dependency on phone.
 * Implements Byte-Pair Encoding with merge ranks, special tokens,
 * and a compact binary format for fast loading.
 */

#include "tokenizer.h"
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace continuum {

// ============================================================================
// Binary read/write helpers
// ============================================================================
void BPETokenizer::write_u32(FILE* f, uint32_t val) {
    fwrite(&val, sizeof(uint32_t), 1, f);
}
void BPETokenizer::write_u16(FILE* f, uint16_t val) {
    fwrite(&val, sizeof(uint16_t), 1, f);
}
void BPETokenizer::write_str(FILE* f, const std::string& s) {
    write_u16(f, (uint16_t)s.size());
    fwrite(s.data(), 1, s.size(), f);
}
uint32_t BPETokenizer::read_u32(FILE* f) {
    uint32_t v = 0;
    fread(&v, sizeof(uint32_t), 1, f);
    return v;
}
uint16_t BPETokenizer::read_u16(FILE* f) {
    uint16_t v = 0;
    fread(&v, sizeof(uint16_t), 1, f);
    return v;
}
std::string BPETokenizer::read_str(FILE* f) {
    uint16_t len = read_u16(f);
    std::string s(len, '\0');
    fread(&s[0], 1, len, f);
    return s;
}

// ============================================================================
// Load from binary format
// ============================================================================
bool BPETokenizer::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic = read_u32(f);
    if (magic != 0x54504F42) { fclose(f); return false; }  // "BPTO"

    uint32_t version = read_u32(f);
    (void)version;

    vocab_size_ = (int32_t)read_u32(f);
    num_merges_ = (int32_t)read_u32(f);
    uint32_t num_special = read_u32(f);

    // Load vocab
    id_to_token_.resize(vocab_size_);
    token_to_id_.reserve(vocab_size_);
    for (int32_t i = 0; i < vocab_size_; i++) {
        id_to_token_[i] = read_str(f);
        token_to_id_[id_to_token_[i]] = i;
    }

    // Load merges
    merge_ranks_.clear();
    merge_ranks_.reserve(num_merges_);
    for (int i = 0; i < num_merges_; i++) {
        int32_t rank = (int32_t)read_u32(f);
        std::string a = read_str(f);
        std::string b = read_str(f);
        merge_ranks_[a + b] = rank;
    }

    // Load special tokens
    special_tokens_.clear();
    for (uint32_t i = 0; i < num_special; i++) {
        int32_t id = (int32_t)read_u32(f);
        std::string tok = read_str(f);
        special_tokens_[tok] = id;
        if (tok == "<|eos|>") eos_id_ = id;
        if (tok == "<|user|>") user_id_ = id;
        if (tok == "<|assistant|>") assistant_id_ = id;
        if (tok == "<|system|>") system_id_ = id;
    }

    fclose(f);
    loaded_ = true;
    return true;
}

// ============================================================================
// Save to binary format
// ============================================================================
bool BPETokenizer::save(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    write_u32(f, 0x54504F42);  // magic "BPTO"
    write_u32(f, 1);            // version
    write_u32(f, (uint32_t)vocab_size_);
    write_u32(f, (uint32_t)num_merges_);
    write_u32(f, (uint32_t)special_tokens_.size());

    // Write vocab
    for (int32_t i = 0; i < vocab_size_; i++) {
        write_str(f, id_to_token_[i]);
    }

    // Write merges
    for (const auto& [merged, rank] : merge_ranks_) {
        write_u32(f, (uint32_t)rank);
        // We store merged string — need to split back into a+b
        // Actually we store the full merged token, rank is the key
        // For simplicity, store the merged token as a, empty as b
        write_str(f, merged);
        write_str(f, "");
    }

    // Write special tokens
    for (const auto& [tok, id] : special_tokens_) {
        write_u32(f, (uint32_t)id);
        write_str(f, tok);
    }

    fclose(f);
    return true;
}

// ============================================================================
// Build from Python data (used by export script)
// ============================================================================
void BPETokenizer::build_from_python(
    const std::vector<std::string>& vocab,
    const std::vector<std::pair<std::string, std::string>>& merges,
    int32_t eos_id, int32_t user_id, int32_t assistant_id, int32_t system_id
) {
    vocab_size_ = (int32_t)vocab.size();
    num_merges_ = (int32_t)merges.size();
    id_to_token_ = vocab;
    token_to_id_.reserve(vocab_size_);
    for (int32_t i = 0; i < vocab_size_; i++) {
        token_to_id_[vocab[i]] = i;
    }
    merge_ranks_.clear();
    for (int i = 0; i < num_merges_; i++) {
        merge_ranks_[merges[i].first + merges[i].second] = i;
    }
    eos_id_ = eos_id;
    user_id_ = user_id;
    assistant_id_ = assistant_id;
    system_id_ = system_id;
    special_tokens_["<|eos|>"] = eos_id;
    if (user_id >= 0) special_tokens_["<|user|>"] = user_id;
    if (assistant_id >= 0) special_tokens_["<|assistant|>"] = assistant_id;
    if (system_id >= 0) special_tokens_["<|system|>"] = system_id;
    loaded_ = true;
}

// ============================================================================
// BPE encode a single word
// ============================================================================
std::vector<int32_t> BPETokenizer::bpe_encode(const std::string& word) const {
    // Split word into characters (byte-level BPE)
    std::vector<std::string> symbols;
    for (size_t i = 0; i < word.size(); i++) {
        symbols.push_back(std::string(1, word[i]));
    }

    if (symbols.size() <= 1) {
        // Single char or empty — lookup directly
        std::vector<int32_t> result;
        for (const auto& s : symbols) {
            auto it = token_to_id_.find(s);
            if (it != token_to_id_.end()) result.push_back(it->second);
        }
        return result;
    }

    // Greedy BPE merges: find lowest-rank merge, apply, repeat
    while (symbols.size() > 1) {
        int32_t best_rank = -1;
        size_t best_idx = 0;
        for (size_t i = 0; i < symbols.size() - 1; i++) {
            auto it = merge_ranks_.find(symbols[i] + symbols[i + 1]);
            if (it != merge_ranks_.end() && (best_rank == -1 || it->second < best_rank)) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_rank == -1) break;  // no more merges

        // Apply best merge
        symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    // Convert symbols to token IDs
    std::vector<int32_t> result;
    for (const auto& s : symbols) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            result.push_back(it->second);
        }
    }
    return result;
}

// ============================================================================
// Encode text → token IDs
// ============================================================================
std::vector<int32_t> BPETokenizer::encode(const std::string& text, bool add_special) const {
    std::vector<int32_t> tokens;

    if (add_special && user_id_ >= 0 && assistant_id_ >= 0) {
        tokens.push_back(user_id_);
    }

    // Simple word-level splitting (split on whitespace, keep punctuation)
    std::string current_word;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == ' ' || c == '\n' || c == '\t') {
            if (!current_word.empty()) {
                auto word_tokens = bpe_encode(current_word);
                tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
                current_word.clear();
            }
            // Encode whitespace as its own token
            auto ws_tokens = bpe_encode(std::string(1, c));
            tokens.insert(tokens.end(), ws_tokens.begin(), ws_tokens.end());
        } else {
            current_word += c;
        }
    }
    if (!current_word.empty()) {
        auto word_tokens = bpe_encode(current_word);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }

    if (add_special && assistant_id_ >= 0) {
        tokens.push_back(assistant_id_);
    }

    return tokens;
}

// ============================================================================
// Decode token IDs → text
// ============================================================================
std::string BPETokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string result;
    for (int32_t id : token_ids) {
        if (id == eos_id_ || id == user_id_ || id == assistant_id_ || id == system_id_) {
            continue;  // skip special tokens in output
        }
        result += decode_token(id);
    }
    return result;
}

std::string BPETokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= vocab_size_) return "";
    return id_to_token_[token_id];
}

} // namespace continuum
