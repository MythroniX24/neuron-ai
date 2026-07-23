/*
 * tokenizer.h — BPE tokenizer in pure C++ for Continuum SLM mobile inference.
 *
 * Phase D: C++ tokenizer — eliminates Python dependency on phone.
 * - Loads BPE vocab + merges from a compact binary format
 * - encode(): text → token IDs
 * - decode(): token IDs → text
 * - Special tokens: <|user|>, <|assistant|>, <|system|>, <|eos|>
 *
 * Binary format (tokenizer.bin):
 *   uint32_t magic (0x54504F42 = "BPTO" reversed)
 *   uint32_t version
 *   uint32_t vocab_size
 *   uint32_t num_merges
 *   uint32_t num_special_tokens
 *   For each vocab entry: uint16_t len + bytes (UTF-8 token string)
 *   For each merge: uint32_t rank + uint16_t len_a + bytes_a + uint16_t len_b + bytes_b
 *   For each special token: uint32_t id + uint16_t len + bytes
 */

#ifndef CONTINUUM_TOKENIZER_H
#define CONTINUUM_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace continuum {

class BPETokenizer {
public:
    BPETokenizer() = default;

    // Load from binary format file
    bool load(const std::string& path);

    // Save Python tokenizer to binary format (used by export script)
    bool save(const std::string& path) const;

    // Encode text → token IDs
    // add_special: wrap with <|user|>\n...\n<|assistant|>\n
    std::vector<int32_t> encode(const std::string& text, bool add_special = true) const;

    // Decode token IDs → text
    std::string decode(const std::vector<int32_t>& token_ids) const;

    // Decode single token → text
    std::string decode_token(int32_t token_id) const;

    // Get special token IDs
    int32_t eos_id() const { return eos_id_; }
    int32_t user_id() const { return user_id_; }
    int32_t assistant_id() const { return assistant_id_; }
    int32_t system_id() const { return system_id_; }

    bool is_loaded() const { return loaded_; }
    int32_t vocab_size() const { return vocab_size_; }

    // Build tokenizer from Python data (used by export script)
    void build_from_python(
        const std::vector<std::string>& vocab,
        const std::vector<std::pair<std::string, std::string>>& merges,
        int32_t eos_id, int32_t user_id, int32_t assistant_id, int32_t system_id
    );

private:
    bool loaded_ = false;
    int32_t vocab_size_ = 0;
    int32_t num_merges_ = 0;

    // vocab[id] = token string
    std::vector<std::string> id_to_token_;

    // token string → id
    std::unordered_map<std::string, int32_t> token_to_id_;

    // merge rank: pair(token_a, token_b) → rank
    std::unordered_map<std::string, int32_t> merge_ranks_;

    // Special tokens
    int32_t eos_id_ = 2;
    int32_t user_id_ = -1;
    int32_t assistant_id_ = -1;
    int32_t system_id_ = -1;

    std::unordered_map<std::string, int32_t> special_tokens_;

    // BPE merge algorithm for a single word
    std::vector<int32_t> bpe_encode(const std::string& word) const;

    // Read/write helpers
    static void write_u32(FILE* f, uint32_t val);
    static void write_u16(FILE* f, uint16_t val);
    static void write_str(FILE* f, const std::string& s);
    static uint32_t read_u32(FILE* f);
    static uint16_t read_u16(FILE* f);
    static std::string read_str(FILE* f);
};

} // namespace continuum

#endif // CONTINUUM_TOKENIZER_H
