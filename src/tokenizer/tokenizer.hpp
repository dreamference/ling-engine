// Qwen's tokenizer in C++: byte-level BPE, NFC normalization, the Split pre-tokenizer pattern from
// tokenizer.json, and added tokens matched before anything else (as the reference tokenizer does).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ling {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json_path);
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  std::vector<int> encode(std::string_view text) const;
  // The bytes a token stands for (special tokens decode to their text).
  std::string token_bytes(int id) const;
  std::string decode(const std::vector<int>& ids) const;
  int token_id(const std::string& token) const;  // -1 if absent
  bool is_special(int id) const;
  int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

 private:
  void encode_segment(std::string_view text, std::vector<int>& out) const;
  void bpe(const std::string& piece, std::vector<int>& out) const;

  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_rank_;  // "a b" -> rank
  std::vector<std::pair<std::string, int>> added_;    // sorted longest first
  std::vector<bool> special_;
  std::string byte_to_unicode_[256];
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;
  void* regex_ = nullptr;  // pcre2_code*
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

// Turns tokens into text as they are generated, holding back bytes until they form whole UTF-8
// characters.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& t) : tok_(t) {}
  std::string push(int id);
  std::string flush();

 private:
  const Tokenizer& tok_;
  std::string pending_;
};

}  // namespace ling
