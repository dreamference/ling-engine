#include "tokenizer/tokenizer.hpp"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <climits>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ling {
namespace {

std::string utf8_of(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else if (cp < 0x800) {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return s;
}

// Splits a UTF-8 string into its characters.
std::vector<std::string> utf8_chars(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
    n = std::min(n, s.size() - i);
    out.emplace_back(s.substr(i, n));
    i += n;
  }
  return out;
}

std::string nfc(std::string_view text) {
  UErrorCode err = U_ZERO_ERROR;
  const icu::Normalizer2* n = icu::Normalizer2::getNFCInstance(err);
  if (U_FAILURE(err)) throw std::runtime_error("ICU NFC unavailable");
  icu::UnicodeString u = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
  if (n->isNormalized(u, err) && U_SUCCESS(err)) return std::string(text);
  err = U_ZERO_ERROR;
  icu::UnicodeString out = n->normalize(u, err);
  if (U_FAILURE(err)) throw std::runtime_error("NFC normalization failed");
  std::string s;
  out.toUTF8String(s);
  return s;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  nlohmann::json j = nlohmann::json::parse(f);
  const auto& model = j["model"];
  if (model["type"] != "BPE") throw std::runtime_error("only BPE tokenizers are supported");
  int max_id = 0;
  for (auto& [tok, id] : model["vocab"].items()) {
    vocab_[tok] = id.get<int>();
    max_id = std::max(max_id, id.get<int>());
  }
  for (const auto& a : j["added_tokens"]) max_id = std::max(max_id, a["id"].get<int>());
  id_to_token_.resize(max_id + 1);
  special_.assign(max_id + 1, false);
  for (auto& [tok, id] : vocab_) id_to_token_[id] = tok;
  int rank = 0;
  for (const auto& m : model["merges"]) {
    std::string key = m.is_string() ? m.get<std::string>() : m[0].get<std::string>() + " " + m[1].get<std::string>();
    merge_rank_.emplace(std::move(key), rank++);
  }
  for (const auto& a : j["added_tokens"]) {
    const std::string content = a["content"];
    const int id = a["id"];
    added_.emplace_back(content, id);
    id_to_token_[id] = content;
    vocab_[content] = id;
    special_[id] = a.value("special", false);
  }
  std::sort(added_.begin(), added_.end(), [](auto& x, auto& y) { return x.first.size() > y.first.size(); });

  // GPT-2's byte-to-unicode table.
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  std::vector<bool> direct(256, false);
  for (int b : bs) direct[b] = true;
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    const uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : 256 + extra++;
    byte_to_unicode_[b] = utf8_of(cp);
    unicode_to_byte_[byte_to_unicode_[b]] = static_cast<uint8_t>(b);
  }

  // The Split pre-tokenizer's pattern.
  std::string pattern;
  const auto& pre = j["pre_tokenizer"];
  if (pre["type"] == "Sequence") {
    for (const auto& p : pre["pretokenizers"])
      if (p["type"] == "Split") pattern = p["pattern"]["Regex"];
  } else if (pre["type"] == "Split") {
    pattern = pre["pattern"]["Regex"];
  }
  if (pattern.empty()) throw std::runtime_error("no Split pattern in tokenizer.json");
  int errcode = 0;
  PCRE2_SIZE erroff = 0;
  pcre2_code* re = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.c_str()), pattern.size(),
                                 PCRE2_UTF | PCRE2_UCP, &errcode, &erroff, nullptr);
  if (!re) throw std::runtime_error("cannot compile the pre-tokenizer pattern");
  pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
  regex_ = re;
}

Tokenizer::~Tokenizer() {
  if (regex_) pcre2_code_free(static_cast<pcre2_code*>(regex_));
}

int Tokenizer::token_id(const std::string& token) const {
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

bool Tokenizer::is_special(int id) const { return id >= 0 && id < static_cast<int>(special_.size()) && special_[id]; }

std::vector<int> Tokenizer::encode(std::string_view text) const {
  std::vector<int> out;
  size_t start = 0, i = 0;
  while (i < text.size()) {
    bool matched = false;
    if (text[i] == '<') {
      for (const auto& [content, id] : added_) {
        if (text.compare(i, content.size(), content) == 0) {
          if (i > start) encode_segment(text.substr(start, i - start), out);
          out.push_back(id);
          i += content.size();
          start = i;
          matched = true;
          break;
        }
      }
    }
    if (!matched) ++i;
  }
  if (start < text.size()) encode_segment(text.substr(start), out);
  return out;
}

void Tokenizer::encode_segment(std::string_view raw, std::vector<int>& out) const {
  const std::string text = nfc(raw);
  pcre2_code* re = static_cast<pcre2_code*>(regex_);
  pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);
  size_t offset = 0, last = 0;
  auto emit = [&](size_t a, size_t b) {
    if (b <= a) return;
    std::string piece;
    for (size_t k = a; k < b; ++k) piece += byte_to_unicode_[static_cast<unsigned char>(text[k])];
    bpe(piece, out);
  };
  while (offset <= text.size()) {
    int rc = pcre2_match(re, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset, 0, md, nullptr);
    if (rc < 0) break;
    PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
    size_t s = ov[0], e = ov[1];
    if (e == s) {  // empty match: step one character
      offset = s + 1;
      while (offset < text.size() && (static_cast<unsigned char>(text[offset]) & 0xC0) == 0x80) ++offset;
      continue;
    }
    emit(last, s);  // a gap between matches is its own piece ("Isolated")
    emit(s, e);
    last = e;
    offset = e;
  }
  emit(last, text.size());
  pcre2_match_data_free(md);
}

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
  if (auto it = cache_.find(piece); it != cache_.end()) {
    out.insert(out.end(), it->second.begin(), it->second.end());
    return;
  }
  std::vector<int> ids;
  if (auto it = vocab_.find(piece); it != vocab_.end()) {
    ids.push_back(it->second);
  } else {
    std::vector<std::string> sym = utf8_chars(piece);
    while (sym.size() > 1) {
      int best = INT_MAX;
      size_t at = 0;
      for (size_t k = 0; k + 1 < sym.size(); ++k) {
        auto it = merge_rank_.find(sym[k] + " " + sym[k + 1]);
        if (it != merge_rank_.end() && it->second < best) {
          best = it->second;
          at = k;
        }
      }
      if (best == INT_MAX) break;
      sym[at] += sym[at + 1];
      sym.erase(sym.begin() + at + 1);
    }
    for (const std::string& s : sym) {
      auto it = vocab_.find(s);
      if (it == vocab_.end()) throw std::runtime_error("BPE produced a symbol outside the vocabulary");
      ids.push_back(it->second);
    }
  }
  if (cache_.size() < 200000) cache_.emplace(piece, ids);
  out.insert(out.end(), ids.begin(), ids.end());
}

std::string Tokenizer::token_bytes(int id) const {
  if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return {};
  const std::string& tok = id_to_token_[id];
  for (const auto& [content, aid] : added_)
    if (aid == id) return content;
  std::string bytes;
  for (const std::string& ch : utf8_chars(tok)) {
    auto it = unicode_to_byte_.find(ch);
    if (it != unicode_to_byte_.end()) bytes += static_cast<char>(it->second);
    else bytes += ch;
  }
  return bytes;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string s;
  for (int id : ids) s += token_bytes(id);
  return s;
}

std::string StreamDecoder::push(int id) {
  pending_ += tok_.token_bytes(id);
  // Emit the longest prefix of whole UTF-8 characters.
  size_t end = pending_.size(), i = pending_.size();
  int back = 0;
  while (i > 0 && back < 4) {
    --i;
    ++back;
    unsigned char c = static_cast<unsigned char>(pending_[i]);
    if ((c & 0xC0) != 0x80) {
      size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
      if (pending_.size() - i < need) end = i;
      break;
    }
  }
  std::string out = pending_.substr(0, end);
  pending_.erase(0, end);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out;
  out.swap(pending_);
  return out;
}

}  // namespace ling
