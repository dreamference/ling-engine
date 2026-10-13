// The copy proposer's session state (moonshot P1, specs/drafts/MOONSHOT-repository-aware-decoding-2026-10-13.md §8):
// S1, the session's own tokens (copy_proposer.hpp), and, behind their flags,
//   - S2, emission views: every tool output of the session (the text between <tool_response> and </tool_response>)
//     re-tokenized in the forms the agent writes it back in: each line prefixed with ' ', '-' or '+' (a unified diff
//     or apply_patch's context, removed and added lines), and JSON-escaped (a string argument), each tokenized as a
//     whole span so token boundaries are the ones the model emits;
//   - edit mode: when the output has just written a hunk header, the proposal jumps to the place in the file the
//     header names (apply_patch's `@@ <line of the file>`: the line after it; a unified `@@ -L,n +L,m @@`: line L of
//     the file the patch's header names, read in an earlier tool output) and copies the file's ' '-prefixed lines
//     from there.
// Like the history's index, the views are a function of the history alone (the tool outputs in it, in order), so
// proposals do not depend on how the history was reached. Nothing here changes what a verify accepts.
#pragma once

#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/copy_proposer.hpp"

namespace ling {

// The tokenizer's side, which the engine core does not have: ling-serve and ling-run fill it from their tokenizer.
struct TextCodec {
  std::function<std::string(int)> token_bytes;                     // the bytes a token stands for
  std::function<std::vector<int>(const std::string&)> encode;       // text -> tokens, as the server tokenizes
  int tool_response_open = -1, tool_response_close = -1;           // <tool_response>, </tool_response>
  int tool_call_open = -1, tool_call_close = -1;                   // <tool_call>, </tool_call>
  bool valid() const { return token_bytes && encode && tool_response_open >= 0 && tool_response_close >= 0; }
};

// The views (EngineOptions::copy_views, a mask).
constexpr unsigned kCopyViewDiff = 1;  // ' ', '-' and '+' line prefixes
constexpr unsigned kCopyViewJson = 2;  // JSON string escaping

// Text transforms. diff_view prefixes every line (the text after each '\n', and the first) with `prefix`; a final
// '\n' gets no line after it. json_view escapes as a JSON string's contents (no quotes): \" \\ \n \r \t \b \f and
// \u00XX for the other control characters; everything else, UTF-8 included, as is.
std::string diff_view(std::string_view text, char prefix);
std::string json_view(std::string_view text);

// A hunk header line (without its '\n'): apply_patch's `@@ <anchor>` or a unified `@@ -L[,n] +L[,m] @@[ heading]`.
struct HunkHeader {
  bool unified = false;
  int old_line = 0;     // unified: L (1-based)
  std::string anchor;   // apply_patch: the line after `@@ `; unified: the heading after the second `@@` (may be empty)
};
bool parse_hunk_header(std::string_view line, HunkHeader* h);

struct CopyConfig {
  int ngram = 8;
  unsigned views = 0;  // kCopyViewDiff | kCopyViewJson
  bool edit = false;
  CopyParams params;
};

// Counters of the views' work, for tests and the stats.
struct CopyViewStats {
  long spans = 0;          // tool outputs turned into views
  long view_tokens = 0;    // tokens of views built
  long edit_anchors = 0;   // hunk headers that found their place
  long edit_misses = 0;    // hunk headers that did not
};

class CopySession {
 public:
  // Brings the state to `history` followed by `tail` (the step's anchor) and returns the best proposal: edit mode's
  // jump right after a hunk header, else the longer agreement of the history's and the views' (the history's on a
  // tie). Views and edit mode need `codec` (null: the history only). With edit mode alone (no views), the ' ' view it
  // builds serves its jumps only.
  CopyProposal propose(std::span<const int> history, std::span<const int> tail, const CopyConfig& cfg,
                       const TextCodec* codec);
  void clear();
  const CopyIndex& history_index() const { return s1_; }
  const CopyIndex& view_index() const { return side_; }
  const CopyViewStats& stats() const { return stats_; }

 private:
  struct Span {
    int open = 0, close = 0;  // the markers' positions in the history
    int corpus0 = 0;          // its first view token in side_
    std::string call;         // the tool call before it (edit mode: which file it read)
    std::vector<std::string> lines;  // the output's lines (edit mode)
    std::vector<int> line_tok;       // edit mode: side_ position where line i of the ' ' view starts, -1 off a token boundary
  };
  void sync_views(size_t kept, const CopyConfig& cfg, const TextCodec& codec);
  void build_span(int open, int close, const CopyConfig& cfg, const TextCodec& codec);
  CopyProposal edit_jump(const CopyConfig& cfg, const TextCodec& codec);
  std::string bytes(int from, int to, const TextCodec& codec) const;  // the history's text [from, to)

  CopyIndex s1_, side_{8};
  std::vector<Span> spans_;
  int scanned_ = 0;       // history positions before this hold no tool output not in spans_
  unsigned built_views_ = 0;
  bool built_edit_ = false;
  int built_ngram_ = -1;
  CopyViewStats stats_;
};

}  // namespace ling
