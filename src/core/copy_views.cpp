#include "core/copy_views.hpp"

#include <algorithm>
#include <cstdio>

namespace ling {
namespace {

constexpr size_t kMaxViewBytes = 512 * 1024;  // a tool output's text past this is not viewed (tokenizing costs time)
constexpr size_t kMaxCallBytes = 4096;        // the tool call kept per output (edit mode's path lookup)
constexpr int kHeaderTokens = 64;             // how far back a just-written hunk header line is looked for
constexpr int kPatchTokens = 4096;            // how far back the patch's file header is looked for

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

bool read_int(std::string_view s, size_t& i, int* v) {
  const size_t b = i;
  long x = 0;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9' && i - b < 9) x = x * 10 + (s[i++] - '0');
  *v = static_cast<int>(x);
  return i > b;
}

// The path a patch names last before its hunk: `+++ b/P`, `--- a/P` or apply_patch's `*** Update File: P`.
std::string patch_path(std::string_view text) {
  struct Form {
    const char* head;
    const char* strip;
  };
  const Form forms[] = {{"\n+++ ", "b/"}, {"\n--- ", "a/"}, {"*** Update File: ", ""}};
  size_t best_at = std::string_view::npos;
  std::string best;
  for (const Form& f : forms) {
    const size_t at = text.rfind(f.head);
    if (at == std::string_view::npos || (best_at != std::string_view::npos && at < best_at)) continue;
    size_t b = at + std::char_traits<char>::length(f.head), e = text.find('\n', b);
    std::string_view p = trim(text.substr(b, e == std::string_view::npos ? std::string_view::npos : e - b));
    if (p.substr(0, std::char_traits<char>::length(f.strip)) == f.strip) p.remove_prefix(std::char_traits<char>::length(f.strip));
    const size_t tab = p.find('\t');  // unified headers may carry a timestamp after a tab
    if (tab != std::string_view::npos) p = p.substr(0, tab);
    if (p.empty() || p == "/dev/null") continue;
    best_at = at;
    best = std::string(p);
  }
  return best;
}

}  // namespace

std::string diff_view(std::string_view text, char prefix) {
  std::string out;
  out.reserve(text.size() + text.size() / 16 + 1);
  bool line_start = true;
  for (char ch : text) {
    if (line_start) out.push_back(prefix);
    line_start = ch == '\n';
    out.push_back(ch);
  }
  return out;
}

std::string json_view(std::string_view text) {
  std::string out;
  out.reserve(text.size() + text.size() / 8);
  for (unsigned char ch : text) {
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (ch < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", ch);
          out += buf;
        } else {
          out.push_back(static_cast<char>(ch));
        }
    }
  }
  return out;
}

bool parse_hunk_header(std::string_view line, HunkHeader* h) {
  while (!line.empty() && line.back() == '\r') line.remove_suffix(1);
  if (line.substr(0, 2) != "@@") return false;
  *h = HunkHeader{};
  // Unified: @@ -L[,n] +L[,m] @@[ heading]
  size_t i = 2;
  while (i < line.size() && line[i] == ' ') ++i;
  int n = 0;
  if (i < line.size() && line[i] == '-' && read_int(line, ++i, &h->old_line)) {
    if (i < line.size() && line[i] == ',') read_int(line, ++i, &n);
    while (i < line.size() && line[i] == ' ') ++i;
    if (i < line.size() && line[i] == '+' && read_int(line, ++i, &n)) {
      if (i < line.size() && line[i] == ',') read_int(line, ++i, &n);
      while (i < line.size() && line[i] == ' ') ++i;
      if (line.substr(i, 2) == "@@") {
        h->unified = true;
        h->anchor = std::string(trim(line.substr(i + 2)));
        return true;
      }
    }
  }
  // apply_patch: @@ <a line of the file> (or a bare @@)
  h->old_line = 0;
  h->anchor = std::string(trim(line.substr(2)));
  return true;
}

void CopySession::clear() {
  s1_.clear();
  side_.clear();
  spans_.clear();
  scanned_ = 0;
}

std::string CopySession::bytes(int from, int to, const TextCodec& codec) const {
  std::string out;
  const std::vector<int>& h = s1_.tokens();
  for (int k = std::max(0, from); k < to && k < static_cast<int>(h.size()); ++k) out += codec.token_bytes(h[k]);
  return out;
}

void CopySession::build_span(int open, int close, const CopyConfig& cfg, const TextCodec& codec) {
  Span sp;
  sp.open = open;
  sp.close = close;
  sp.corpus0 = static_cast<int>(side_.size());
  std::string text = bytes(open + 1, close, codec);
  // The template puts the output on its own lines: <tool_response>\nTEXT\n</tool_response>.
  if (!text.empty() && text.front() == '\n') text.erase(0, 1);
  if (!text.empty() && text.back() == '\n') text.pop_back();
  if (text.size() > kMaxViewBytes) text.resize(kMaxViewBytes);
  const bool space = (cfg.views & kCopyViewDiff) || cfg.edit;
  auto add = [&](const std::string& view, bool record_lines) {
    const std::vector<int> toks = codec.encode(view);
    if (record_lines) {
      // Each line's first token in side_, when the line starts on a token boundary (edit mode's targets).
      std::vector<size_t> starts;
      starts.reserve(toks.size());
      size_t off = 0;
      for (int t : toks) {
        starts.push_back(off);
        off += codec.token_bytes(t).size();
      }
      size_t line0 = 0;
      while (line0 < view.size()) {
        const auto it = std::lower_bound(starts.begin(), starts.end(), line0);
        sp.line_tok.push_back(it != starts.end() && *it == line0 ? sp.corpus0 + static_cast<int>(it - starts.begin()) : -1);
        const size_t nl = view.find('\n', line0);
        if (nl == std::string::npos) break;
        line0 = nl + 1;
      }
    }
    side_.append(toks);
    const int sep = -1;
    side_.append(std::span<const int>(&sep, 1));
    stats_.view_tokens += static_cast<long>(toks.size());
  };
  if (space) add(diff_view(text, ' '), cfg.edit);  // first: its tokens start at corpus0
  if (cfg.views & kCopyViewDiff) {
    add(diff_view(text, '-'), false);
    add(diff_view(text, '+'), false);
  }
  if (cfg.views & kCopyViewJson) add(json_view(text), false);
  if (cfg.edit) {
    size_t b = 0;
    while (b <= text.size()) {
      const size_t e = text.find('\n', b);
      sp.lines.emplace_back(text.substr(b, e == std::string::npos ? std::string::npos : e - b));
      if (e == std::string::npos) break;
      b = e + 1;
    }
    sp.line_tok.resize(sp.lines.size(), -1);
    // The tool call that produced this output: the last <tool_call> .. </tool_call> since the previous output.
    if (codec.tool_call_open >= 0 && codec.tool_call_close >= 0) {
      const std::vector<int>& h = s1_.tokens();
      const int floor = spans_.empty() ? 0 : spans_.back().close + 1;
      int c1 = -1;
      for (int k = open - 1; k >= floor; --k)
        if (h[k] == codec.tool_call_close) {
          c1 = k;
          break;
        }
      if (c1 >= 0) {
        int c0 = c1 - 1;
        while (c0 >= floor && h[c0] != codec.tool_call_open) --c0;
        if (c0 >= floor) {
          sp.call = bytes(c0 + 1, c1, codec);
          if (sp.call.size() > kMaxCallBytes) sp.call.resize(kMaxCallBytes);
        }
      }
    }
  }
  ++stats_.spans;
  spans_.push_back(std::move(sp));
}

void CopySession::sync_views(size_t kept, const CopyConfig& cfg, const TextCodec& codec) {
  // Outputs whose closing marker is past the kept prefix go, newest first, with their views.
  bool dropped = false;
  while (!spans_.empty() && spans_.back().close >= static_cast<int>(kept)) {
    side_.truncate(spans_.back().corpus0);
    spans_.pop_back();
    dropped = true;
  }
  if (dropped) scanned_ = spans_.empty() ? 0 : spans_.back().close + 1;
  scanned_ = std::min(scanned_, static_cast<int>(kept));
  const std::vector<int>& h = s1_.tokens();
  const int L = static_cast<int>(h.size());
  int i = scanned_;
  while (i < L) {
    if (h[i] != codec.tool_response_open) {
      ++i;
      continue;
    }
    int j = i + 1;
    while (j < L && h[j] != codec.tool_response_close) ++j;
    if (j >= L) break;  // not closed yet: looked at again next time
    build_span(i, j, cfg, codec);
    i = j + 1;
  }
  scanned_ = i;
}

CopyProposal CopySession::edit_jump(const CopyConfig& cfg, const TextCodec& codec) {
  CopyProposal r;
  const std::vector<int>& h = s1_.tokens();
  const int L = static_cast<int>(h.size());
  if (L == 0 || spans_.empty()) return r;
  const std::string last = codec.token_bytes(h[L - 1]);
  if (last.empty() || last.back() != '\n') return r;  // a header line has just ended only with a newline
  // The line that has just ended.
  std::string tail = last;
  int k = L - 1;
  while (k > 0 && L - k < kHeaderTokens && std::count(tail.begin(), tail.end(), '\n') < 2) tail.insert(0, codec.token_bytes(h[--k]));
  if (tail.size() < 2) return r;
  const size_t end = tail.size() - 1;  // the final '\n'
  const size_t prev = tail.rfind('\n', end - 1);
  if (prev == std::string::npos && k > 0) return r;  // the line starts further back than was looked at
  const size_t b = prev == std::string::npos ? 0 : prev + 1;
  const std::string_view line = std::string_view(tail).substr(b, end - b);
  HunkHeader hh;
  if (!parse_hunk_header(line, &hh)) return r;
  int span = -1, at_line = -1;
  if (hh.unified && hh.old_line >= 1) {
    const std::string path = patch_path(bytes(L - kPatchTokens, L, codec));
    if (!path.empty()) {
      const std::string base = path.substr(path.rfind('/') == std::string::npos ? 0 : path.rfind('/') + 1);
      for (int s = static_cast<int>(spans_.size()) - 1; s >= 0 && span < 0; --s) {
        const Span& sp = spans_[s];
        if (sp.call.find(path) == std::string::npos && (base.empty() || sp.call.find(base) == std::string::npos)) continue;
        if (hh.old_line <= static_cast<int>(sp.lines.size())) {
          span = s;
          at_line = hh.old_line - 1;
        }
      }
    }
  }
  if (span < 0 && !hh.anchor.empty()) {  // the anchor line, in the newest output that has it: copy from the line after
    const std::string_view anchor = trim(hh.anchor);
    for (int s = static_cast<int>(spans_.size()) - 1; s >= 0 && span < 0; --s) {
      const Span& sp = spans_[s];
      for (size_t i = 0; i < sp.lines.size(); ++i)
        if (trim(sp.lines[i]) == anchor) {
          span = s;
          at_line = static_cast<int>(i) + 1;
          break;
        }
    }
  }
  if (span < 0 || at_line >= static_cast<int>(spans_[span].lines.size()) || spans_[span].line_tok[at_line] < 0) {
    ++stats_.edit_misses;
    return r;
  }
  const int at = spans_[span].line_tok[at_line];
  const std::vector<int>& c = side_.tokens();
  for (int i = at; i < static_cast<int>(c.size()) && c[i] >= 0 && static_cast<int>(r.tokens.size()) < cfg.params.max_tokens; ++i)
    r.tokens.push_back(c[i]);
  if (r.tokens.empty()) return r;
  r.source = at;
  r.from = CopySource::Edit;
  ++stats_.edit_anchors;
  return r;
}

CopyProposal CopySession::propose(std::span<const int> history, std::span<const int> tail, const CopyConfig& cfg,
                                  const TextCodec* codec) {
  if (cfg.ngram != built_ngram_ || cfg.views != built_views_ || cfg.edit != built_edit_) {
    clear();
    s1_.configure(cfg.ngram);
    side_.configure(cfg.ngram);
    built_ngram_ = cfg.ngram;
    built_views_ = cfg.views;
    built_edit_ = cfg.edit;
  }
  const size_t kept = s1_.sync(history, tail);
  CopyProposal best = s1_.propose(cfg.params);
  const bool views = codec && codec->valid() && (cfg.views || cfg.edit);
  if (!views) {  // nothing kept that a later call could take for current
    if (side_.size() > 0 || !spans_.empty()) {
      side_.clear();
      spans_.clear();
    }
    scanned_ = 0;
    return best;
  }
  sync_views(kept, cfg, *codec);
  if (cfg.edit) {
    CopyProposal e = edit_jump(cfg, *codec);
    if (!e.tokens.empty()) return e;
  }
  if (cfg.views && side_.size() > 0) {  // edit mode alone builds the ' ' view only for its jumps
    CopyProposal v = side_.propose(s1_.tokens(), cfg.params);
    v.from = CopySource::View;
    if (!v.tokens.empty() && (best.tokens.empty() || v.match > best.match)) best = std::move(v);
  }
  return best;
}

}  // namespace ling
