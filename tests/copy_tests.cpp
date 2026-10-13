// The long-copy proposer's checks (moonshot P1), on the host with no GPU:
//   - proposals: the continuation of the earlier occurrence that agrees furthest back, the gates and caps;
//   - the index is a function of the corpus alone: built in random pieces, truncated and regrown, synced from
//     two spans, it proposes exactly what an index built in one go proposes (the GPU exactness checks compare
//     steps across prefill splits and resumes, which needs this);
//   - extend mode's chain (extend_with_copy);
//   - greedy exactness end to end: a toy target whose greedy output is a fixed text full of edited copies is
//     decoded with the engine's step logic (index synced to history + anchor, replace or extend chain, the
//     greedy accept) and must give the text token for token, in fewer steps than the drafter alone;
//   - sampled exactness: chains that mix the drafter's sampled tokens with copied ones (extend) or are all copied
//     (replace) leave the joint distribution of the output the target's (chi-square, as tests/spec_tests.cpp);
//   - cost: sync and proposal per step on a 64K-token history, and a turn's truncate-and-regrow.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "core/accept.hpp"
#include "core/copy_proposer.hpp"
#include "core/copy_views.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

using ling::CopyIndex;
using ling::CopyParams;
using ling::CopyProposal;

bool report(const char* what, bool ok) {
  std::printf("%s %s\n", what, ok ? "ok" : "FAIL");
  return ok;
}

CopyProposal propose_seq(const std::vector<int>& seq, int ngram, CopyParams p) {
  CopyIndex idx(ngram);
  idx.append(seq);
  return idx.propose(p);
}

bool check_proposals() {
  bool ok = true;
  CopyParams p;
  p.min_match = 3;
  p.max_tokens = 4;
  // "a b c d e f a b c": the last "a b c" occurred at 0; propose "d e f a", match 3.
  auto r = propose_seq({1, 2, 3, 4, 5, 6, 1, 2, 3}, 3, p);
  ok &= r.tokens == std::vector<int>({4, 5, 6, 1}) && r.match == 3 && r.source == 3;
  // Two occurrences agreeing equally far back (4 tokens): the most recent wins.
  p.max_tokens = 2;
  r = propose_seq({9, 1, 2, 3, 7, 0, 9, 1, 2, 3, 8, 9, 1, 2, 3}, 3, p);
  ok &= r.tokens == std::vector<int>({8, 9}) && r.match == 4 && r.candidates == 2;
  // The occurrence that agrees furthest back wins over a more recent one (the context lookup would take {6, 6}).
  const std::vector<int> longest = {7, 8, 1, 2, 3, 4, 5, 0, 1, 2, 3, 6, 6, 7, 8, 1, 2, 3};
  r = propose_seq(longest, 3, p);
  ok &= r.tokens == std::vector<int>({4, 5}) && r.match == 5;
  // Only the most recent candidate examined.
  p.max_candidates = 1;
  r = propose_seq(longest, 3, p);
  ok &= r.tokens == std::vector<int>({6, 6}) && r.match == 3;
  p.max_candidates = 16;
  // The gate: a 5-token agreement is not enough for min_match 6 (match still reported).
  p.min_match = 6;
  r = propose_seq(longest, 3, p);
  ok &= r.tokens.empty() && r.match == 5;
  p.min_match = 3;
  // The length rule: alpha * match, at least min_tokens.
  p.max_tokens = 10;
  p.alpha = 0.5;
  r = propose_seq(longest, 3, p);
  ok &= r.tokens == std::vector<int>({4, 5});  // 0.5 * 5 = 2
  p.alpha = 0.1;
  p.min_tokens = 3;
  r = propose_seq(longest, 3, p);
  ok &= r.tokens == std::vector<int>({4, 5, 0});
  p.alpha = 0;
  p.min_tokens = 1;
  // The proposal stops at the corpus's end; the query's own end is never an occurrence.
  r = propose_seq({5, 5, 5, 5}, 3, p);
  ok &= r.tokens == std::vector<int>({5}) && r.match == 3;
  // No earlier occurrence, or a corpus shorter than the n-gram: nothing.
  ok &= propose_seq({1, 2, 3, 4, 5}, 3, p).tokens.empty();
  ok &= propose_seq({1, 2}, 3, p).tokens.empty();
  // A separate query against a corpus (how a side corpus, S2 or S3, would be asked).
  CopyIndex idx(3);
  idx.append(std::vector<int>{10, 11, 12, 13, 14, 15});
  r = idx.propose(std::vector<int>{0, 10, 11, 12}, p);
  ok &= r.tokens == std::vector<int>({13, 14, 15}) && r.match == 3;
  // The agreement is measured up to match_cap.
  std::vector<int> rep(100, 0);
  for (int i = 0; i < 100; ++i) rep[i] = i % 10;
  p.match_cap = 20;
  r = propose_seq(rep, 3, p);
  ok &= r.match == 20 && r.tokens.size() == 10;
  // A negative token is a separator: the continuation stops before it (the views' side corpus).
  r = propose_seq({1, 2, 3, 4, 5, -1, 9, 1, 2, 3}, 3, p);
  ok &= r.tokens == std::vector<int>({4, 5});
  return report("copy proposals", ok);
}

// Every query's proposal, over several parameter sets, must be the same from both indexes.
bool same_proposals(const CopyIndex& a, const CopyIndex& b, std::mt19937_64& rng) {
  if (a.tokens() != b.tokens()) return false;
  std::vector<CopyParams> ps(3);
  ps[0].min_match = 1;
  ps[1].min_match = 4;
  ps[1].max_candidates = 2;
  ps[2].min_match = 2;
  ps[2].alpha = 1.5;
  ps[2].match_cap = 12;
  const auto& t = a.tokens();
  for (const CopyParams& p : ps) {
    for (int k = 0; k < 8; ++k) {
      // The corpus itself, and queries made of a stretch of it plus a random token or two.
      std::vector<int> q;
      if (k == 0) q = t;
      else if (!t.empty()) {
        const size_t end = rng() % t.size() + 1, len = std::min<size_t>(end, 1 + rng() % 40);
        q.assign(t.begin() + (end - len), t.begin() + end);
        if (k % 2) q.push_back(static_cast<int>(rng() % 5));
      }
      const CopyProposal x = a.propose(q, p), y = b.propose(q, p);
      if (x.tokens != y.tokens || x.match != y.match || x.source != y.source || x.candidates != y.candidates)
        return false;
    }
  }
  return true;
}

bool check_incremental() {
  std::mt19937_64 rng(7);
  bool ok = true;
  long undone = 0;
  for (int ngram : {1, 3, 8}) {
    CopyIndex inc(ngram);
    std::vector<int> seq;
    for (int op = 0; op < 400 && ok; ++op) {
      const int kind = static_cast<int>(rng() % 4);
      if (kind == 0 || seq.size() < 50) {  // append a piece: random tokens or a copy of an earlier stretch
        std::vector<int> piece;
        if (rng() % 2 && seq.size() > 20) {
          const size_t from = rng() % (seq.size() - 10), len = std::min<size_t>(seq.size() - from, 1 + rng() % 30);
          piece.assign(seq.begin() + from, seq.begin() + from + len);
        } else {
          for (size_t i = 0, n = 1 + rng() % 12; i < n; ++i) piece.push_back(static_cast<int>(rng() % 5));
        }
        seq.insert(seq.end(), piece.begin(), piece.end());
        inc.append(piece);
      } else if (kind == 1) {  // a turn re-rendered: cut back and regrow differently, through sync
        const size_t cut = seq.size() - std::min<size_t>(seq.size(), rng() % 60);
        seq.resize(cut);
        for (size_t i = 0, n = rng() % 50; i < n; ++i) seq.push_back(static_cast<int>(rng() % 5));
        inc.sync(seq);
      } else if (kind == 2) {  // a step: history and anchor as two spans
        const int anchor = static_cast<int>(rng() % 5);
        inc.sync(seq, std::span<const int>(&anchor, 1));
        seq.push_back(anchor);
      } else {  // truncate only
        const size_t cut = seq.size() - std::min<size_t>(seq.size(), rng() % 20);
        seq.resize(cut);
        inc.truncate(cut);
      }
      CopyIndex fresh(ngram);
      fresh.append(seq);
      ok &= same_proposals(inc, fresh, rng);
    }
    // A sync to an unrelated sequence, and to nothing.
    std::vector<int> other(300);
    for (int& x : other) x = static_cast<int>(rng() % 5);
    inc.sync(other);
    CopyIndex fresh(ngram);
    fresh.append(other);
    ok &= same_proposals(inc, fresh, rng);
    inc.sync({});
    ok &= inc.size() == 0 && inc.propose(other, CopyParams{}).tokens.empty();
    undone += inc.undone();
  }
  ok &= undone > 0;  // truncations took the undo path
  return report("copy index: incremental equals fresh (append, truncate, regrow, sync)", ok);
}

bool check_extend() {
  bool ok = true;
  std::vector<int> d = {1, 2, 3};
  ok &= ling::extend_with_copy(d, {1, 2, 3, 4, 5, 6}, 10) == 3 && d == std::vector<int>({1, 2, 3, 4, 5, 6});
  d = {1, 2, 3};
  ok &= ling::extend_with_copy(d, {1, 2, 3, 4, 5, 6}, 5) == 2 && d == std::vector<int>({1, 2, 3, 4, 5});
  d = {1, 9, 3};  // the drafter disagrees: unchanged
  ok &= ling::extend_with_copy(d, {1, 2, 3, 4, 5, 6}, 10) == 0 && d == std::vector<int>({1, 9, 3});
  d = {1, 2, 3};  // nothing beyond the drafter's chain, or no room
  ok &= ling::extend_with_copy(d, {1, 2, 3}, 10) == 0 && ling::extend_with_copy(d, {1, 2, 3, 4}, 3) == 0;
  return report("extend chain", ok);
}

// A text an agent might write: files read into the context, then rewritten with edits (insertions, deletions,
// replacements), plus prose between. Tokens from a large vocabulary so that chance matches are rare.
std::vector<int> agent_text(std::mt19937_64& rng, int* context_len) {
  auto fresh = [&](int n) {
    std::vector<int> v(n);
    for (int& x : v) x = 100 + static_cast<int>(rng() % 50000);
    return v;
  };
  std::vector<std::vector<int>> files;
  std::vector<int> text;
  for (int f = 0; f < 3; ++f) {
    files.push_back(fresh(800));
    auto prose = fresh(60);
    text.insert(text.end(), prose.begin(), prose.end());
    text.insert(text.end(), files.back().begin(), files.back().end());
  }
  *context_len = static_cast<int>(text.size());
  for (int f = 0; f < 3; ++f) {
    auto prose = fresh(40);
    text.insert(text.end(), prose.begin(), prose.end());
    const auto& src = files[f];
    size_t i = 0;
    while (i < src.size()) {
      const size_t run = std::min(src.size() - i, size_t(20 + rng() % 150));
      text.insert(text.end(), src.begin() + i, src.begin() + i + run);
      i += run;
      switch (rng() % 3) {
        case 0: {  // an insertion
          auto ins = fresh(1 + static_cast<int>(rng() % 30));
          text.insert(text.end(), ins.begin(), ins.end());
          break;
        }
        case 1:  // a deletion
          i += rng() % 20;
          break;
        default: {  // a replacement
          auto rep = fresh(1 + static_cast<int>(rng() % 10));
          text.insert(text.end(), rep.begin(), rep.end());
          i += 1 + rng() % 10;
        }
      }
    }
  }
  return text;
}

// Decodes the text after its context the way Engine::speculate does, with a toy target whose greedy choice at
// every position is the text's token there (any target gives the same argmax for a row whatever else is in
// the verify: the rows path is row-invariant), and a toy drafter that gets the next few tokens right.
// mode 0: drafter only; 1 replace; 2 extend. The copy proposals come from a CopySession with `cfg` (and `codec`
// for views and edit mode), synced to the history and the anchor as plan_begin syncs it.
struct DecodeStats {
  long steps = 0, copied_accepted = 0;
  long by_source[3] = {};  // copied tokens accepted, by CopySource
};

std::vector<int> decode_greedy(const std::vector<int>& text, int context_len, int mode, int block, const ling::CopyConfig& cfg,
                               const ling::TextCodec* codec, DecodeStats* st) {
  std::mt19937_64 rng(11);
  std::vector<int> history(text.begin(), text.begin() + context_len);
  ling::CopySession session;
  int anchor = text[context_len];  // chosen from the prefill's logits
  std::vector<int> out = {anchor};
  *st = DecodeStats{};
  const int L = static_cast<int>(text.size());
  while (static_cast<int>(history.size()) + 1 < L) {
    const int pos = static_cast<int>(history.size());  // the anchor's position
    const int E = block - 1;
    CopyProposal prop;
    if (mode > 0) prop = session.propose(history, std::span<const int>(&anchor, 1), cfg, codec);
    std::vector<int> drafts;
    int copy_from = INT32_MAX;
    if (mode == 1 && static_cast<int>(prop.tokens.size()) > E) {
      drafts = prop.tokens;
      copy_from = 0;
    } else {
      // The toy drafter: right for a geometric number of tokens (mean ~4.5), then wrong.
      int right = 0;
      while (right < E && rng() % 100 < 82) ++right;
      for (int e = 0; e < E; ++e) {
        const int at = pos + 1 + e;
        const int truth = at < L ? text[at] : 0;
        drafts.push_back(e < right ? truth : truth + 1);
      }
      if (mode == 2) {
        const int add = ling::extend_with_copy(drafts, prop.tokens, cfg.params.max_tokens);
        if (add > 0) copy_from = E;
      }
    }
    // The verify: row i's argmax is the text's token after position pos + i (past the end: an end token).
    const int V = static_cast<int>(drafts.size()) + 1;
    std::vector<int> argmax(V);
    for (int i = 0; i < V; ++i) argmax[i] = pos + 1 + i < L ? text[pos + 1 + i] : 0;
    const auto [accepted, next] = ling::accept_greedy(drafts, argmax.data());
    history.push_back(anchor);
    history.insert(history.end(), drafts.begin(), drafts.begin() + accepted);
    out.insert(out.end(), drafts.begin(), drafts.begin() + accepted);
    out.push_back(next);
    anchor = next;
    st->steps += 1;
    if (accepted > copy_from) {
      st->copied_accepted += accepted - copy_from;
      st->by_source[static_cast<int>(prop.from)] += accepted - copy_from;
    }
  }
  out.resize(std::min(out.size(), size_t(L - context_len)));
  return out;
}

ling::CopyConfig history_only() {
  ling::CopyConfig c;
  c.params.min_match = 16;
  c.params.max_tokens = 63;
  return c;
}

bool check_greedy_exact() {
  bool ok = true;
  for (uint64_t seed : {1, 2, 3}) {
    std::mt19937_64 rng(seed);
    int context_len = 0;
    const std::vector<int> text = agent_text(rng, &context_len);
    const std::vector<int> expect(text.begin() + context_len, text.end());
    DecodeStats st[3];
    for (int mode = 0; mode < 3; ++mode) {
      const std::vector<int> got = decode_greedy(text, context_len, mode, 12, history_only(), nullptr, &st[mode]);
      ok &= got == expect;
    }
    const double n = double(expect.size());
    std::printf("  seed %llu: %zu output tokens; tokens/step drafter %.2f, replace %.2f, extend %.2f "
                "(copied tokens accepted: replace %ld, extend %ld)\n",
                static_cast<unsigned long long>(seed), expect.size(), n / st[0].steps, n / st[1].steps, n / st[2].steps,
                st[1].copied_accepted, st[2].copied_accepted);
    ok &= st[1].steps < st[0].steps && st[2].steps < st[0].steps && st[1].copied_accepted > 0 && st[2].copied_accepted > 0;
  }
  return report("greedy: output token for token with copies (replace, extend, up to 63), fewer steps", ok);
}

// ---- Emission views and edit mode (S2, D2/D3 of 2026-10-13) ----

bool check_transforms() {
  bool ok = true;
  ok &= ling::diff_view("a\nb\n", ' ') == " a\n b\n";
  ok &= ling::diff_view("a\n\nb", '-') == "-a\n-\n-b";
  ok &= ling::diff_view("", '+').empty();
  ok &= ling::json_view("say \"hi\"\\\n\tx\x01") == "say \\\"hi\\\"\\\\\\n\\tx\\u0001";
  ok &= ling::json_view("caf\xc3\xa9") == "caf\xc3\xa9";  // UTF-8 as is
  ling::HunkHeader h;
  ok &= ling::parse_hunk_header("@@ -12,7 +12,8 @@ def foo():", &h) && h.unified && h.old_line == 12 && h.anchor == "def foo():";
  ok &= ling::parse_hunk_header("@@ -3 +3,2 @@", &h) && h.unified && h.old_line == 3 && h.anchor.empty();
  ok &= ling::parse_hunk_header("@@ class Foo:\r", &h) && !h.unified && h.anchor == "class Foo:";
  ok &= ling::parse_hunk_header("@@", &h) && !h.unified && h.anchor.empty();
  ok &= !ling::parse_hunk_header(" @@ x", &h) && !ling::parse_hunk_header("-@@", &h);
  ok &= ling::parse_hunk_header("@@ -x +1 @@", &h) && !h.unified && h.anchor == "-x +1 @@";  // not unified: an anchor
  return report("views: diff and JSON transforms, hunk headers", ok);
}

// A byte-level codec: a token per byte, and four marker tokens. Every line starts on a token boundary.
constexpr int kOpen = 300, kClose = 301, kCallOpen = 302, kCallClose = 303;
ling::TextCodec byte_codec() {
  ling::TextCodec c;
  c.token_bytes = [](int t) { return t < 256 ? std::string(1, static_cast<char>(t)) : std::string("<m>"); };
  c.encode = [](const std::string& s) {
    std::vector<int> v;
    for (unsigned char ch : s) v.push_back(ch);
    return v;
  };
  c.tool_response_open = kOpen;
  c.tool_response_close = kClose;
  c.tool_call_open = kCallOpen;
  c.tool_call_close = kCallClose;
  return c;
}

void put(std::vector<int>& v, const std::string& s) {
  for (unsigned char ch : s) v.push_back(ch);
}

// A Python-like file with `n` functions; line numbers are 1-based in unified headers.
std::string py_file(std::mt19937_64& rng, int n) {
  std::string f;
  for (int i = 1; i <= n; ++i) {
    f += "def f" + std::to_string(i) + "(x, scale=" + std::to_string(rng() % 90 + 10) + "):\n";
    f += "    y = x * scale + " + std::to_string(rng() % 1000) + "  # step " + std::to_string(i) + "\n";
    f += "    if y > " + std::to_string(rng() % 5000) + ":\n";
    f += "        return compute_result(y - " + std::to_string(rng() % 77) + ", label=\"f" + std::to_string(i) + "\")\n";
    f += "    return y\n\n";
  }
  return f;
}

std::vector<std::string> lines_of(const std::string& s) {
  std::vector<std::string> v;
  size_t b = 0;
  while (b <= s.size()) {
    const size_t e = s.find('\n', b);
    v.push_back(s.substr(b, e == std::string::npos ? std::string::npos : e - b));
    if (e == std::string::npos) break;
    b = e + 1;
  }
  return v;
}

// The session so far: a request, a tool call that reads `path`, its output (the file), then the start of an answer.
std::vector<int> session_with_file(const std::string& path, const std::string& file, const std::string& answer) {
  std::vector<int> h;
  put(h, "user: fix the scaling in " + path + "\nassistant: ");
  h.push_back(kCallOpen);
  put(h, "\n<function=exec_command>\n<parameter=cmd>\ncat " + path + "\n</parameter>\n</function>\n");
  h.push_back(kCallClose);
  put(h, "\nuser: ");
  h.push_back(kOpen);
  put(h, "\n" + file + "\n");
  h.push_back(kClose);
  put(h, "\nassistant: ");
  put(h, answer);
  return h;
}

std::vector<int> bytes_of(const std::string& s, size_t n) {
  std::vector<int> v;
  for (size_t i = 0; i < s.size() && i < n; ++i) v.push_back(static_cast<unsigned char>(s[i]));
  return v;
}

bool check_views_and_edit() {
  std::mt19937_64 rng(3);
  const std::string file = py_file(rng, 12);
  const std::vector<std::string> L = lines_of(file);
  const ling::TextCodec codec = byte_codec();
  ling::CopyConfig all = history_only();
  all.views = ling::kCopyViewDiff | ling::kCopyViewJson;
  all.edit = true;
  bool ok = true;
  auto ask = [&](const std::string& answer, const ling::CopyConfig& cfg, const ling::TextCodec* c) {
    std::vector<int> h = session_with_file("src/mod.py", file, answer);
    const int anchor = h.back();
    h.pop_back();
    ling::CopySession s;
    return s.propose(h, std::span<const int>(&anchor, 1), cfg, c);
  };
  // '-' view: removed lines quoted with their prefix. The history alone continues the raw line (no '-' after the
  // newline); the view continues with "\n-" and the next line.
  {
    const std::string answer = "*** Begin Patch\n*** Update File: src/mod.py\n-" + L[6] + "\n-" + L[7].substr(0, 20);
    const CopyProposal v = ask(answer, all, &codec), h = ask(answer, history_only(), nullptr);
    const std::string want = ling::diff_view(file, '-').substr(ling::diff_view(file, '-').find(L[7]) + 20);
    ok &= v.from == ling::CopySource::View && v.tokens == bytes_of(want, 63);
    ok &= h.tokens.empty() || h.tokens != v.tokens;
    if (!ok) std::printf("  '-' view failed\n");
  }
  // JSON view: the file as an escaped string argument.
  {
    const std::string js = ling::json_view(file);
    const size_t cut = js.find("f3(") + 40;
    const CopyProposal v = ask("{\"input\": \"" + js.substr(0, cut), all, &codec);
    ok &= v.from == ling::CopySource::View && v.tokens == bytes_of(js.substr(cut), 63);
    if (!(v.from == ling::CopySource::View)) std::printf("  json view failed\n");
  }
  // Edit mode, apply_patch's anchor: right after "@@ def f5(...):\n", the lines after that line, ' '-prefixed.
  const std::string sp = ling::diff_view(file, ' ');
  auto view_from_line = [&](size_t line) {
    size_t at = 0;
    for (size_t i = 0; i < line; ++i) at = sp.find('\n', at) + 1;
    return sp.substr(at);
  };
  size_t f5 = 0;
  while (L[f5].rfind("def f5(", 0) != 0) ++f5;
  {
    const CopyProposal e = ask("*** Begin Patch\n*** Update File: src/mod.py\n@@ " + L[f5] + "\n", all, &codec);
    ok &= e.from == ling::CopySource::Edit && e.tokens == bytes_of(view_from_line(f5 + 1), 63);
    if (!(e.from == ling::CopySource::Edit)) std::printf("  edit (apply_patch anchor) failed\n");
  }
  // Edit mode, unified: "@@ -L,n +L,m @@" jumps to line L of the file the patch's header names (read by `cat`).
  {
    const CopyProposal e = ask("--- a/src/mod.py\n+++ b/src/mod.py\n@@ -20,4 +20,5 @@\n", all, &codec);
    ok &= e.from == ling::CopySource::Edit && e.tokens == bytes_of(view_from_line(19), 63);
    // A file no tool output read, but a heading: the heading's line, then the line after it.
    const CopyProposal h = ask("--- a/other.py\n+++ b/other.py\n@@ -20,4 +20,5 @@ " + L[f5] + "\n", all, &codec);
    ok &= h.from == ling::CopySource::Edit && h.tokens == bytes_of(view_from_line(f5 + 1), 63);
    // Neither: no jump (the views and the history still propose as usual).
    const CopyProposal n = ask("--- a/other.py\n+++ b/other.py\n@@ -20,4 +20,5 @@\n", all, &codec);
    ok &= n.from != ling::CopySource::Edit;
    if (!(e.from == ling::CopySource::Edit && h.from == ling::CopySource::Edit)) std::printf("  edit (unified) failed\n");
  }
  // Off by default: without the flags (or without a codec) nothing comes from views or edit mode.
  {
    const CopyProposal e = ask("@@ " + L[f5] + "\n", history_only(), &codec);
    const CopyProposal n = ask("@@ " + L[f5] + "\n", all, nullptr);
    ok &= e.from == ling::CopySource::History && n.from == ling::CopySource::History;
  }
  return report("views ('-', JSON) and edit mode (apply_patch anchor, unified line, heading) propose the right text", ok);
}

// A random session over the byte codec: text, tool calls and outputs (files), and answers that quote them with
// prefixes, JSON escaping and hunk headers.
std::vector<int> random_session(std::mt19937_64& rng, int parts) {
  std::vector<std::string> files;
  std::vector<int> h;
  for (int p = 0; p < parts; ++p) {
    const int kind = static_cast<int>(rng() % 4);
    if (kind == 0 || files.empty()) {
      files.push_back(py_file(rng, 2 + static_cast<int>(rng() % 3)));
      const std::string path = "src/m" + std::to_string(files.size()) + ".py";
      const std::vector<int> s = session_with_file(path, files.back(), "");
      h.insert(h.end(), s.begin(), s.end());
    } else {
      const std::string& f = files[rng() % files.size()];
      const std::vector<std::string> L = lines_of(f);
      const size_t a = rng() % L.size();
      std::string out;
      if (kind == 1) out = "@@ " + L[a] + "\n" + ling::diff_view(f.substr(f.find(L[a])), ' ').substr(0, 200);
      else if (kind == 2) out = "@@ -" + std::to_string(a + 1) + ",3 +" + std::to_string(a + 1) + ",4 @@\n-" + L[a] + "\n+new\n";
      else out = "\"" + ling::json_view(f).substr(0, 150) + "\"";
      put(h, out);
    }
  }
  return h;
}

bool same_proposal(const CopyProposal& a, const CopyProposal& b) {
  return a.tokens == b.tokens && a.match == b.match && a.source == b.source && a.from == b.from;
}

bool check_session_incremental() {
  std::mt19937_64 rng(17);
  const ling::TextCodec codec = byte_codec();
  ling::CopyConfig cfg = history_only();
  cfg.params.min_match = 6;
  cfg.views = ling::kCopyViewDiff | ling::kCopyViewJson;
  cfg.edit = true;
  bool ok = true;
  ling::CopySession inc;
  std::vector<int> seq;
  for (int op = 0; op < 300 && ok; ++op) {
    const int kind = static_cast<int>(rng() % 3);
    if (kind == 0 || seq.size() < 100) {  // grow by a piece of a session
      const std::vector<int> more = random_session(rng, 1);
      seq.insert(seq.end(), more.begin(), more.end());
    } else if (kind == 1) {  // cut anywhere (inside a tool output too) and regrow differently
      seq.resize(seq.size() - std::min<size_t>(seq.size(), rng() % 400));
      const std::vector<int> more = random_session(rng, 1);
      seq.insert(seq.end(), more.begin(), more.end());
    } else {  // a few steps, token by token
      for (int i = 0; i < 20 && !seq.empty(); ++i) seq.push_back(static_cast<int>('a' + rng() % 3));
    }
    for (int q = 0; q < 3 && seq.size() > 2; ++q) {  // ask at the end and at a few earlier cut points
      const size_t at = q == 0 ? seq.size() - 1 : 1 + rng() % (seq.size() - 1);
      const std::span<const int> h(seq.data(), at);
      const int anchor = seq[at];
      const CopyProposal x = inc.propose(h, std::span<const int>(&anchor, 1), cfg, &codec);
      ling::CopySession fresh;
      const CopyProposal y = fresh.propose(h, std::span<const int>(&anchor, 1), cfg, &codec);
      ok &= same_proposal(x, y) && inc.view_index().tokens() == fresh.view_index().tokens();
    }
  }
  return report("session state (history, views, edit anchors): incremental equals fresh under cuts inside tool outputs", ok);
}

// An answer that rewrites parts of the files it read: apply_patch hunks with context, removed and added lines.
std::string patch_answer(std::mt19937_64& rng, const std::string& path, const std::string& file, int hunks) {
  const std::vector<std::string> L = lines_of(file);
  std::string out = "*** Begin Patch\n*** Update File: " + path + "\n";
  for (int k = 0; k < hunks; ++k) {
    size_t a = 0;
    do a = rng() % (L.size() - 12);
    while (L[a].rfind("def ", 0) != 0);
    out += "@@ " + L[a] + "\n";
    for (size_t i = a + 1; i < a + 3; ++i) out += " " + L[i] + "\n";
    out += "-" + L[a + 3] + "\n+" + L[a + 3] + "  # checked\n";
    for (size_t i = a + 4; i < a + 10; ++i) out += " " + L[i] + "\n";
  }
  return out + "*** End Patch\n";
}

bool check_greedy_views() {
  bool ok = true;
  const ling::TextCodec codec = byte_codec();
  for (uint64_t seed : {5, 6}) {
    std::mt19937_64 rng(seed);
    const std::string file = py_file(rng, 30);
    std::vector<int> text = session_with_file("src/mod.py", file, "");
    const int context_len = static_cast<int>(text.size());
    put(text, patch_answer(rng, "src/mod.py", file, 6));
    const std::vector<int> expect(text.begin() + context_len, text.end());
    ling::CopyConfig views = history_only();
    views.views = ling::kCopyViewDiff | ling::kCopyViewJson;
    ling::CopyConfig edit = views;
    edit.edit = true;
    ling::CopyConfig edit_only = history_only();
    edit_only.edit = true;
    DecodeStats h, v, eo, e, ex;
    ok &= decode_greedy(text, context_len, 1, 12, history_only(), &codec, &h) == expect;
    ok &= decode_greedy(text, context_len, 1, 12, views, &codec, &v) == expect;
    ok &= decode_greedy(text, context_len, 1, 12, edit_only, &codec, &eo) == expect;
    ok &= decode_greedy(text, context_len, 1, 12, edit, &codec, &e) == expect;
    ok &= decode_greedy(text, context_len, 2, 12, edit, &codec, &ex) == expect;
    const double n = double(expect.size());
    std::printf("  seed %llu: %zu tokens of patch; replace tokens/step: history %.2f, + views %.2f, + edit alone %.2f, "
                "+ views + edit %.2f; extend + views + edit %.2f (copied accepted by history/view/edit: edit alone "
                "%ld/%ld/%ld, views + edit %ld/%ld/%ld)\n",
                static_cast<unsigned long long>(seed), expect.size(), n / h.steps, n / v.steps, n / eo.steps, n / e.steps,
                n / ex.steps, eo.by_source[0], eo.by_source[1], eo.by_source[2], e.by_source[0], e.by_source[1],
                e.by_source[2]);
    ok &= v.steps < h.steps && v.by_source[1] > 0 && eo.steps < h.steps && eo.by_source[2] > 0;
  }
  return report("greedy: patches decoded token for token with views and edit mode, in fewer steps", ok);
}

// Sampled exactness of mixed chains: a toy target whose next-token distribution depends only on the position is
// decoded speculatively many times; the joint distribution of the first kLen tokens must be the target's.
constexpr int kVocab = 5, kLen = 4, kTrials = 400000;

ling::Dist random_dist(std::mt19937_64& rng, bool sparse) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  ling::Dist d;
  double sum = 0;
  for (int t = 0; t < kVocab; ++t) {
    if (sparse && u(rng) < 0.35) continue;
    const double w = u(rng) + 0.05;
    d.push_back({t, w});
    sum += w;
  }
  if (d.empty()) d.push_back({0, sum = 1.0});
  for (auto& [t, w] : d) w /= sum;
  return d;
}

double prob(const ling::Dist& d, int t) {
  for (const auto& [x, p] : d)
    if (x == t) return p;
  return 0;
}

// mode 1: replace (the whole chain copied, length C); mode 2: extend (Ed drafted tokens from q, then C - Ed
// copied tokens when the drafted ones equal the copy's start). The "copy" is a fixed token sequence per position,
// so a deterministic function of the prefix, as the session's index is.
bool check_sampled(int mode, int Ed, int C, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<ling::Dist> target(kLen + C + 1), draftq(kLen + C + 1);
  for (auto& d : target) d = random_dist(rng, true);
  // The drafter puts half its mass on the copy's token, so that extensions happen often.
  auto copy_tok = [&](int pos, int i) { return (pos * 3 + i * 2 + 1) % kVocab; };
  for (size_t j = 0; j < draftq.size(); ++j) draftq[j] = random_dist(rng, false);
  std::map<std::vector<int>, long> counts;
  long extended = 0, reached = 0;  // steps whose chain was extended; steps whose accept examined a copied token
  for (int trial = 0; trial < kTrials; ++trial) {
    std::vector<int> out;
    while (static_cast<int>(out.size()) < kLen) {
      const int pos = static_cast<int>(out.size());
      std::vector<int> copy(C);
      for (int i = 0; i < C; ++i) copy[i] = copy_tok(pos, i);
      std::vector<int> drafts;
      std::vector<ling::Dist> qd;
      int copy_from = 0;
      bool ext = mode == 1;
      if (mode == 1) {
        drafts = copy;
      } else {
        for (int i = 0; i < Ed; ++i) {
          // q: half on the copy's token, half the random distribution.
          ling::Dist q;
          for (int t = 0; t < kVocab; ++t) q.push_back({t, 0.5 * prob(draftq[pos + i], t) + (t == copy[i] ? 0.5 : 0)});
          drafts.push_back(ling::draw(q, rng));
          qd.push_back(q);
        }
        copy_from = Ed;
        ext = ling::extend_with_copy(drafts, copy, C) > 0;
        extended += ext;
      }
      const int E = static_cast<int>(drafts.size());
      std::vector<ling::Dist> P(target.begin() + pos, target.begin() + pos + E + 1);
      auto q = [&](int i, int t) { return i >= copy_from ? (t == drafts[i] ? 1.0 : 0.0) : prob(qd[i], t); };
      const auto [acc, next] = ling::accept_sampled(P, drafts, q, rng);
      reached += ext && acc >= copy_from && pos + copy_from < kLen;
      out.insert(out.end(), drafts.begin(), drafts.begin() + acc);
      out.push_back(next);
    }
    out.resize(kLen);
    counts[out]++;
  }
  double chi = 0;
  int cells = 0;
  std::vector<int> s(kLen, 0);
  bool impossible = false;
  for (int idx = 0; idx < std::pow(kVocab, kLen); ++idx) {
    int r = idx;
    double expected = kTrials;
    for (int j = 0; j < kLen; ++j) {
      s[j] = r % kVocab;
      r /= kVocab;
      expected *= prob(target[j], s[j]);
    }
    const long seen = counts.count(s) ? counts[s] : 0;
    if (expected == 0) {
      impossible |= seen != 0;
      continue;
    }
    chi += (seen - expected) * (seen - expected) / expected;
    ++cells;
  }
  const double k = cells - 1, z = 3.09;
  const double limit = k * std::pow(1 - 2 / (9 * k) + z * std::sqrt(2 / (9 * k)), 3);
  // The copied positions must really be examined inside the first kLen tokens, or the check proves nothing.
  const bool ok = !impossible && chi < limit && reached > kTrials / 20;
  std::printf("sampled (%s, drafted %d, chain %d): chi-square %.1f over %d cells (99.9%% limit %.1f), %ld steps "
              "extended, %ld examined a copied token %s\n",
              mode == 1 ? "replace" : "extend", Ed, C, chi, cells, limit, extended, reached, ok ? "ok" : "FAIL");
  return ok;
}


// With the real Qwen tokenizer (the path to tokenizer.json as the first argument; skipped without it): the views and
// edit mode must line up with how the model's own patch tokenizes, which a byte codec cannot show. A session renders
// as the chat template renders it (a tool call that reads a file, its output, then an apply_patch call with hunks),
// is tokenized once as a whole, and decoded greedily with each source.
bool check_real_tokenizer(const char* path) {
  ling::Tokenizer tok(path, ling::kProductionPretokenizer);
  ling::TextCodec codec;
  codec.token_bytes = [&tok](int id) { return tok.token_bytes(id); };
  codec.encode = [&tok](const std::string& s) { return tok.encode(s); };
  codec.tool_response_open = tok.token_id("<tool_response>");
  codec.tool_response_close = tok.token_id("</tool_response>");
  codec.tool_call_open = tok.token_id("<tool_call>");
  codec.tool_call_close = tok.token_id("</tool_call>");
  if (!codec.valid() || codec.tool_call_open < 0) return report("real tokenizer: markers found", false);
  bool ok = true;
  for (uint64_t seed : {7, 8}) {
    std::mt19937_64 rng(seed);
    const std::string file = py_file(rng, 40);
    const std::string prompt =
        "<|im_start|>user\nFix the scaling in src/mod.py.<|im_end|>\n<|im_start|>assistant\n<tool_call>\n"
        "<function=exec_command>\n<parameter=cmd>\ncat src/mod.py\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n" + file + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n";
    const std::string answer = "<tool_call>\n<function=apply_patch>\n<parameter=input>\n" +
                               patch_answer(rng, "src/mod.py", file, 6) + "</parameter>\n</function>\n</tool_call><|im_end|>";
    const std::vector<int> text = tok.encode(prompt + answer), head = tok.encode(prompt);
    if (!std::equal(head.begin(), head.end(), text.begin())) return report("real tokenizer: prompt tokens are a prefix", false);
    const int context_len = static_cast<int>(head.size());
    const std::vector<int> expect(text.begin() + context_len, text.end());
    ling::CopyConfig views = history_only();
    views.views = ling::kCopyViewDiff | ling::kCopyViewJson;
    ling::CopyConfig edit = views;
    edit.edit = true;
    DecodeStats d, h, v, e, ex;
    auto t0 = std::chrono::steady_clock::now();
    ok &= decode_greedy(text, context_len, 0, 12, history_only(), &codec, &d) == expect;
    ok &= decode_greedy(text, context_len, 1, 12, history_only(), &codec, &h) == expect;
    ok &= decode_greedy(text, context_len, 1, 12, views, &codec, &v) == expect;
    ok &= decode_greedy(text, context_len, 1, 12, edit, &codec, &e) == expect;
    ok &= decode_greedy(text, context_len, 2, 12, edit, &codec, &ex) == expect;
    // The cost of building the views once (four views of the file, tokenized).
    ling::CopySession s;
    const std::span<const int> hist(text.data(), context_len);
    const int anchor = text[context_len];
    auto t1 = std::chrono::steady_clock::now();
    s.propose(hist, std::span<const int>(&anchor, 1), edit, &codec);
    const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    (void)t0;
    const double n = double(expect.size());
    std::printf("  seed %llu: %zu tokens of output, file %zu bytes; tokens/step: drafter %.2f; replace: history %.2f, "
                "+ views %.2f, + edit %.2f; extend + edit %.2f; accepted copied by history/view/edit (replace + edit) "
                "%ld/%ld/%ld; views built in %.1f ms (%ld view tokens)\n",
                static_cast<unsigned long long>(seed), expect.size(), file.size(), n / d.steps, n / h.steps, n / v.steps,
                n / e.steps, n / ex.steps, e.by_source[0], e.by_source[1], e.by_source[2], build_ms, s.stats().view_tokens);
    ok &= v.by_source[1] > 0 && e.by_source[2] > 0 && v.steps < h.steps;
  }
  return report("real tokenizer: views and edit mode line up with the model's own patch tokens", ok);
}

bool check_cost() {
  // A 64K-token history: 2K-token files read in, then rewritten with edits, and steps of ~5 tokens.
  std::mt19937_64 rng(5);
  std::vector<int> seq;
  std::vector<std::vector<int>> files;
  while (seq.size() < 65536) {
    if (files.empty() || rng() % 2) {
      std::vector<int> f(2048);
      for (int& x : f) x = static_cast<int>(rng() % 151000);
      files.push_back(f);
      seq.insert(seq.end(), f.begin(), f.end());
    } else {
      const auto& f = files[rng() % files.size()];
      const size_t from = rng() % 1024;
      seq.insert(seq.end(), f.begin() + from, f.begin() + from + 512);
      for (int i = 0; i < 64; ++i) seq.push_back(static_cast<int>(rng() % 151000));
    }
  }
  seq.resize(65536);
  using clk = std::chrono::steady_clock;
  CopyIndex idx(8);
  auto t0 = clk::now();
  idx.sync(std::span<const int>(seq.data(), 60000));
  const double build_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  // Steps: sync to history + anchor, then propose.
  CopyParams p;
  std::vector<double> us;
  long proposals = 0;
  for (size_t at = 60000; at + 6 < seq.size(); at += 5) {
    const int anchor = seq[at];
    auto t = clk::now();
    idx.sync(std::span<const int>(seq.data(), at), std::span<const int>(&anchor, 1));
    const CopyProposal r = idx.propose(p);
    us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t).count());
    proposals += !r.tokens.empty();
  }
  std::sort(us.begin(), us.end());
  const double median = us[us.size() / 2], p99 = us[us.size() * 99 / 100];
  // A new turn: the last 400 tokens re-rendered differently.
  std::vector<int> turn(seq.begin(), seq.end() - 400);
  for (int i = 0; i < 400; ++i) turn.push_back(static_cast<int>(rng() % 151000));
  auto t1 = clk::now();
  idx.sync(turn);
  const double turn_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
  std::printf("  64K history: build %.1f ms; per step (sync + propose) median %.1f us, p99 %.1f us over %zu steps "
              "(%ld with a proposal); a turn's 400-token regrow %.2f ms\n",
              build_ms, median, p99, us.size(), proposals, turn_ms);
  // Generous bounds (a busy machine): the step's host work must stay far below a ~100 ms step.
  return report("copy index cost", median < 500 && turn_ms < 50 && proposals > 0);
}

}  // namespace

int main(int argc, char** argv) {
  bool ok = check_proposals();
  ok &= check_incremental();
  ok &= check_extend();
  ok &= check_greedy_exact();
  ok &= check_sampled(1, 0, 3, 21);
  ok &= check_sampled(1, 0, 6, 22);
  ok &= check_sampled(2, 1, 3, 23);
  ok &= check_sampled(2, 2, 5, 27);
  ok &= check_transforms();
  ok &= check_views_and_edit();
  ok &= check_session_incremental();
  ok &= check_greedy_views();
  ok &= check_cost();
  if (argc > 1) ok &= check_real_tokenizer(argv[1]);
  else std::printf("real tokenizer: skipped (pass tokenizer.json's path)\n");
  std::printf(ok ? "all copy tests passed\n" : "COPY TESTS FAILED\n");
  return ok ? 0 : 1;
}
