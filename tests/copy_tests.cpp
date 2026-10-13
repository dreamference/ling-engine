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
// mode 0: drafter only; 1 replace; 2 extend. Returns the tokens decoded; *steps the steps taken.
std::vector<int> decode_greedy(const std::vector<int>& text, int context_len, int mode, int block, long* steps,
                               long* copied_accepted) {
  std::mt19937_64 rng(11);
  std::vector<int> history(text.begin(), text.begin() + context_len);
  CopyIndex idx(8);
  CopyParams cp;
  cp.min_match = 16;
  cp.max_tokens = 31;
  int anchor = text[context_len];  // chosen from the prefill's logits
  std::vector<int> out = {anchor};
  *steps = 0;
  *copied_accepted = 0;
  const int L = static_cast<int>(text.size());
  while (static_cast<int>(history.size()) + 1 < L) {
    const int pos = static_cast<int>(history.size());  // the anchor's position
    const int E = block - 1;
    CopyProposal prop;
    if (mode > 0) {
      idx.sync(history, std::span<const int>(&anchor, 1));
      prop = idx.propose(cp);
    }
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
        const int add = ling::extend_with_copy(drafts, prop.tokens, cp.max_tokens);
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
    *steps += 1;
    if (accepted > copy_from) *copied_accepted += accepted - copy_from;
  }
  out.resize(std::min(out.size(), size_t(L - context_len)));
  return out;
}

bool check_greedy_exact() {
  bool ok = true;
  for (uint64_t seed : {1, 2, 3}) {
    std::mt19937_64 rng(seed);
    int context_len = 0;
    const std::vector<int> text = agent_text(rng, &context_len);
    const std::vector<int> expect(text.begin() + context_len, text.end());
    long steps[3], copied[3];
    for (int mode = 0; mode < 3; ++mode) {
      const std::vector<int> got = decode_greedy(text, context_len, mode, 12, &steps[mode], &copied[mode]);
      ok &= got == expect;
    }
    const double n = double(expect.size());
    std::printf("  seed %llu: %zu output tokens; tokens/step drafter %.2f, replace %.2f, extend %.2f "
                "(copied tokens accepted: replace %ld, extend %ld)\n",
                static_cast<unsigned long long>(seed), expect.size(), n / steps[0], n / steps[1], n / steps[2],
                copied[1], copied[2]);
    ok &= steps[1] < steps[0] && steps[2] < steps[0] && copied[1] > 0 && copied[2] > 0;
  }
  return report("greedy: output token for token with copies (replace, extend), fewer steps", ok);
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

int main() {
  bool ok = check_proposals();
  ok &= check_incremental();
  ok &= check_extend();
  ok &= check_greedy_exact();
  ok &= check_sampled(1, 0, 3, 21);
  ok &= check_sampled(1, 0, 6, 22);
  ok &= check_sampled(2, 1, 3, 23);
  ok &= check_sampled(2, 2, 5, 27);
  ok &= check_cost();
  std::printf(ok ? "all copy tests passed\n" : "COPY TESTS FAILED\n");
  return ok ? 0 : 1;
}
