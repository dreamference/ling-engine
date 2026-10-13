// The long-copy proposer (moonshot P1, specs/drafts/MOONSHOT-repository-aware-decoding-2026-10-13.md §8): S1, the
// session's own tokens, indexed so that each step finds the earlier occurrences of the last few tokens in
// microseconds and proposes the continuation of the one that agrees furthest back, up to a long cap.
//
// The proposer changes only what a verify is offered, never what it accepts: every proposed token is checked
// by the target, so the output is what decoding without it gives (greedy: token for token; sampled: the
// target's distribution, a copied token being a deterministic proposal with q = 1). Nothing here touches the
// GPU; the index lives on the host beside each resident session.
#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace ling {

struct CopyParams {
  int min_match = 16;      // propose only when the corpus agrees with the query for at least this many tokens
  int max_tokens = 31;     // the longest proposal (the verify's rows minus the anchor's)
  int max_candidates = 16; // earlier occurrences of the query's last n-gram examined, most recent first
  int match_cap = 256;     // how far back the agreement is measured
  // > 0: the proposal is at most max(min_tokens, alpha * match) tokens (SuffixDecoding's and AgSpec's length
  // rule); 0: as long as the continuation and max_tokens allow (P0's "realistic proposer").
  double alpha = 0;
  int min_tokens = 1;
};

struct CopyProposal {
  std::vector<int> tokens;  // the proposed continuation (empty: none)
  int match = 0;            // the agreement before it: tokens of the query that equal the corpus at that occurrence
  int source = -1;          // the corpus position the proposal starts at
  int candidates = 0;       // occurrences examined
};

// An index of every n-gram of a token sequence (the corpus) that has at least one token after it: a hash of
// the n tokens to the most recent position where one ends, and for each position the previous one with the
// same hash. Its state is a function of the corpus alone, however the corpus was reached (appended in any
// pieces, truncated and regrown), so proposals do not depend on how a session's history was built (the
// exactness checks compare steps across prefill splits, resumes and sessions).
class CopyIndex {
 public:
  explicit CopyIndex(int ngram = 8);

  int ngram() const { return ngram_; }
  // Changes n (an empty index if it differs).
  void configure(int ngram);
  void clear();
  size_t size() const { return tokens_.size(); }
  const std::vector<int>& tokens() const { return tokens_; }

  // Brings the corpus to exactly a ++ b: keeps the common prefix with what is indexed, undoes the entries past
  // it (newest first), and appends the rest. A step's cost is the prefix comparison plus the new tokens.
  void sync(std::span<const int> a, std::span<const int> b = {});
  void append(std::span<const int> t);
  void truncate(size_t n);

  // The continuation of the earlier occurrence of `query`'s last n tokens that agrees with `query` furthest
  // back (ties: the most recent), read from the corpus. The corpus may be the query itself (the session's own
  // history followed by the anchor): an occurrence always has a token after it, so it is never the query's
  // own end.
  CopyProposal propose(std::span<const int> query, const CopyParams& p) const;
  CopyProposal propose(const CopyParams& p) const { return propose(tokens_, p); }

  // Entries undone by truncations so far (the tests check that a regrown corpus takes the incremental path).
  long undone() const { return undone_; }

 private:
  uint64_t hash_at(const int* end) const;  // the n tokens ending at `end` (inclusive)
  void insert(int end);                    // the n-gram ending at `end`, whose next token now exists
  void remove(int end);                    // its exact undo (the newest entry only)

  int ngram_;
  std::vector<int> tokens_;
  std::vector<int> prev_;  // prev_[e]: the previous indexed end with the same hash, or -1 (e indexed)
  std::unordered_map<uint64_t, int> head_;
  long undone_ = 0;
};

// Extend mode's chain: the drafter's chain `drafts` (sampled or greedy from its own distribution q) followed by
// the copy's continuation, only when the drafter's whole chain equals the copy's first tokens; at most
// `max_tokens` in all. Returns how many copied tokens were appended (0: the drafter's chain unchanged). The
// appended tokens are a deterministic function of the drafted prefix, so they are proposals with q = 1 and the
// rejection rule stays exact (tests/copy_tests.cpp checks the joint distribution).
int extend_with_copy(std::vector<int>& drafts, const std::vector<int>& copy, int max_tokens);

}  // namespace ling
