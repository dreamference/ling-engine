// The context-lookup draft source (SPEC.md §10): agents copy (file contents into patches, names into
// commands, earlier tool calls into new ones), so the continuation of the most recent earlier occurrence
// of the last few tokens is often what comes next. Costs microseconds on the host.
#pragma once

#include <vector>

namespace ling {

struct LookupProposal {
  std::vector<int> tokens;  // the proposed continuation (empty: no match)
  int match = 0;            // how many tokens before the proposal agree with the matched occurrence (>= ngram)
};

// The continuation of the most recent earlier occurrence of the last `ngram` tokens of `seq`, at most
// `max` tokens. `match` counts how far back the agreement goes (capped at 64): a long match is a
// verbatim copy in progress.
LookupProposal lookup_propose(const std::vector<int>& seq, int max, int ngram = 3);

}  // namespace ling
