#include "core/lookup.hpp"

#include <algorithm>

namespace ling {

LookupProposal lookup_propose(const std::vector<int>& seq, int max, int ngram) {
  LookupProposal p;
  const int L = static_cast<int>(seq.size());
  if (L <= ngram || max <= 0) return p;
  const int* s = seq.data();
  const int* key = s + L - ngram;
  // Most recent occurrence first; an occurrence must leave at least one token after it.
  for (int i = L - ngram - 1; i >= 0; --i) {
    if (s[i] != key[0] || !std::equal(key, key + ngram, s + i)) continue;
    int m = ngram;
    while (m < 64 && i - (m - ngram) - 1 >= 0 && s[i - (m - ngram) - 1] == s[L - m - 1]) ++m;
    const int start = i + ngram, n = std::min(max, L - start);
    p.tokens.assign(s + start, s + start + n);
    p.match = m;
    return p;
  }
  return p;
}

}  // namespace ling
