// Host-side sampling checks (no GPU): the finiteness guard, with NaN and inf injected into a logits row
// and into a top-K result as topk_rows returns it, and the sampler's basic choices.
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "core/sampling.hpp"

using ling::NonFiniteLogits;
using ling::SamplingParams;

static int failures = 0;
#define EXPECT(cond)                                              \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

// Whether sampling `logits` throws NonFiniteLogits (and returns no token).
static bool refused(const std::vector<float>& logits, const SamplingParams& p) {
  std::mt19937_64 rng(1);
  try {
    (void)ling::sample_logits(logits, p, {}, rng);
  } catch (const NonFiniteLogits&) {
    return true;
  }
  return false;
}

static bool topk_refused(const std::vector<float>& vals, const std::vector<int>& ids, int vocab) {
  try {
    ling::require_finite_topk(vals, ids, vocab);
  } catch (const NonFiniteLogits&) {
    return true;
  }
  return false;
}

int main() {
  constexpr int V = 1000;
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  SamplingParams greedy;
  greedy.temperature = 0.f;
  SamplingParams sampled;  // the checkpoint's defaults: temperature 1, top-k 20, top-p 0.95
  SamplingParams untruncated;
  untruncated.top_k = -1;
  untruncated.top_p = 1.f;
  std::vector<float> row(V);
  for (int i = 0; i < V; ++i) row[i] = std::sin(0.37f * i);  // finite, no ties at the top
  {
    // Finite logits sample normally.
    EXPECT(!refused(row, greedy) && !refused(row, sampled) && !refused(row, untruncated));
  }
  {
    // sglang#33187, sglang#33697, vllm#53305: one NaN, a whole NaN row, +inf or -inf end the request;
    // they never become token 0 ('!') or a uniform draw. The position does not matter.
    for (int at : {0, 1, V / 2, V - 1}) {
      for (float bad : {nan, inf, -inf}) {
        std::vector<float> l = row;
        l[at] = bad;
        EXPECT(refused(l, greedy));
        EXPECT(refused(l, sampled));
        EXPECT(refused(l, untruncated));
      }
    }
    EXPECT(refused(std::vector<float>(V, nan), greedy));
    EXPECT(refused(std::vector<float>(V, nan), sampled));
    // A penalty does not hide a NaN.
    SamplingParams pen = sampled;
    pen.presence_penalty = 1.5f;
    std::vector<float> l = row;
    l[7] = nan;
    std::mt19937_64 rng(1);
    bool threw = false;
    try {
      const std::vector<int> out = {7, 8};
      (void)ling::sample_logits(l, pen, out, rng);
    } catch (const NonFiniteLogits&) {
      threw = true;
    }
    EXPECT(threw);
  }
  {
    // The speculative accept step's view: topk_rows skips NaN, so an all-NaN row comes back as -FLT_MAX
    // with id 0x7fffffff; an inf is kept. Two rows of K = 2.
    const std::vector<float> ok_vals = {3.f, 2.f, 1.5f, -4.f};
    const std::vector<int> ok_ids = {5, 9, 0, 999};
    EXPECT(!topk_refused(ok_vals, ok_ids, V));
    EXPECT(topk_refused({3.f, 2.f, -FLT_MAX, -FLT_MAX}, {5, 9, 0x7fffffff, 0x7fffffff}, V));  // second row all NaN
    EXPECT(topk_refused({inf, 2.f, 1.f, 0.f}, {5, 9, 1, 2}, V));
    EXPECT(topk_refused({nan, 2.f, 1.f, 0.f}, {5, 9, 1, 2}, V));
    EXPECT(topk_refused({3.f, 2.f, 1.f, 0.f}, {5, V, 1, 2}, V));  // an id past the vocabulary
    EXPECT(topk_refused({3.f}, {-1}, V));
  }
  {
    // Greedy picks the largest logit, ties to the lower id; a truncation to one token is greedy too.
    std::vector<float> l(V, 0.f);
    l[42] = 5.f;
    l[17] = 5.f;
    std::mt19937_64 rng(3);
    EXPECT(ling::sample_logits(l, greedy, {}, rng) == 17);
    SamplingParams one = sampled;
    one.top_k = 1;
    EXPECT(ling::sample_logits(row, one, {}, rng) == ling::sample_logits(row, greedy, {}, rng));
  }
  std::printf(failures ? "%d SAMPLING TEST FAILURES\n" : "all sampling tests passed\n", failures);
  return failures ? 1 : 0;
}
