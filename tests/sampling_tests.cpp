// Host-side sampling checks (no GPU): the finiteness guard, with NaN and inf injected into a logits row
// and into a top-K result as topk_rows returns it, and the sampler's basic choices.
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <utility>
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
  {
    // sglang#41124, with production's semantics: presence and repetition penalties count the request's
    // output only. Engine::sample hands sample_logits history() past the last prompt, so a token that
    // is the prompt's own and not yet generated keeps its logit: at the first output position (no output
    // yet) a penalty never changes the greedy choice.
    std::vector<float> l(V, 0.f);
    l[42] = 4.f;  // say, the secret word the prompt asks the model to repeat
    l[17] = 3.f;
    std::mt19937_64 rng(5);
    for (const auto& [presence, repetition] : {std::pair{2.f, 1.f}, std::pair{0.f, 2.f}}) {
      SamplingParams pen = greedy;
      pen.presence_penalty = presence;
      pen.repetition_penalty = repetition;
      EXPECT(ling::sample_logits(l, pen, {}, rng) == 42);                         // no output yet
      const std::vector<int> once = {42}, three_times = {42, 42, 42};
      EXPECT(ling::sample_logits(l, pen, once, rng) == 17);                       // generated once: penalized
      EXPECT(ling::sample_logits(l, pen, three_times, rng) == 17);                // once per distinct token
    }
    // A repetition penalty divides a positive logit and multiplies a negative one; presence subtracts,
    // once however often the token was generated: 42 at 4 -> 4 - 1.5 = 2.5 against 17 at 3.
    SamplingParams pres = greedy;
    pres.presence_penalty = 1.5f;
    const std::vector<int> repeated = {42, 42, 42, 42};
    EXPECT(ling::sample_logits(l, pres, repeated, rng) == 17);
    pres.presence_penalty = 0.5f;  // 3.5 still beats 3
    EXPECT(ling::sample_logits(l, pres, repeated, rng) == 42);
    std::vector<float> neg(V, -10.f);
    neg[3] = -1.f;
    neg[4] = -1.5f;
    SamplingParams rep = greedy;
    rep.repetition_penalty = 2.f;  // -1 * 2 = -2 falls below -1.5
    const std::vector<int> three = {3};
    EXPECT(ling::sample_logits(neg, rep, three, rng) == 4);
  }
  std::printf(failures ? "%d SAMPLING TEST FAILURES\n" : "all sampling tests passed\n", failures);
  return failures ? 1 : 0;
}
