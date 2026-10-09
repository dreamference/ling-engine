// Sampling on the host: one row of logits to one token, with the finiteness guard. No CUDA here, so it is
// tested without a GPU (tests/sampling_tests.cpp).
#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>

namespace ling {

struct SamplingParams {
  float temperature = 1.0f;  // 0 = greedy
  int top_k = 20;
  float top_p = 0.95f;
  float min_p = 0.f;
  float presence_penalty = 0.f;
  float repetition_penalty = 1.f;
  uint64_t seed = 0;  // 0 = random
};

// Logits that are not all finite (NaN or inf) mean the forward pass went wrong. Sampled anyway, they gave
// token 0 ('!') over and over, or a uniform draw streamed as a healthy answer (sglang#33187, vllm#53305);
// the request must end with an error instead.
class NonFiniteLogits : public std::runtime_error {
 public:
  NonFiniteLogits() : std::runtime_error("the model produced non-finite logits (NaN or inf); the request was stopped") {}
};

bool all_finite(std::span<const float> values);
// Throws NonFiniteLogits unless every logit is finite.
void require_finite(std::span<const float> logits);
// The same for topk_rows' result copied to the host (`vals` and `ids`, row by row). The kernel skips NaN,
// so a row with no finite logit comes back as -FLT_MAX with an id past the vocabulary; an inf is kept.
// Both are caught here. A row with only some NaN logits is not: that needs a scan on the device.
void require_finite_topk(std::span<const float> vals, std::span<const int> ids, int vocab);

// Picks the next token from one row of logits. Presence and repetition penalties apply once to each
// distinct token of `output`, the tokens the request has generated so far: like production, they never
// count the prompt (sglang#41124). Then greedy (temperature <= 0, ties to the lower id), or temperature,
// top-k, top-p and min-p and a draw from `rng`. Throws NonFiniteLogits if any logit is NaN or inf.
int sample_logits(std::span<const float> logits, const SamplingParams& p, std::span<const int> output,
                  std::mt19937_64& rng);

}  // namespace ling
