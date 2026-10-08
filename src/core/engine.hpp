// One sequence at a time: the forward pass (chunked prefill, then one token per step), its state
// (paged-free v0: a flat KV cache for the full-attention layers, DeltaNet and conv state for the rest),
// and sampling.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "core/model.hpp"

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

struct EngineOptions {
  int max_context = 65536;
  int prefill_chunk = 1024;
};

struct EngineStats {
  double prefill_seconds = 0;
  int prefill_tokens = 0;
  int reused_tokens = 0;
};

class Engine {
 public:
  Engine(const std::string& model_dir, EngineOptions opts = {});
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const ModelConfig& config() const { return model_->config(); }
  const Model& model() const { return *model_; }
  int max_context() const { return opts_.max_context; }

  // Brings the state to exactly `prompt` (reusing the current state when `prompt` extends the tokens
  // already processed) and returns the logits after its last token.
  const std::vector<float>& prefill(const std::vector<int>& prompt, EngineStats* stats = nullptr);
  // Appends one token and returns the logits after it.
  const std::vector<float>& step(int token);
  // Picks the next token from logits.
  int sample(const std::vector<float>& logits, const SamplingParams& p, const std::vector<int>& history);

  const std::vector<int>& history() const { return history_; }
  void reset();

 private:
  void forward(const int* ids, int M);  // appends M tokens at position pos_; logits for the last one
  void linear_fp4(const Fp4Weight& w, const float* x, int M, float* y);
  void linear_fp8(const Fp8Weight& w, const float* x, int M, float* y);
  void linear_bf16(const Bf16Weight& w, const float* x, int M, float* y);
  void take_snapshot();
  void restore_snapshot();

  EngineOptions opts_;
  std::unique_ptr<Model> model_;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  std::vector<void*> buffers_;
  template <typename T>
  T* alloc(size_t count);

  // Activations, sized for one prefill chunk.
  float *h_ = nullptr, *xn_ = nullptr, *t1_ = nullptr, *t2_ = nullptr, *mixed_ = nullptr, *z_ = nullptr;
  float *a_ = nullptr, *b_ = nullptr, *g_ = nullptr, *beta_ = nullptr, *core_ = nullptr, *normed_ = nullptr;
  float *q_gate_ = nullptr, *k_ = nullptr, *v_ = nullptr, *q_ = nullptr, *gate_ = nullptr, *attn_ = nullptr;
  float *attn_scratch_ = nullptr, *logits_dev_ = nullptr;
  __nv_bfloat16 *x_bf16_ = nullptr, *w_bf16_ = nullptr;
  int* ids_dev_ = nullptr;

  // Sequence state.
  std::vector<__nv_bfloat16*> kcache_, vcache_;  // per full-attention layer, [max_context][Hkv][D]
  std::vector<float*> gdn_state_, conv_state_;    // per linear layer
  std::vector<int> layer_slot_;                   // layer -> index into the vectors above
  std::vector<int> history_;
  // DeltaNet and conv state at the end of the last prompt, and that prompt.
  std::vector<float*> snap_gdn_, snap_conv_;
  std::vector<int> snapshot_tokens_;
  int pos_ = 0;

  std::vector<float> logits_;
  bool profile_ = false;
  std::map<std::string, double> phase_seconds_;
  std::mt19937_64 rng_;
};

}  // namespace ling
