// One sequence at a time: the forward pass (chunked prefill, then decode or speculative verify), its
// state (paged-free: a flat KV cache for the full-attention layers, DeltaNet and conv state for the rest),
// sampling, and speculation with the DFlash2 drafter (speculate.cpp).
//
// Two forward paths:
//   - the rows path, M <= 32 tokens (decode, verify, short prompts): tensor-core weight streaming
//     (stream_gemm.cu) and fixed-range attention, every kernel row-invariant, so a token verified in a
//     block of 16 gets bit-for-bit the numbers it gets when decoded alone;
//   - the prefill path, longer chunks: activations quantized as production quantizes them (NVFP4 for the
//     NVFP4 matrices, FP8 for the FP8 ones) and block-scaled tensor-core GEMMs over the same tiled
//     weights (prefill_gemm.cu); tiled prefill attention.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/drafter.hpp"
#include "core/model.hpp"
#include "core/sampling.hpp"

namespace ling {

struct EngineOptions {
  int max_context = 65536;
  int prefill_chunk = 2048;  // measured: 647 tokens/s against 582 at 1024 on a 7.7K-token prompt
  std::string draft_dir;     // the DFlash2 drafter's checkpoint; empty: no speculation
  int draft_block = 16;      // rows per verify: the anchor and block - 1 drafted tokens (2 .. 32)
  // The context-lookup source: 0 off; 1 shadow (proposed and scored afterwards against what was really
  // generated, never verified); 2 auto (verified instead of the drafter's chain when the match is at
  // least lookup_min_match tokens long, and the drafter's pass is skipped).
  int lookup_mode = 1;
  int lookup_min_match = 8;
  // Prefix checkpoints: DeltaNet and conv states kept at chosen prompt positions (the end of the first two
  // messages, i.e. after `boundary_token`, and multiples of prefill_chunk up to checkpoint_limit), so a later
  // prompt that shares the prefix resumes there. Each costs ~157 MB; 0 turns them off.
  int prefix_checkpoints = 8;
  int boundary_token = -1;  // <|im_end|>
  int checkpoint_limit = 32768;
};

struct EngineStats {
  double prefill_seconds = 0;
  int prefill_tokens = 0;
  int reused_tokens = 0;
};

// Counters of speculative steps since the last reset.
struct SpecStats {
  long steps = 0;
  long drafted = 0;   // drafted tokens offered to the verify
  long accepted = 0;  // drafted tokens accepted
  std::vector<long> accept_hist = std::vector<long>(33, 0);
  double draft_seconds = 0, verify_seconds = 0, commit_seconds = 0;
  long lookup_steps = 0, lookup_accepted = 0;  // steps that verified the lookup chain, and its accepted tokens
  // Shadow scoring, per bucket of match length (3, 4-7, 8-15, 16-31, 32+): steps with a proposal, the
  // tokens the proposal would have had accepted (its agreement with what was generated next), and the
  // drafter's accepted tokens on the same steps. `shadow_best` adds up the better of the two per step.
  long shadow_steps[5] = {}, shadow_lookup[5] = {}, shadow_dflash[5] = {};
  long shadow_best = 0, shadow_total_steps = 0, shadow_dflash_all = 0;
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
  // already processed) and returns the logits after its last token. Throws NonFiniteLogits, keeping no
  // state from the failed pass, when a pass's logits are not finite.
  const std::vector<float>& prefill(const std::vector<int>& prompt, EngineStats* stats = nullptr);
  // Appends one token and returns the logits after it.
  const std::vector<float>& step(int token);
  // Picks the next token from logits (sample_logits). Throws NonFiniteLogits if any logit is NaN or inf.
  int sample(const std::vector<float>& logits, const SamplingParams& p, const std::vector<int>& history);

  // Speculation. `anchor` is the next token, chosen but not yet processed (what step() would take).
  // One step drafts block - 1 tokens, verifies anchor + drafts in one pass, and keeps the anchor and the
  // accepted drafts. Returns the accepted drafts followed by the next token (chosen, not processed: the
  // next call's anchor). Greedy output equals plain decoding's token for token; sampled output follows
  // the same distribution (exact rejection sampling).
  bool has_drafter() const { return draft_ != nullptr; }
  bool can_speculate(const SamplingParams& p) const;
  std::vector<int> speculate(int anchor, const SamplingParams& p);
  const SpecStats& spec_stats() {
    score_shadow(true);
    return spec_stats_;
  }
  void reset_spec_stats() {
    spec_stats_ = SpecStats{};
    shadow_.clear();
  }
  void set_lookup(int mode, int min_match) {
    opts_.lookup_mode = mode;
    opts_.lookup_min_match = min_match;
  }
  int draft_block() const { return opts_.draft_block; }
  void set_draft_block(int b);

  const std::vector<int>& history() const { return history_; }
  void reset();
  // FNV-1a over every DeltaNet and conv state, bit for bit (the exactness tests compare runs with it).
  uint64_t state_hash();

 private:
  enum class Pass { Prefill, Decode, Verify };
  // Appends M tokens at position pos_ (a verify writes their KV but leaves pos_ and the recurrent state
  // to commit()). Prefill and Decode leave the logits after the last token in logits_; Verify leaves the
  // logits of every row in logits_dev_.
  // keep_at (prefill only, 0 < keep_at < M): also keep the DeltaNet and conv state after the first keep_at
  // tokens in snap_ (the end-of-prompt state at a multiple of 32, without a separate pass for the tail).
  void forward(const int* ids, int M, Pass pass, int keep_at = -1);
  // The rows path's linear layers take x through an FP16 copy; `x_ready` reuses the previous call's copy
  // (the same x feeding several matrices).
  void linear_fp4(const Fp4Weight& w, const float* x, int M, float* y, bool x_ready = false);
  void linear_fp8(const Fp8Weight& w, const float* x, int M, float* y, bool x_ready = false);
  // Several matrices with the same input: one launch on the rows path, separate calls otherwise.
  void linear_fp4_multi(std::initializer_list<std::pair<const Fp4Weight*, float*>> ws, const float* x, int M,
                        bool x_ready = false);
  void linear_fp8_multi(std::initializer_list<std::pair<const Fp8Weight*, float*>> ws, const float* x, int M,
                        bool x_ready = false);
  // h_ += x . W^T (the out-projections into the residual stream; the prefill GEMM adds in its epilogue).
  void linear_fp8_residual(const Fp8Weight& w, const float* x, int M, bool x_ready = false);
  // The prefill path (quantized activations, prefill_gemm.cu) or the rows path (stream_gemm.cu)?
  bool quantized(int M) const { return quantized_ || M > kRowsMax; }
  static constexpr int kRowsMax = 32;  // kernels::kMaxStreamRows
  static constexpr int kAttnSlice = 256;
  static constexpr int kResumeAlign = 32;  // kept states sit at multiples of this (the DeltaNet's chunk)  // prefill attention: queries per call (bounds the partials' scratch)
  bool quantized_ = false;                // set by forward(): the pass is a prefill
  // RMSNorm; on the rows path it also leaves the FP16 copy for the next linear call (x_ready).
  // On the prefill path it writes the quantized input of the next GEMMs instead (input scale `in_scale`,
  // NVFP4 or FP8), and `out` only when `f32`.
  bool norm_rows(const float* x, const __nv_bfloat16* w, float* out, int M, float in_scale, bool nvfp4, bool f32);
  // What xq_ holds: the quantization of the current input with this input scale and encoding (0: nothing
  // reusable). x_ready on the prefill path means "xq_ holds it".
  float xq_scale_ = 0.f;
  bool xq_nvfp4_ = false;
  void set_xq(float scale, bool nvfp4) {
    xq_scale_ = scale;
    xq_nvfp4_ = nvfp4;
  }
  bool xq_is(float scale, bool nvfp4) const { return scale != 0.f && xq_scale_ == scale && xq_nvfp4_ == nvfp4; }
  void linear_bf16(const Bf16Weight& w, const float* x, int M, float* y);
  void linear_bf16_cublas(const Bf16Weight& w, const float* x, int M, float* y);  // the drafter's BF16 matrices
  // DeltaNet and conv state, saved or restored as a whole.
  struct StateSlot {
    std::vector<float*> gdn, conv;
  };
  StateSlot alloc_slot();
  void copy_state(const StateSlot& slot, bool save);  // save: state -> slot; else slot -> state
  void reset_state();                                 // the zero state of an empty sequence
  void save_checkpoint(int pos);
  void truncate_history(int n);  // forget the tokens from position n on, and every state past them

  // Speculation (speculate.cpp).
  void commit(int n);                                 // advance the recurrent state over a verify's first n rows
  void draft_materialize(int rows, int pos0, int cap_row0);  // target features -> the drafter's context KV
  void draft_propose(int anchor, int B);              // the drafter's block, top-16 candidates and lattice
  void alloc_drafter();

  EngineOptions opts_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<DraftModel> draft_;
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
  __nv_bfloat16* x_bf16_ = nullptr;
  uint8_t *xq_ = nullptr, *xsf_ = nullptr;  // the prefill path's quantized activations and their block scales
  uint8_t *xq2_ = nullptr, *xsf2_ = nullptr;
  __half* xh_ = nullptr;     // the rows path's FP16 activations [32][max K]
  float* xinv_ = nullptr;    // and their per-row scales
  int* ids_dev_ = nullptr;

  // Sequence state.
  std::vector<__nv_bfloat16*> kcache_, vcache_;  // per full-attention layer, [max_context][Hkv][D]
  std::vector<float*> gdn_state_, conv_state_;    // per linear layer
  std::vector<int> layer_slot_;                   // layer -> index into the vectors above
  std::vector<int> history_;
  // Kept states. Each is the state after history_[0, pos), valid while those tokens stay in history_.
  StateSlot snap_;     // the end of the last prompt
  int snap_pos_ = -1;
  struct Checkpoint {
    int pos, slot;
  };
  std::vector<StateSlot> ckpt_slots_;
  std::vector<Checkpoint> ckpts_;  // sorted by position
  int dkv_lo_ = 0;  // the drafter's context KV is valid for positions [dkv_lo_, history_.size())
  int pos_ = 0;

  // A verify's per-layer DeltaNet inputs, kept for commit(): the in-projection before and after the conv,
  // the decay and the write strength, for up to 32 rows.
  std::vector<float*> v_pre_, v_post_, v_g_, v_beta_;
  float* conv_tmp_ = nullptr;

  // Drafter: target features [rows][layers * hidden] (BF16), its context KV per layer [max_context][kv],
  // and the block's buffers.
  std::vector<int> feature_layer_;  // target layer -> index of its feature slot, or -1
  __nv_bfloat16* cap_ = nullptr;
  int cap_stride_ = 0;
  std::vector<__nv_bfloat16*> dkc_, dvc_;
  float *dctx_ = nullptr, *dk_ = nullptr, *dv_ = nullptr, *dres_ = nullptr, *dh_ = nullptr, *dh2_ = nullptr;
  float *dout_ = nullptr, *dcoef_ = nullptr, *dq_ = nullptr, *dg_ = nullptr, *du_ = nullptr, *dattn_ = nullptr;
  float *dlogits_ = nullptr, *dunary_ = nullptr, *dhp_ = nullptr, *dscores_ = nullptr;
  int* dcand_ = nullptr;
  float *topk_scratch_ = nullptr, *vals_dev_ = nullptr;
  int* ids_out_dev_ = nullptr;
  std::vector<int> cand_host_;
  std::vector<float> scores_host_;
  struct ShadowRecord {
    int pos;                  // the anchor's position
    std::vector<int> tokens;  // the lookup's proposal for the positions after it
    int match, dflash_accepted, bucket;
  };
  std::vector<ShadowRecord> shadow_;
  void score_shadow(bool all);  // scores records whose continuation is known (all: the rest as far as known)  // the prompt's drafter context starts here (positions before are not needed)

  std::vector<float> logits_;
  bool profile_ = false;
  std::map<std::string, double> phase_seconds_;
  std::mt19937_64 rng_;
  SpecStats spec_stats_;
};

}  // namespace ling
