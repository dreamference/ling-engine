#include "core/engine.hpp"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "core/kernels.cuh"

namespace ling {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};

}  // namespace

template <typename T>
T* Engine::alloc(size_t count) {
  void* p = nullptr;
  check(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)), "cudaMalloc buffer");
  check(cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(T)), "cudaMemset buffer");
  buffers_.push_back(p);
  return static_cast<T*>(p);
}
template float* Engine::alloc<float>(size_t);
template int* Engine::alloc<int>(size_t);
template __half* Engine::alloc<__half>(size_t);
template __nv_bfloat16* Engine::alloc<__nv_bfloat16>(size_t);

Engine::Engine(const std::string& model_dir, EngineOptions opts) : opts_(opts) {
  model_ = std::make_unique<Model>(model_dir);
  const ModelConfig& c = model_->config();
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
  if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
  cublasSetStream(cublas_, stream_);
  cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH);

  const size_t M = opts_.prefill_chunk;
  const int C = c.lin_conv_channels(), Vs = c.lin_value_size();
  const int qsize = c.heads * c.head_dim, kvsize = c.kv_heads * c.head_dim;
  h_ = alloc<float>(M * c.hidden);
  xn_ = alloc<float>(M * c.hidden);
  t1_ = alloc<float>(M * c.intermediate);
  t2_ = alloc<float>(M * c.intermediate);
  mixed_ = alloc<float>(M * C);
  z_ = alloc<float>(M * Vs);
  a_ = alloc<float>(M * c.lin_v_heads);
  b_ = alloc<float>(M * c.lin_v_heads);
  g_ = alloc<float>(M * c.lin_v_heads);
  beta_ = alloc<float>(M * c.lin_v_heads);
  core_ = alloc<float>(M * Vs);
  normed_ = alloc<float>(M * Vs);
  q_gate_ = alloc<float>(M * 2 * qsize);
  k_ = alloc<float>(M * kvsize);
  v_ = alloc<float>(M * kvsize);
  q_ = alloc<float>(M * qsize);
  gate_ = alloc<float>(M * qsize);
  attn_ = alloc<float>(M * qsize);
  attn_scratch_ = alloc<float>(std::max(
      kernels::attention_scratch_floats(M, c.heads, c.kv_heads, c.head_dim, opts_.max_context),
      kernels::attention_rows_scratch_floats(kernels::kMaxStreamRows, c.heads, c.head_dim, opts_.max_context)));
  logits_dev_ = alloc<float>(size_t(kernels::kMaxStreamRows) * c.vocab);
  xh_ = alloc<__half>(size_t(kernels::kMaxStreamRows) * std::max(c.intermediate, std::max(c.hidden, std::max(C, qsize))));
  xinv_ = alloc<float>(kernels::kMaxStreamRows);
  x_bf16_ = alloc<__nv_bfloat16>(M * std::max(c.intermediate, std::max(c.hidden, C)));
  size_t largest = 0;
  for (const LayerWeights& L : model_->layers()) {
    for (const Fp4Weight* w : {&L.gate, &L.up, &L.down}) largest = std::max(largest, size_t(w->N) * w->K);
    for (const Fp8Weight* w : {&L.in_qkv, &L.in_z, &L.out, &L.q, &L.k, &L.v, &L.o})
      largest = std::max(largest, size_t(w->N) * w->K);
  }
  w_bf16_ = alloc<__nv_bfloat16>(largest);
  ids_dev_ = alloc<int>(M);

  layer_slot_.resize(c.layers);
  for (int i = 0; i < c.layers; ++i) {
    if (c.full_attention[i]) {
      layer_slot_[i] = static_cast<int>(kcache_.size());
      kcache_.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * kvsize));
      vcache_.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * kvsize));
    } else {
      layer_slot_[i] = static_cast<int>(gdn_state_.size());
      gdn_state_.push_back(alloc<float>(size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim));
      conv_state_.push_back(alloc<float>(size_t(C) * (c.conv_kernel - 1)));
      snap_gdn_.push_back(alloc<float>(size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim));
      snap_conv_.push_back(alloc<float>(size_t(C) * (c.conv_kernel - 1)));
    }
  }
  if (!opts_.draft_dir.empty()) {
    draft_ = std::make_unique<DraftModel>(opts_.draft_dir);
    alloc_drafter();
    std::fprintf(stderr, "drafter: %.2f GB, block %d\n", draft_->device_bytes() / 1e9, opts_.draft_block);
  }
  logits_.resize(c.vocab);
  profile_ = std::getenv("LING_PROFILE") != nullptr;
  rng_.seed(std::random_device{}());
}

Engine::~Engine() {
  if (profile_) {
    for (const auto& [phase, sec] : phase_seconds_) std::fprintf(stderr, "profile %-24s %8.3f s\n", phase.c_str(), sec);
  }
  for (void* p : buffers_) cudaFree(p);
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Engine::reset() {
  const ModelConfig& c = model_->config();
  for (float* s : gdn_state_)
    check(cudaMemsetAsync(s, 0, size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float), stream_), "reset");
  for (float* s : conv_state_)
    check(cudaMemsetAsync(s, 0, size_t(c.lin_conv_channels()) * (c.conv_kernel - 1) * sizeof(float), stream_),
          "reset");
  check(cudaStreamSynchronize(stream_), "reset");
  history_.clear();
  pos_ = 0;
}

uint64_t Engine::state_hash() {
  const ModelConfig& c = model_->config();
  const size_t gdn = size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim, conv = size_t(c.lin_conv_channels()) * (c.conv_kernel - 1);
  std::vector<float> host(gdn);
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](const float* d, size_t n) {
    check(cudaMemcpy(host.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost), "state_hash");
    const unsigned char* b = reinterpret_cast<const unsigned char*>(host.data());
    for (size_t i = 0; i < n * sizeof(float); ++i) h = (h ^ b[i]) * 1099511628211ull;
  };
  check(cudaStreamSynchronize(stream_), "state_hash");
  for (size_t i = 0; i < gdn_state_.size(); ++i) {
    mix(gdn_state_[i], gdn);
    mix(conv_state_[i], conv);
  }
  return h;
}

void Engine::set_draft_block(int b) {
  if (b < 2 || b > kernels::kMaxStreamRows) throw std::runtime_error("draft block must be in [2, 32]");
  opts_.draft_block = b;
}

void Engine::take_snapshot() {
  const ModelConfig& c = model_->config();
  const size_t gdn_bytes = size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float);
  const size_t conv_bytes = size_t(c.lin_conv_channels()) * (c.conv_kernel - 1) * sizeof(float);
  for (size_t i = 0; i < gdn_state_.size(); ++i) {
    check(cudaMemcpyAsync(snap_gdn_[i], gdn_state_[i], gdn_bytes, cudaMemcpyDeviceToDevice, stream_), "snapshot");
    check(cudaMemcpyAsync(snap_conv_[i], conv_state_[i], conv_bytes, cudaMemcpyDeviceToDevice, stream_), "snapshot");
  }
  check(cudaStreamSynchronize(stream_), "snapshot");
  snapshot_tokens_ = history_;
}

void Engine::restore_snapshot() {
  const ModelConfig& c = model_->config();
  const size_t gdn_bytes = size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float);
  const size_t conv_bytes = size_t(c.lin_conv_channels()) * (c.conv_kernel - 1) * sizeof(float);
  for (size_t i = 0; i < gdn_state_.size(); ++i) {
    check(cudaMemcpyAsync(gdn_state_[i], snap_gdn_[i], gdn_bytes, cudaMemcpyDeviceToDevice, stream_), "restore");
    check(cudaMemcpyAsync(conv_state_[i], snap_conv_[i], conv_bytes, cudaMemcpyDeviceToDevice, stream_), "restore");
  }
  check(cudaStreamSynchronize(stream_), "restore");
  history_ = snapshot_tokens_;
  pos_ = static_cast<int>(history_.size());
}

void Engine::linear_fp4(const Fp4Weight& w, const float* x, int M, float* y, bool x_ready) {
  if (M <= kernels::kMaxStreamRows) {  // the rows path: stream the weights once for every row
    if (!x_ready) kernels::to_half_rows(x, M, w.K, xh_, xinv_, stream_);
    kernels::stream_gemm_nvfp4(xh_, xinv_, M, w.w, nullptr, w.scale2, y, w.N, w.K, stream_);
    return;
  }
  kernels::dequant_tiled_nvfp4(w.w, w.scale2, w_bf16_, w.N, w.K, stream_);
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w_bf16_, y, w.N, w.K);
}

void Engine::linear_fp4_multi(std::initializer_list<std::pair<const Fp4Weight*, float*>> ws, const float* x, int M) {
  if (M > kernels::kMaxStreamRows) {
    for (const auto& [w, y] : ws) linear_fp4(*w, x, M, y);
    return;
  }
  kernels::StreamTarget t[4];
  int n = 0, K = 0;
  for (const auto& [w, y] : ws) {
    t[n++] = {w->w, nullptr, w->scale2, y, w->N};
    K = w->K;
  }
  kernels::to_half_rows(x, M, K, xh_, xinv_, stream_);
  kernels::stream_gemm_multi(true, xh_, xinv_, M, t, n, K, stream_);
}

void Engine::linear_fp8_multi(std::initializer_list<std::pair<const Fp8Weight*, float*>> ws, const float* x, int M) {
  if (M > kernels::kMaxStreamRows) {
    for (const auto& [w, y] : ws) linear_fp8(*w, x, M, y);
    return;
  }
  kernels::StreamTarget t[4];
  int n = 0, K = 0;
  for (const auto& [w, y] : ws) {
    t[n++] = {w->w, nullptr, w->scale, y, w->N};
    K = w->K;
  }
  kernels::to_half_rows(x, M, K, xh_, xinv_, stream_);
  kernels::stream_gemm_multi(false, xh_, xinv_, M, t, n, K, stream_);
}

void Engine::linear_bf16(const Bf16Weight& w, const float* x, int M, float* y) {
  if (M <= kernels::kMaxStreamRows) {  // the weights read once for every row, row-invariant
    kernels::bf16_rows(x, M, w.w, y, w.N, w.K, stream_);
    return;
  }
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w.w, y, w.N, w.K);
}

void Engine::linear_bf16_cublas(const Bf16Weight& w, const float* x, int M, float* y) {
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w.w, y, w.N, w.K);
}

void Engine::linear_fp8(const Fp8Weight& w, const float* x, int M, float* y, bool x_ready) {
  if (M <= kernels::kMaxStreamRows) {
    if (!x_ready) kernels::to_half_rows(x, M, w.K, xh_, xinv_, stream_);
    kernels::stream_gemm_fp8(xh_, xinv_, M, w.w, w.scale, y, w.N, w.K, stream_);
    return;
  }
  kernels::dequant_tiled_fp8(w.w, w.scale, w_bf16_, w.N, w.K, stream_);
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w_bf16_, y, w.N, w.K);
}

void Engine::forward(const int* ids, int M, Pass pass) {
  const ModelConfig& c = model_->config();
  if (pos_ + M > opts_.max_context) throw std::runtime_error("context is longer than max_context");
  const bool rows = M <= kernels::kMaxStreamRows, verify = pass == Pass::Verify;
  if (verify && !rows) throw std::runtime_error("verify: at most 32 rows");
  NvtxRange range(verify ? "verify" : M == 1 ? "decode" : "prefill_chunk");
  const int H = c.hidden, I = c.intermediate, C = c.lin_conv_channels(), Vs = c.lin_value_size();
  const int qsize = c.heads * c.head_dim;
  // LING_PROFILE=1: synchronize after each phase and add up its wall time (slow; for finding hot spots).
  auto t_last = std::chrono::steady_clock::now();
  const char* kind = verify ? "verify/" : M == 1 ? "decode/" : rows ? "rows/" : "prefill/";
  auto mark = [&](const char* phase) {
    if (!profile_) return;
    check(cudaStreamSynchronize(stream_), phase);
    auto now = std::chrono::steady_clock::now();
    phase_seconds_[std::string(kind) + phase] += std::chrono::duration<double>(now - t_last).count();
    t_last = now;
  };
  check(cudaMemcpyAsync(ids_dev_, ids, M * sizeof(int), cudaMemcpyHostToDevice, stream_), "ids");
  kernels::embed(model_->embed(), ids_dev_, M, H, h_, stream_);

  for (int li = 0; li < c.layers; ++li) {
    const LayerWeights& L = model_->layers()[li];
    const int slot = layer_slot_[li];
    kernels::rmsnorm(h_, L.input_norm, xn_, M, H, c.eps, true, stream_);
    if (L.full) {
      NvtxRange r("attention");
      linear_fp8_multi({{&L.q, q_gate_}, {&L.k, k_}, {&L.v, v_}}, xn_, M);
      mark("attn_proj");
      kernels::attn_prepare(q_gate_, k_, v_, L.q_norm, L.k_norm, pos_, M, c.heads, c.kv_heads, c.head_dim,
                            c.rotary_dim, c.rope_theta, c.eps, q_, gate_, kcache_[slot], vcache_[slot], stream_);
      if (rows)
        kernels::attention_rows(q_, kcache_[slot], vcache_[slot], pos_, M, c.heads, c.kv_heads, c.head_dim,
                                attn_scratch_, attn_, stream_);
      else
        kernels::attention(q_, kcache_[slot], vcache_[slot], pos_, M, c.heads, c.kv_heads, c.head_dim,
                           attn_scratch_, attn_, stream_);
      kernels::sigmoid_mul(attn_, gate_, M * qsize, stream_);
      mark("attn_core");
      linear_fp8(L.o, attn_, M, t1_);
      mark("attn_proj");
    } else {
      NvtxRange r("deltanet");
      // A verify keeps this layer's inputs for commit() and leaves the conv and recurrent state as they
      // were: the conv runs on a copy of its window, the recurrence reads the state and does not write it.
      float* mixed = verify ? v_post_[slot] : mixed_;
      float* g = verify ? v_g_[slot] : g_;
      float* beta = verify ? v_beta_[slot] : beta_;
      linear_fp8_multi({{&L.in_qkv, verify ? v_pre_[slot] : mixed}, {&L.in_z, z_}}, xn_, M);
      linear_bf16(L.in_a, xn_, M, a_);
      linear_bf16(L.in_b, xn_, M, b_);
      mark("gdn_proj");
      float* conv_state = conv_state_[slot];
      if (verify) {
        check(cudaMemcpyAsync(mixed, v_pre_[slot], size_t(M) * C * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
              "verify inputs");
        check(cudaMemcpyAsync(conv_tmp_, conv_state, size_t(C) * (c.conv_kernel - 1) * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream_),
              "verify conv");
        conv_state = conv_tmp_;
      }
      kernels::gdn_conv(mixed, conv_state, L.conv, M, C, stream_);
      kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g, beta, M, c.lin_v_heads, stream_);
      mark("gdn_conv");
      kernels::gdn_recurrent(mixed, g, beta, gdn_state_[slot], verify ? nullptr : gdn_state_[slot], core_, M,
                             c.lin_k_heads, c.lin_v_heads, stream_);
      mark("gdn_recurrent");
      kernels::gated_rmsnorm(core_, z_, L.lin_norm, normed_, M * c.lin_v_heads, c.lin_v_dim, c.eps, stream_);
      linear_fp8(L.out, normed_, M, t1_);
      mark("gdn_proj");
    }
    kernels::add_inplace(h_, t1_, M * H, stream_);
    {
      NvtxRange r("mlp");
      kernels::rmsnorm(h_, L.post_norm, xn_, M, H, c.eps, true, stream_);
      linear_fp4_multi({{&L.gate, t1_}, {&L.up, t2_}}, xn_, M);
      kernels::silu_mul(t1_, t2_, t1_, M * I, stream_);
      linear_fp4(L.down, t1_, M, t2_);
      kernels::add_inplace(h_, t2_, M * H, stream_);
      mark("mlp");
    }
    // The drafter conditions on the residual stream after these layers (SGLang captures it at the input
    // of the next layer).
    if (draft_ && feature_layer_[li] >= 0)
      kernels::copy_rows_bf16(h_, M, H, cap_ + size_t(feature_layer_[li]) * H, cap_stride_, stream_);
  }
  {
    NvtxRange r("lm_head");
    const Fp4Weight& lm = model_->lm_head();
    if (verify) {  // every row's logits stay on the GPU for the accept step
      kernels::rmsnorm(h_, model_->final_norm(), xn_, M, H, c.eps, true, stream_);
      linear_fp4(lm, xn_, M, logits_dev_);
      mark("lm_head");
      return;
    }
    const float* last = h_ + size_t(M - 1) * H;
    kernels::rmsnorm(last, model_->final_norm(), xn_, 1, H, c.eps, true, stream_);
    linear_fp4(lm, xn_, 1, logits_dev_);
  }
  check(cudaMemcpyAsync(logits_.data(), logits_dev_, logits_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_),
        "logits");
  check(cudaStreamSynchronize(stream_), "forward");
  mark("lm_head");
  pos_ += M;
}

const std::vector<float>& Engine::prefill(const std::vector<int>& prompt, EngineStats* stats) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");
  auto t0 = std::chrono::steady_clock::now();
  size_t reuse = 0;
  auto extends = [&](const std::vector<int>& prefix) {
    return !prefix.empty() && prefix.size() < prompt.size() && std::equal(prefix.begin(), prefix.end(), prompt.begin());
  };
  // Two states can be resumed: the current one (when the prompt extends everything processed so far)
  // and the snapshot taken at the end of the previous prompt. An agent's next turn re-renders the
  // model's output, so it rarely extends the current state, but it always extends the previous prompt.
  // The current state counts only if it exactly matches history_ (a failed request can leave it
  // half-advanced). Full-attention KV beyond the resumed position is simply overwritten.
  const bool current_ok = pos_ == static_cast<int>(history_.size()) && extends(history_);
  const bool snapshot_ok = extends(snapshot_tokens_);
  if (current_ok && (!snapshot_ok || history_.size() >= snapshot_tokens_.size())) {
    reuse = history_.size();
  } else if (snapshot_ok) {
    restore_snapshot();
    reuse = snapshot_tokens_.size();
  } else {
    snapshot_tokens_.clear();  // the prefill below overwrites the KV cache the snapshot relies on
    reset();
  }
  // The drafter attends to a sliding window of 2048 positions: only the prompt's last ones need its KV.
  if (draft_) window_start_ = std::max(0, static_cast<int>(prompt.size()) - draft_->config().sliding_window);
  for (size_t i = reuse; i < prompt.size();) {
    const int n = static_cast<int>(std::min<size_t>(opts_.prefill_chunk, prompt.size() - i));
    const int pos0 = pos_;
    forward(prompt.data() + i, n, Pass::Prefill);
    if (draft_) {
      const int first = std::max(pos0, window_start_);
      if (first < pos0 + n) draft_materialize(pos0 + n - first, first, first - pos0);
    }
    i += n;
  }
  history_ = prompt;
  take_snapshot();
  if (stats) {
    stats->reused_tokens = static_cast<int>(reuse);
    stats->prefill_tokens = static_cast<int>(prompt.size() - reuse);
    stats->prefill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return logits_;
}

const std::vector<float>& Engine::step(int token) {
  const int pos0 = pos_;
  forward(&token, 1, Pass::Decode);
  if (draft_) draft_materialize(1, pos0, 0);
  history_.push_back(token);
  return logits_;
}

int Engine::sample(const std::vector<float>& logits, const SamplingParams& p, const std::vector<int>& history) {
  const int V = static_cast<int>(logits.size());
  std::vector<float> l(logits);
  if (p.presence_penalty != 0.f || p.repetition_penalty != 1.f) {
    std::unordered_set<int> seen(history.begin(), history.end());
    for (int t : seen) {
      if (t < 0 || t >= V) continue;
      if (p.repetition_penalty != 1.f) l[t] = l[t] > 0 ? l[t] / p.repetition_penalty : l[t] * p.repetition_penalty;
      l[t] -= p.presence_penalty;
    }
  }
  if (p.temperature <= 0.f) return static_cast<int>(std::max_element(l.begin(), l.end()) - l.begin());
  const int k = (p.top_k > 0 && p.top_k < V) ? p.top_k : V;
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  idx.resize(k);
  std::vector<double> probs(k);
  const double mx = l[idx[0]];
  double sum = 0;
  for (int i = 0; i < k; ++i) sum += probs[i] = std::exp((l[idx[i]] - mx) / p.temperature);
  for (double& pr : probs) pr /= sum;
  int keep = k;
  if (p.top_p < 1.f) {
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      cum += probs[i];
      if (cum >= p.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  if (p.min_p > 0.f) {
    const double floor = probs[0] * p.min_p;
    int n = 0;
    while (n < keep && probs[n] >= floor) ++n;
    keep = std::max(n, 1);
  }
  if (p.seed != 0) rng_.seed(p.seed + history.size());
  std::discrete_distribution<int> dist(probs.begin(), probs.begin() + keep);
  return idx[dist(rng_)];
}

}  // namespace ling
