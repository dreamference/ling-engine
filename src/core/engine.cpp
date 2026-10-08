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
  attn_scratch_ = alloc<float>(kernels::attention_scratch_floats(M, c.heads, c.kv_heads, c.head_dim, opts_.max_context));
  logits_dev_ = alloc<float>(c.vocab);
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

void Engine::linear_fp4(const Fp4Weight& w, const float* x, int M, float* y) {
  if (M <= kernels::kMaxGemvRows) {
    kernels::gemv_nvfp4(x, M, w.w, w.scale, w.scale2, y, w.N, w.K, stream_);
    return;
  }
  kernels::dequant_nvfp4(w.w, w.scale, w.scale2, w_bf16_, w.N, w.K, stream_);
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w_bf16_, y, w.N, w.K);
}

void Engine::linear_bf16(const Bf16Weight& w, const float* x, int M, float* y) {
  if (M <= kernels::kMaxGemvRows) {
    kernels::gemv_bf16(x, M, w.w, y, w.N, w.K, stream_);
    return;
  }
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w.w, y, w.N, w.K);
}

void Engine::linear_fp8(const Fp8Weight& w, const float* x, int M, float* y) {
  if (M <= kernels::kMaxGemvRows) {
    kernels::gemv_fp8(x, M, w.w, w.scale, y, w.N, w.K, stream_);
    return;
  }
  kernels::dequant_fp8(w.w, w.scale, w_bf16_, w.N, w.K, stream_);
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w_bf16_, y, w.N, w.K);
}

void Engine::forward(const int* ids, int M) {
  const ModelConfig& c = model_->config();
  if (pos_ + M > opts_.max_context) throw std::runtime_error("context is longer than max_context");
  NvtxRange range(M == 1 ? "decode" : "prefill_chunk");
  const int H = c.hidden, I = c.intermediate, C = c.lin_conv_channels(), Vs = c.lin_value_size();
  const int qsize = c.heads * c.head_dim;
  // LING_PROFILE=1: synchronize after each phase and add up its wall time (slow; for finding hot spots).
  auto t_last = std::chrono::steady_clock::now();
  auto mark = [&](const char* phase) {
    if (!profile_) return;
    check(cudaStreamSynchronize(stream_), phase);
    auto now = std::chrono::steady_clock::now();
    phase_seconds_[std::string(M == 1 ? "decode/" : "prefill/") + phase] += std::chrono::duration<double>(now - t_last).count();
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
      linear_fp8(L.q, xn_, M, q_gate_);
      linear_fp8(L.k, xn_, M, k_);
      linear_fp8(L.v, xn_, M, v_);
      mark("attn_proj");
      kernels::attn_prepare(q_gate_, k_, v_, L.q_norm, L.k_norm, pos_, M, c.heads, c.kv_heads, c.head_dim,
                            c.rotary_dim, c.rope_theta, c.eps, q_, gate_, kcache_[slot], vcache_[slot], stream_);
      kernels::attention(q_, kcache_[slot], vcache_[slot], pos_, M, c.heads, c.kv_heads, c.head_dim,
                         attn_scratch_, attn_, stream_);
      kernels::sigmoid_mul(attn_, gate_, M * qsize, stream_);
      mark("attn_core");
      linear_fp8(L.o, attn_, M, t1_);
      mark("attn_proj");
    } else {
      NvtxRange r("deltanet");
      linear_fp8(L.in_qkv, xn_, M, mixed_);
      linear_fp8(L.in_z, xn_, M, z_);
      linear_bf16(L.in_a, xn_, M, a_);
      linear_bf16(L.in_b, xn_, M, b_);
      mark("gdn_proj");
      kernels::gdn_conv(mixed_, conv_state_[slot], L.conv, M, C, stream_);
      kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g_, beta_, M, c.lin_v_heads, stream_);
      mark("gdn_conv");
      kernels::gdn_recurrent(mixed_, g_, beta_, gdn_state_[slot], core_, M, c.lin_k_heads, c.lin_v_heads, stream_);
      mark("gdn_recurrent");
      kernels::gated_rmsnorm(core_, z_, L.lin_norm, normed_, M * c.lin_v_heads, c.lin_v_dim, c.eps, stream_);
      linear_fp8(L.out, normed_, M, t1_);
      mark("gdn_proj");
    }
    kernels::add_inplace(h_, t1_, M * H, stream_);
    {
      NvtxRange r("mlp");
      kernels::rmsnorm(h_, L.post_norm, xn_, M, H, c.eps, true, stream_);
      linear_fp4(L.gate, xn_, M, t1_);
      linear_fp4(L.up, xn_, M, t2_);
      kernels::silu_mul(t1_, t2_, t1_, M * I, stream_);
      linear_fp4(L.down, t1_, M, t2_);
      kernels::add_inplace(h_, t2_, M * H, stream_);
      mark("mlp");
    }
  }
  {
    NvtxRange r("lm_head");
    const float* last = h_ + size_t(M - 1) * H;
    kernels::rmsnorm(last, model_->final_norm(), xn_, 1, H, c.eps, true, stream_);
    const Fp4Weight& lm = model_->lm_head();
    kernels::gemv_nvfp4(xn_, 1, lm.w, lm.scale, lm.scale2, logits_dev_, lm.N, lm.K, stream_);
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
  for (size_t i = reuse; i < prompt.size();) {
    const int n = static_cast<int>(std::min<size_t>(opts_.prefill_chunk, prompt.size() - i));
    forward(prompt.data() + i, n);
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
  forward(&token, 1);
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
