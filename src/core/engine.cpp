#include "core/engine.hpp"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

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
template uint8_t* Engine::alloc<uint8_t>(size_t);

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
  attn_scratch_ = alloc<float>(kernels::attention_rows_scratch_floats(std::max(kAttnSlice, kernels::kMaxStreamRows),
                                                                      c.heads, c.head_dim, opts_.max_context));
  logits_dev_ = alloc<float>(size_t(kernels::kMaxStreamRows) * c.vocab);
  xh_ = alloc<__half>(size_t(kernels::kMaxStreamRows) * std::max(c.intermediate, std::max(c.hidden, std::max(C, qsize))));
  xinv_ = alloc<float>(kernels::kMaxStreamRows);
  x_bf16_ = alloc<__nv_bfloat16>(M * std::max(c.intermediate, std::max(c.hidden, C)));
  const size_t widest = std::max<size_t>(c.intermediate, std::max(c.hidden, std::max(C, 2 * qsize)));
  xq_ = alloc<uint8_t>(M * widest);         // FP8 activations, or NVFP4 in the first half
  xsf_ = alloc<uint8_t>(M * widest / 16);  // NVFP4 block scales
  xq2_ = alloc<uint8_t>(M * c.intermediate / 2);  // the FFN's fused gate/up output (the down projection's input)
  xsf2_ = alloc<uint8_t>(M * c.intermediate / 16);
  ids_dev_ = alloc<int>(M);
  pos_dev_ = alloc<int>(1);
  check(cudaHostAlloc(reinterpret_cast<void**>(&pin_), sizeof(StepPinned), cudaHostAllocDefault), "pinned step buffers");
  kernels::prepare_kernels();  // attributes are set here, never inside a graph capture

  // Head-major KV caches: one KV head's keys are one contiguous stream for the attention kernels.
  kernels::set_kv_layout(size_t(opts_.max_context) * c.head_dim, c.head_dim);
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
    }
  }
  snap_ = alloc_slot();
  for (int i = 0; i < opts_.prefix_checkpoints; ++i) ckpt_slots_.push_back(alloc_slot());
  if (!opts_.draft_dir.empty()) {
    draft_ = std::make_unique<DraftModel>(opts_.draft_dir);
    alloc_drafter();
    std::fprintf(stderr, "drafter: %.2f GB, block %d\n", draft_->device_bytes() / 1e9, opts_.draft_block);
  }
  logits_.resize(c.vocab);
  profile_ = std::getenv("LING_PROFILE") != nullptr;
  rng_.seed(std::random_device{}());
  // Step graphs: asked for by the options or LING_STEP_GRAPHS, and only where nothing inside a step would
  // synchronize (LING_PROFILE's per-phase timing) or allocate (LING_KSPLIT's split-K buffer) while capturing.
  bool graphs = opts_.step_graphs;
  if (const char* e = std::getenv("LING_STEP_GRAPHS")) graphs = std::atoi(e) != 0;
  const char* ksplit = std::getenv("LING_KSPLIT");
  graphs_allowed_ = draft_ != nullptr && !profile_ && !(ksplit && std::atoi(ksplit) != 0);
  if (graphs && !graphs_allowed_) std::fprintf(stderr, "step graphs off: no drafter, or LING_PROFILE or LING_KSPLIT is set\n");
  graphs_ = graphs && graphs_allowed_;
  if (graphs_) std::fprintf(stderr, "step graphs on\n");
  bool pdl = opts_.pdl;
  if (const char* e = std::getenv("LING_PDL")) pdl = std::atoi(e) != 0;
  kernels::set_pdl(pdl);
  if (pdl) std::fprintf(stderr, "programmatic dependent launch on\n");
  bool bulk = opts_.attention_bulk;
  if (const char* e = std::getenv("LING_ATTN_BULK")) bulk = std::atoi(e) != 0;
  kernels::set_attention_bulk(bulk);
  if (bulk) std::fprintf(stderr, "attention: bulk-copy tiles\n");
  int prefetch = opts_.attention_prefetch;
  if (const char* e = std::getenv("LING_ATTN_PREFETCH")) prefetch = std::atoi(e);
  kernels::set_attention_prefetch(prefetch);
}

Engine::~Engine() {
  clear_graphs();
  if (pin_) cudaFreeHost(pin_);
  if (profile_) {
    for (const auto& [phase, sec] : phase_seconds_) std::fprintf(stderr, "profile %-24s %8.3f s\n", phase.c_str(), sec);
  }
  for (void* p : buffers_) cudaFree(p);
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Engine::reset() {
  reset_state();
  check(cudaStreamSynchronize(stream_), "reset");
  history_.clear();
  ckpts_.clear();
  snap_pos_ = -1;
  dkv_lo_ = 0;
  pos_ = 0;
}

void Engine::reset_state() {
  const ModelConfig& c = model_->config();
  for (float* s : gdn_state_)
    check(cudaMemsetAsync(s, 0, size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float), stream_), "reset");
  for (float* s : conv_state_)
    check(cudaMemsetAsync(s, 0, size_t(c.lin_conv_channels()) * (c.conv_kernel - 1) * sizeof(float), stream_),
          "reset");
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
  if (b != opts_.draft_block) clear_graphs();  // they were captured for the old block
  opts_.draft_block = b;
}

Engine::StateSlot Engine::alloc_slot() {
  const ModelConfig& c = model_->config();
  StateSlot s;
  for (size_t i = 0; i < gdn_state_.size(); ++i) {
    s.gdn.push_back(alloc<float>(size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim));
    s.conv.push_back(alloc<float>(size_t(c.lin_conv_channels()) * (c.conv_kernel - 1)));
  }
  return s;
}

void Engine::copy_state(const StateSlot& slot, bool save) {
  const ModelConfig& c = model_->config();
  const size_t gdn_bytes = size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float);
  const size_t conv_bytes = size_t(c.lin_conv_channels()) * (c.conv_kernel - 1) * sizeof(float);
  for (size_t i = 0; i < gdn_state_.size(); ++i) {
    check(cudaMemcpyAsync(save ? slot.gdn[i] : gdn_state_[i], save ? gdn_state_[i] : slot.gdn[i], gdn_bytes,
                          cudaMemcpyDeviceToDevice, stream_),
          "state copy");
    check(cudaMemcpyAsync(save ? slot.conv[i] : conv_state_[i], save ? conv_state_[i] : slot.conv[i], conv_bytes,
                          cudaMemcpyDeviceToDevice, stream_),
          "state copy");
  }
}

void Engine::truncate_history(int n) {
  if (n < static_cast<int>(history_.size())) history_.resize(n);
  ckpts_.erase(std::remove_if(ckpts_.begin(), ckpts_.end(), [&](const Checkpoint& k) { return k.pos > n; }), ckpts_.end());
  if (snap_pos_ > n) snap_pos_ = -1;
  dkv_lo_ = std::min(dkv_lo_, n);
}

void Engine::save_checkpoint(int pos) {
  for (const Checkpoint& k : ckpts_)
    if (k.pos == pos) return;
  int slot = -1;
  std::vector<bool> used(ckpt_slots_.size(), false);
  for (const Checkpoint& k : ckpts_) used[k.slot] = true;
  for (size_t i = 0; i < used.size(); ++i)
    if (!used[i]) slot = static_cast<int>(i);
  if (slot < 0) {  // full: the earliest positions are the ones other sessions share, so the latest gives way
    if (ckpts_.empty() || ckpts_.back().pos < pos) return;
    slot = ckpts_.back().slot;
    ckpts_.pop_back();
  }
  copy_state(ckpt_slots_[slot], true);
  ckpts_.push_back({pos, slot});
  std::sort(ckpts_.begin(), ckpts_.end(), [](const Checkpoint& a, const Checkpoint& b) { return a.pos < b.pos; });
}

void Engine::linear_fp4(const Fp4Weight& w, const float* x, int M, float* y, bool x_ready) {
  if (!quantized(M)) {  // the rows path: stream the weights once for every row
    if (!x_ready) kernels::to_half_rows(x, M, w.K, xh_, xinv_, stream_);
    kernels::stream_gemm_nvfp4(xh_, xinv_, M, w.w, nullptr, w.scale2, y, w.N, w.K, stream_);
    return;
  }
  kernels::quant_act_nvfp4(x, M, w.K, w.in_scale, xq_, xsf_, stream_);
  set_xq(w.in_scale, true);
  kernels::prefill_gemm_nvfp4(xq_, xsf_, M, w.w, w.in_scale * w.scale2, y, w.N, w.N, w.K, false, stream_);
}

void Engine::linear_fp4_multi(std::initializer_list<std::pair<const Fp4Weight*, float*>> ws, const float* x, int M,
                              bool x_ready) {
  if (quantized(M)) {  // one quantization of x for every matrix that shares its input scale
    for (const auto& [w, y] : ws) {
      if (!(x_ready && xq_is(w->in_scale, true))) kernels::quant_act_nvfp4(x, M, w->K, w->in_scale, xq_, xsf_, stream_);
      set_xq(w->in_scale, true);
      x_ready = true;
      kernels::prefill_gemm_nvfp4(xq_, xsf_, M, w->w, w->in_scale * w->scale2, y, w->N, w->N, w->K, false, stream_);
    }
    return;
  }
  kernels::StreamTarget t[4];
  int n = 0, K = 0;
  for (const auto& [w, y] : ws) {
    t[n++] = {w->w, nullptr, w->scale2, y, w->N};
    K = w->K;
  }
  if (!x_ready) kernels::to_half_rows(x, M, K, xh_, xinv_, stream_);
  kernels::stream_gemm_multi(true, xh_, xinv_, M, t, n, K, stream_);
}

void Engine::linear_fp8_multi(std::initializer_list<std::pair<const Fp8Weight*, float*>> ws, const float* x, int M,
                              bool x_ready) {
  if (quantized(M)) {
    for (const auto& [w, y] : ws) {
      if (!(x_ready && xq_is(w->in_scale, false))) kernels::quant_act_fp8(x, M, w->K, w->in_scale, xq_, stream_);
      set_xq(w->in_scale, false);
      x_ready = true;
      kernels::prefill_gemm_fp8(xq_, M, w->w, w->in_scale * w->scale, y, w->N, w->N, w->K, false, stream_);
    }
    return;
  }
  kernels::StreamTarget t[4];
  int n = 0, K = 0;
  for (const auto& [w, y] : ws) {
    t[n++] = {w->w, nullptr, w->scale, y, w->N};
    K = w->K;
  }
  if (!x_ready) kernels::to_half_rows(x, M, K, xh_, xinv_, stream_);
  kernels::stream_gemm_multi(false, xh_, xinv_, M, t, n, K, stream_);
}

void Engine::linear_fp8_residual(const Fp8Weight& w, const float* x, int M, bool x_ready) {
  const int H = model_->config().hidden;
  if (!quantized(M)) {
    linear_fp8(w, x, M, t1_);
    kernels::add_inplace(h_, t1_, M * H, stream_);
    return;
  }
  if (!(x_ready && xq_is(w.in_scale, false))) kernels::quant_act_fp8(x, M, w.K, w.in_scale, xq_, stream_);
  set_xq(w.in_scale, false);
  kernels::prefill_gemm_fp8(xq_, M, w.w, w.in_scale * w.scale, h_, H, w.N, w.K, true, stream_);
}

bool Engine::norm_rows(const float* x, const __nv_bfloat16* w, float* out, int M, float in_scale, bool nvfp4, bool f32) {
  const ModelConfig& c = model_->config();
  if (!quantized(M)) {
    kernels::rmsnorm_half(x, w, out, xh_, xinv_, M, c.hidden, c.eps, true, stream_);
    return true;
  }
  kernels::rmsnorm_quant(x, w, f32 ? out : nullptr, M, c.hidden, c.eps, nvfp4, in_scale, xq_, xsf_, stream_);
  set_xq(in_scale, nvfp4);
  return true;
}

void Engine::linear_bf16(const Bf16Weight& w, const float* x, int M, float* y) {
  if (M <= kernels::kMaxStreamRows) {  // the weights read once for every row, row-invariant
    kernels::bf16_rows(x, M, w.w, y, w.N, w.K, stream_);
    return;
  }
  // cuBLAS would need a preset workspace inside a graph capture; no step matrix takes this path.
  if (use_dev_pos_) throw std::runtime_error("cuBLAS inside a step graph");
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w.w, y, w.N, w.K);
}

void Engine::linear_bf16_cublas(const Bf16Weight& w, const float* x, int M, float* y) {
  if (M <= kernels::kMaxStreamRows && w.K % 1024 == 0) {  // stream the weights once (cuBLAS: ~90 GB/s here)
    kernels::bf16_rows(x, M, w.w, y, w.N, w.K, stream_);
    return;
  }
  // cuBLAS would need a preset workspace inside a graph capture; no step matrix takes this path.
  if (use_dev_pos_) throw std::runtime_error("cuBLAS inside a step graph");
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16_cublas(cublas_, x_bf16_, M, w.w, y, w.N, w.K);
}

void Engine::linear_fp8(const Fp8Weight& w, const float* x, int M, float* y, bool x_ready) {
  if (!quantized(M)) {
    if (!x_ready) kernels::to_half_rows(x, M, w.K, xh_, xinv_, stream_);
    kernels::stream_gemm_fp8(xh_, xinv_, M, w.w, w.scale, y, w.N, w.K, stream_);
    return;
  }
  kernels::quant_act_fp8(x, M, w.K, w.in_scale, xq_, stream_);
  set_xq(w.in_scale, false);
  kernels::prefill_gemm_fp8(xq_, M, w.w, w.in_scale * w.scale, y, w.N, w.N, w.K, false, stream_);
}

void Engine::forward(const int* ids, int M, Pass pass, int keep_at) {
  const ModelConfig& c = model_->config();
  if (pos_ + M > opts_.max_context) throw std::runtime_error("context is longer than max_context");
  const bool rows = M <= kernels::kMaxStreamRows, verify = pass == Pass::Verify;
  // A prompt's tokens take the prefill path whatever the chunk's size, so the state a prompt leaves does
  // not depend on how its prefill was split or where it resumed (the prefix checkpoints rely on it).
  quantized_ = pass == Pass::Prefill;
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
    // The prefill path quantizes the norm's output for the projections that follow; the DeltaNet's a and b
    // projections (BF16) also read it in FP32.
    // (FP32 too wherever the projections' input scales differ, so the others can quantize it themselves.)
    const bool xn_half =
        L.full ? norm_rows(h_, L.input_norm, xn_, M, L.q.in_scale, false,
                           L.k.in_scale != L.q.in_scale || L.v.in_scale != L.q.in_scale)
               : norm_rows(h_, L.input_norm, xn_, M, L.in_qkv.in_scale, false, true);
    if (L.full) {
      NvtxRange r("attention");
      linear_fp8_multi({{&L.q, q_gate_}, {&L.k, k_}, {&L.v, v_}}, xn_, M, xn_half);
      mark("attn_proj");
      kernels::attn_prepare(q_gate_, k_, v_, L.q_norm, L.k_norm, dpos(), M, c.heads, c.kv_heads, c.head_dim,
                            c.rotary_dim, c.rope_theta, c.eps, q_, gate_, kcache_[slot], vcache_[slot], stream_);
      // Fixed key ranges at fixed positions for every pass (row-invariant), in slices of queries that
      // bound the partial results' scratch. A captured graph launches the ranges of the whole context.
      for (int q0 = 0; q0 < M; q0 += kAttnSlice) {
        const int n = std::min(kAttnSlice, M - q0);
        kernels::attention_rows(q_ + size_t(q0) * qsize, kcache_[slot], vcache_[slot], dpos(q0), n, c.heads,
                                c.kv_heads, c.head_dim, attn_scratch_, attn_ + size_t(q0) * qsize, stream_,
                                use_dev_pos_ ? opts_.max_context : 0);
      }
      if (quantized_) {
        kernels::sigmoid_mul_fp8(attn_, gate_, size_t(M) * qsize, L.o.in_scale, xq_, stream_);
        set_xq(L.o.in_scale, false);
      } else {
        kernels::sigmoid_mul(attn_, gate_, M * qsize, stream_);
      }
      mark("attn_core");
      linear_fp8_residual(L.o, attn_, M, quantized_);
      mark("attn_proj");
    } else {
      NvtxRange r("deltanet");
      // A verify keeps this layer's inputs for commit() and leaves the conv and recurrent state as they
      // were: the conv runs on a copy of its window, the recurrence reads the state and does not write it.
      float* mixed = verify ? v_post_[slot] : mixed_;
      float* g = verify ? v_g_[slot] : g_;
      float* beta = verify ? v_beta_[slot] : beta_;
      // The prefill path convolves out of place: the projection goes to t2_ (free until the MLP).
      linear_fp8_multi({{&L.in_qkv, verify ? v_pre_[slot] : quantized_ ? t2_ : mixed}, {&L.in_z, z_}}, xn_, M, xn_half);
      if (quantized_) {  // tensor cores, row-invariant for any M
        kernels::narrow_bf16(xn_, M, L.in_a.w, L.in_b.w, a_, b_, L.in_a.N, L.in_a.K, stream_);
      } else if (L.in_a.N == L.in_b.N) {  // both in one launch
        kernels::bf16_rows(xn_, M, L.in_a.w, a_, L.in_a.N, L.in_a.K, stream_, L.in_b.w, b_);
      } else {
        linear_bf16(L.in_a, xn_, M, a_);
        linear_bf16(L.in_b, xn_, M, b_);
      }
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
      const bool keep = quantized_ && keep_at > 0 && keep_at < M;
      if (quantized_)
        kernels::gdn_conv_prefill(t2_, mixed, conv_state, L.conv, M, C, stream_, keep ? snap_.conv[slot] : nullptr,
                                  keep_at);
      else kernels::gdn_conv(mixed, conv_state, L.conv, M, C, stream_);
      kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g, beta, M, c.lin_v_heads, stream_);
      mark("gdn_conv");
      if (quantized_)
        kernels::gdn_recurrent_prefill(mixed, g, beta, gdn_state_[slot], core_, M, c.lin_k_heads, c.lin_v_heads, pos_,
                                       stream_, keep ? snap_.gdn[slot] : nullptr, keep_at);
      else
        kernels::gdn_recurrent(mixed, g, beta, gdn_state_[slot], verify ? nullptr : gdn_state_[slot], core_, M,
                               c.lin_k_heads, c.lin_v_heads, stream_);
      mark("gdn_recurrent");
      if (quantized_) {
        kernels::gated_rmsnorm_fp8(core_, z_, L.lin_norm, M * c.lin_v_heads, c.lin_v_dim, c.eps, L.out.in_scale, xq_,
                                   stream_);
        set_xq(L.out.in_scale, false);
      } else {
        kernels::gated_rmsnorm(core_, z_, L.lin_norm, normed_, M * c.lin_v_heads, c.lin_v_dim, c.eps, stream_);
      }
      linear_fp8_residual(L.out, normed_, M, quantized_);
      mark("gdn_proj");
    }
    {
      NvtxRange r("mlp");
      const bool post_half = norm_rows(h_, L.post_norm, xn_, M, L.gate.in_scale, true, L.up.in_scale != L.gate.in_scale);
      if (!quantized_) {
        linear_fp4_multi({{&L.gate, t1_}, {&L.up, t2_}}, xn_, M, post_half);
        kernels::silu_mul(t1_, t2_, t1_, M * I, stream_);
        linear_fp4(L.down, t1_, M, t2_);
        kernels::add_inplace(h_, t2_, M * H, stream_);
      } else if (L.gate.in_scale == L.up.in_scale && xq_is(L.gate.in_scale, true)) {
        // Gate and up in one GEMM whose epilogue writes the down projection's NVFP4 input; the down
        // projection adds into the residual.
        kernels::prefill_swiglu_nvfp4(xq_, xsf_, M, L.gate.w, L.gate.in_scale * L.gate.scale2, L.up.w,
                                      L.up.in_scale * L.up.scale2, I, H, L.down.in_scale, xq2_, xsf2_, stream_);
        kernels::prefill_gemm_nvfp4(xq2_, xsf2_, M, L.down.w, L.down.in_scale * L.down.scale2, h_, H, H, I, true,
                                    stream_);
      } else {  // gate and up quantize differently: separate GEMMs, then silu(gate) * up straight into NVFP4
        linear_fp4_multi({{&L.gate, t1_}, {&L.up, t2_}}, xn_, M, post_half);
        kernels::silu_mul_quant_nvfp4(t1_, t2_, M, I, L.down.in_scale, xq_, xsf_, stream_);
        set_xq(0.f, true);  // not a plain quantization of any one input
        kernels::prefill_gemm_nvfp4(xq_, xsf_, M, L.down.w, L.down.in_scale * L.down.scale2, h_, H, H, I, true, stream_);
      }
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
  const int L = static_cast<int>(prompt.size()), H = static_cast<int>(history_.size());
  // The caches hold the KV of history_; DeltaNet and conv state exist only where a state was kept: the
  // end of the last prompt (snap_) and the prefix checkpoints. The
  // prompt resumes from the furthest of these inside its common prefix with history_, so an agent's next
  // turn (which extends its previous prompt) prefills only its new tokens, and a new session whose
  // system message (instructions and tool schemas) another session already processed starts after it.
  int lcp = 0;
  while (lcp < std::min(H, L) && history_[lcp] == prompt[lcp]) ++lcp;
  // The drafter attends to the last `window` positions: resuming at p > window start needs its context
  // KV from the window start up to p, which is valid only from dkv_lo_ on.
  const int window = draft_ ? std::max(0, L - draft_->config().sliding_window) : 0;
  auto usable = [&](int p) { return p <= lcp && p < L && (!draft_ || p <= window || dkv_lo_ <= window); };
  // Only states a prefill left count, never the current one after decoding: a decoded token went through
  // the rows path, and a prompt's state must be the same however it was reached.
  int reuse = 0;
  const StateSlot* from = nullptr;
  if (snap_pos_ > 0 && usable(snap_pos_)) {
    reuse = snap_pos_;
    from = &snap_;
  }
  for (const Checkpoint& k : ckpts_)
    if (usable(k.pos) && k.pos > reuse) {
      reuse = k.pos;
      from = &ckpt_slots_[k.slot];
    }
  if (from) copy_state(*from, false);
  else reset_state();
  pos_ = reuse;
  truncate_history(reuse);
  if (draft_ && window > reuse) dkv_lo_ = window;

  // Chunk ends: every prefill_chunk tokens, and the checkpoint positions (the end of each of the first
  // two messages, and chunk multiples up to checkpoint_limit), where the state is kept for later prompts.
  // Every kept state sits at a multiple of kResumeAlign: the prefill DeltaNet runs in 32-token chunks at
  // absolute positions, so a prompt resumed there computes exactly what a prefill from scratch does.
  auto align = [](int p) { return p - p % kResumeAlign; };
  std::vector<int> bounds;
  if (!ckpt_slots_.empty()) {
    int messages = 0;
    for (int i = 0; i < L && i < opts_.checkpoint_limit && messages < 2; ++i)
      if (prompt[i] == opts_.boundary_token) {
        if (align(i + 1) > 0) bounds.push_back(align(i + 1));
        ++messages;
      }
    for (int b = opts_.prefill_chunk; b < std::min(L, opts_.checkpoint_limit + 1); b += opts_.prefill_chunk)
      bounds.push_back(b);
    std::sort(bounds.begin(), bounds.end());
  }
  const int snap_at = align(L);  // the end-of-prompt state is kept here (the tokens after it are re-prefilled)
  for (int i = reuse; i < L;) {
    int end = std::min(i + opts_.prefill_chunk, L);
    for (int b : bounds)
      if (b > i && b < end) {
        end = b;
        break;
      }
    // The end-of-prompt state at snap_at < end is captured inside the pass (no separate pass for the tail).
    const int keep_at = snap_at > i && snap_at < end ? snap_at - i : -1;
    if (keep_at > 0) snap_pos_ = -1;  // snap_ is rewritten by the pass
    const int n = end - i, pos0 = pos_;
    forward(prompt.data() + i, n, Pass::Prefill, keep_at);
    // A pass whose logits are not finite left a broken state (a NaN anywhere in it reaches the last
    // token's logits). Keep nothing from it: snap_ stays invalid if the pass wrote it, no checkpoint is
    // saved and its tokens do not join history_. A non-finite checkpoint would be reused by every later
    // prompt that shares the prefix. (A decode step's state is never kept, so only prefill needs this.)
    if (!all_finite(logits_)) {
      pos_ = i;
      throw NonFiniteLogits();
    }
    if (keep_at > 0) snap_pos_ = snap_at;
    if (draft_) {
      const int first = std::max(pos0, window);
      if (first < pos0 + n) draft_materialize(pos0 + n - first, first, first - pos0);  // still quantized_
    }
    history_.insert(history_.end(), prompt.begin() + i, prompt.begin() + end);
    i = end;
    if (end == snap_at) {
      copy_state(snap_, true);
      snap_pos_ = snap_at;
    }
    if (end < L && std::binary_search(bounds.begin(), bounds.end(), end)) save_checkpoint(end);
  }
  check(cudaStreamSynchronize(stream_), "prefill");
  prompt_len_ = L;
  if (stats) {
    stats->reused_tokens = reuse;
    stats->prefill_tokens = L - reuse;
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
  if (p.seed != 0 && p.temperature > 0.f) rng_.seed(p.seed + history.size());
  const size_t output = std::min(history.size(), prompt_len_);  // where the request's output starts
  return sample_logits(logits, p, std::span<const int>(history).subspan(output), rng_);
}

}  // namespace ling
