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
  // --prefill-bf16's tensors live in the FP32 buffers they replace (each BF16 [M][n] fits its FP32 [M][n]).
  xn_bf16_ = reinterpret_cast<__nv_bfloat16*>(xn_);
  t2_bf16_ = reinterpret_cast<__nv_bfloat16*>(t2_);
  z_bf16_ = reinterpret_cast<__nv_bfloat16*>(z_);
  mixed_bf16_ = reinterpret_cast<__nv_bfloat16*>(mixed_);
  core_bf16_ = reinterpret_cast<__nv_bfloat16*>(core_);
  ids_dev_ = alloc<int>(M);
  pos_dev_ = alloc<int>(kernels::kMaxSeqs);
  check(cudaHostAlloc(reinterpret_cast<void**>(&pin_), sizeof(StepPinned), cudaHostAllocDefault), "pinned step buffers");
  kernels::prepare_kernels();  // attributes are set here, never inside a graph capture

  // Head-major KV caches: one KV head's keys are one contiguous stream for the attention kernels.
  kernels::set_kv_layout(size_t(opts_.max_context) * c.head_dim, c.head_dim);
  // The session slots (M4 §1), each allocated in full here: nothing is allocated while serving.
  if (opts_.sessions < 1) throw std::runtime_error("sessions must be at least 1");
  sessions_.resize(opts_.sessions);
  layer_slot_.resize(c.layers);
  int n_full = 0;
  for (int i = 0; i < c.layers; ++i) layer_slot_[i] = c.full_attention[i] ? n_full++ : n_linear_++;
  std::random_device seed_source;
  for (Session& s : sessions_) {
    for (int i = 0; i < n_full; ++i) {
      s.kcache.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * kvsize));
      s.vcache.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * kvsize));
    }
    for (int i = 0; i < n_linear_; ++i) {
      s.gdn.push_back(alloc<float>(size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim));
      s.conv.push_back(alloc<float>(size_t(C) * (c.conv_kernel - 1)));
    }
    s.snap = alloc_slot();
    s.rng.seed(seed_source());
  }
  s_ = &sessions_[0];
  for (int i = 0; i < opts_.prefix_checkpoints; ++i) ckpt_slots_.push_back(alloc_slot());
  if (!opts_.draft_dir.empty()) {
    draft_ = std::make_unique<DraftModel>(opts_.draft_dir);
    alloc_drafter();
    std::fprintf(stderr, "drafter: %.2f GB, block %d\n", draft_->device_bytes() / 1e9, opts_.draft_block);
  }
  if (sessions_.size() > 1) std::fprintf(stderr, "sessions: %zu resident\n", sessions_.size());
  logits_.resize(c.vocab);
  profile_ = std::getenv("LING_PROFILE") != nullptr;
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
  // B34: BF16 DeltaNet intermediates in the prefill (not exact; off by default, MILESTONES §14).
  prefill_bf16_ = opts_.prefill_bf16;
  if (const char* e = std::getenv("LING_PREFILL_BF16")) prefill_bf16_ = std::atoi(e) != 0;
  if (prefill_bf16_) std::fprintf(stderr, "prefill: BF16 DeltaNet intermediates (not the exact path)\n");
  commit_overlap_ = opts_.commit_overlap && draft_ != nullptr;
  if (const char* e = std::getenv("LING_COMMIT_OVERLAP")) commit_overlap_ = draft_ != nullptr && std::atoi(e) != 0;
  if (commit_overlap_) {
    check(cudaStreamCreateWithFlags(&commit_stream_, cudaStreamNonBlocking), "commit stream");
    check(cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming), "commit event");
    commit_core_ = alloc<float>(size_t(kernels::kMaxStreamRows) * Vs);
    std::fprintf(stderr, "commit beside the next draft\n");
  }
}

void Engine::join_commit() {
  if (!commit_pending_) return;
  // A wait on an event recorded outside a capture cannot be captured: every captured body runs with nothing
  // pending (speculate joins before the verify; the draft and the drafter's context KV never join).
  if (use_dev_pos_) throw std::runtime_error("join_commit inside a step graph's capture");
  check(cudaStreamWaitEvent(stream_, commit_done_, 0), "join commit");
  commit_pending_ = false;
}

Engine::~Engine() {
  if (commit_stream_) cudaStreamSynchronize(commit_stream_);
  clear_graphs();
  if (pin_) cudaFreeHost(pin_);
  if (profile_) {
    for (const auto& [phase, sec] : phase_seconds_) std::fprintf(stderr, "profile %-24s %8.3f s\n", phase.c_str(), sec);
  }
  for (void* p : buffers_) cudaFree(p);
  if (cublas_) cublasDestroy(cublas_);
  if (commit_done_) cudaEventDestroy(commit_done_);
  if (commit_stream_) cudaStreamDestroy(commit_stream_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Engine::reset() {
  reset_session();
  ckpts_.clear();
}

void Engine::reset_session() {
  reset_state();
  check(cudaStreamSynchronize(stream_), "reset");
  s_->history.clear();
  s_->prompt_len = 0;
  s_->turn_start = 0;
  s_->snap_pos = -1;
  s_->dkv_lo = 0;
  s_->pos = 0;
  s_->shadow.clear();
}

int Engine::match_session(const std::vector<int>& prompt, const std::string& key) const {
  int best = -1, best_lcp = 0;
  for (size_t i = 0; i < sessions_.size(); ++i) {
    const Session& s = sessions_[i];
    if (!s.owned || s.history.empty() || s.prompt_len == 0) continue;
    const size_t n = std::min(s.history.size(), prompt.size());
    size_t lcp = 0;
    while (lcp < n && s.history[lcp] == prompt[lcp]) ++lcp;
    if (static_cast<int>(lcp) < s.turn_start) continue;  // merely overlapping (a shared system prompt)
    const bool keyed = !key.empty() && s.key == key;
    if (static_cast<int>(lcp) > best_lcp || (static_cast<int>(lcp) == best_lcp && keyed)) {
      best = static_cast<int>(i);
      best_lcp = static_cast<int>(lcp);
    }
  }
  if (best >= 0 || key.empty()) return best;
  for (size_t i = 0; i < sessions_.size(); ++i)
    if (sessions_[i].owned && sessions_[i].key == key) return static_cast<int>(i);
  return -1;
}

int Engine::free_session(double protect_seconds, const std::vector<bool>& pinned) const {
  auto is_pinned = [&](size_t i) { return i < pinned.size() && pinned[i]; };
  for (size_t i = 0; i < sessions_.size(); ++i)
    if (!sessions_[i].owned && !is_pinned(i)) return static_cast<int>(i);
  const auto now = std::chrono::steady_clock::now();
  int best = -1;
  for (size_t i = 0; i < sessions_.size(); ++i) {
    const Session& s = sessions_[i];
    if (is_pinned(i)) continue;
    if (!s.key.empty() && std::chrono::duration<double>(now - s.last_used).count() < protect_seconds) continue;
    if (best < 0 || s.last_used < sessions_[best].last_used) best = static_cast<int>(i);
  }
  return best;
}

void Engine::activate_session(int i, const std::string& key, bool fresh) {
  if (i < 0 || i >= static_cast<int>(sessions_.size())) throw std::runtime_error("no such session");
  s_ = &sessions_[i];
  if (fresh) {
    if (s_->owned) ++evictions_;
    s_->owned = true;
    s_->key = key;
    s_->shadow.clear();  // the lookup's records belong to the previous session's output
  } else if (!key.empty()) {
    s_->key = key;
  }
  touch_session();
}

void Engine::touch_session() { s_->last_used = std::chrono::steady_clock::now(); }

void Engine::touch_session(int i) { sessions_.at(i).last_used = std::chrono::steady_clock::now(); }

void Engine::select_session(int i) {
  if (i < 0 || i >= static_cast<int>(sessions_.size())) throw std::runtime_error("no such session");
  s_ = &sessions_[i];
}

void Engine::reset_all() {
  for (Session& s : sessions_) {
    s_ = &s;
    reset_session();
    s.spec = SpecStats{};
  }
  s_ = &sessions_[0];
  ckpts_.clear();
}

const SpecStats& Engine::session_spec_stats(int i) {
  Session& s = sessions_.at(i);
  score_shadow(s, true);
  return s.spec;
}

SpecStats Engine::spec_stats_total() const {
  SpecStats t;
  for (const Session& s : sessions_) {
    const SpecStats& a = s.spec;
    t.steps += a.steps;
    t.drafted += a.drafted;
    t.accepted += a.accepted;
    for (size_t k = 0; k < t.accept_hist.size() && k < a.accept_hist.size(); ++k) t.accept_hist[k] += a.accept_hist[k];
    t.draft_seconds += a.draft_seconds;
    t.verify_seconds += a.verify_seconds;
    t.commit_seconds += a.commit_seconds;
    t.lookup_steps += a.lookup_steps;
    t.lookup_accepted += a.lookup_accepted;
    t.copy_steps += a.copy_steps;
    t.copy_drafted += a.copy_drafted;
    t.copy_accepted += a.copy_accepted;
    t.batched_steps += a.batched_steps;
    for (int b = 0; b < 5; ++b) {
      t.shadow_steps[b] += a.shadow_steps[b];
      t.shadow_lookup[b] += a.shadow_lookup[b];
      t.shadow_dflash[b] += a.shadow_dflash[b];
    }
    t.shadow_best += a.shadow_best;
    t.shadow_total_steps += a.shadow_total_steps;
    t.shadow_dflash_all += a.shadow_dflash_all;
  }
  return t;
}

int Engine::sessions_resident() const {
  int n = 0;
  for (const Session& s : sessions_) n += s.owned;
  return n;
}

void Engine::adopt_prefix(const Session& from, int n) {
  join_commit();
  const ModelConfig& c = model_->config();
  // Head-major caches: each KV head's positions [0, n) are one contiguous run, max_context positions apart.
  const size_t row = size_t(c.head_dim) * sizeof(__nv_bfloat16), pitch = size_t(opts_.max_context) * row;
  for (size_t l = 0; l < s_->kcache.size(); ++l) {
    check(cudaMemcpy2DAsync(s_->kcache[l], pitch, from.kcache[l], pitch, size_t(n) * row, c.kv_heads,
                            cudaMemcpyDeviceToDevice, stream_),
          "session KV copy");
    check(cudaMemcpy2DAsync(s_->vcache[l], pitch, from.vcache[l], pitch, size_t(n) * row, c.kv_heads,
                            cudaMemcpyDeviceToDevice, stream_),
          "session KV copy");
  }
  if (draft_) {
    const DraftConfig& d = draft_->config();
    const size_t ring_bytes = size_t(ring_) * d.kv_heads * d.head_dim * sizeof(__nv_bfloat16);
    for (size_t l = 0; l < s_->dkc.size(); ++l) {
      check(cudaMemcpyAsync(s_->dkc[l], from.dkc[l], ring_bytes, cudaMemcpyDeviceToDevice, stream_), "ring copy");
      check(cudaMemcpyAsync(s_->dvc[l], from.dvc[l], ring_bytes, cudaMemcpyDeviceToDevice, stream_), "ring copy");
    }
  }
  // The copied ring holds what `from`'s holds: positions [dkv_from(from), from.history.size()); the
  // truncation below keeps the part before n.
  s_->history = from.history;
  s_->dkv_lo = dkv_from(from);
  s_->snap_pos = -1;  // the active session's own end-of-prompt state belongs to the history it replaced
  s_->shadow.clear();
  truncate_history(n);
}

void Engine::reset_state() {
  join_commit();
  const ModelConfig& c = model_->config();
  for (float* s : s_->gdn)
    check(cudaMemsetAsync(s, 0, size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float), stream_), "reset");
  for (float* s : s_->conv)
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
  join_commit();
  check(cudaStreamSynchronize(stream_), "state_hash");
  for (size_t i = 0; i < s_->gdn.size(); ++i) {
    mix(s_->gdn[i], gdn);
    mix(s_->conv[i], conv);
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
  for (int i = 0; i < n_linear_; ++i) {
    s.gdn.push_back(alloc<float>(size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim));
    s.conv.push_back(alloc<float>(size_t(c.lin_conv_channels()) * (c.conv_kernel - 1)));
  }
  return s;
}

void Engine::copy_state(const StateSlot& slot, bool save) {
  join_commit();
  const ModelConfig& c = model_->config();
  const size_t gdn_bytes = size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float);
  const size_t conv_bytes = size_t(c.lin_conv_channels()) * (c.conv_kernel - 1) * sizeof(float);
  for (size_t i = 0; i < s_->gdn.size(); ++i) {
    check(cudaMemcpyAsync(save ? slot.gdn[i] : s_->gdn[i], save ? s_->gdn[i] : slot.gdn[i], gdn_bytes,
                          cudaMemcpyDeviceToDevice, stream_),
          "state copy");
    check(cudaMemcpyAsync(save ? slot.conv[i] : s_->conv[i], save ? s_->conv[i] : slot.conv[i], conv_bytes,
                          cudaMemcpyDeviceToDevice, stream_),
          "state copy");
  }
}

void Engine::truncate_history(int n) {
  // The drafter ring keeps only the last ring_ positions written, which may lie past n: fix the valid range
  // before the history shrinks (positions [dkv_from, n) stay, the ones after n are rewritten later).
  s_->dkv_lo = std::min(dkv_from(*s_), n);
  if (n < static_cast<int>(s_->history.size())) s_->history.resize(n);
  if (s_->snap_pos > n) s_->snap_pos = -1;
  // The checkpoint store keeps its entries: each carries its own token prefix (M4 §1).
}

namespace {
// FNV-1a over token ids: the checkpoint store's prefilter (a hit is confirmed by comparing the ids).
constexpr uint64_t kFnvBasis = 1469598103934665603ull, kFnvPrime = 1099511628211ull;
uint64_t fnv_step(uint64_t h, int token) {
  for (int b = 0; b < 4; ++b) h = (h ^ ((static_cast<uint32_t>(token) >> (8 * b)) & 0xff)) * kFnvPrime;
  return h;
}
uint64_t fnv_tokens(const int* t, size_t n) {
  uint64_t h = kFnvBasis;
  for (size_t i = 0; i < n; ++i) h = fnv_step(h, t[i]);
  return h;
}
}  // namespace

bool Engine::ckpt_match(const Checkpoint& k, const std::vector<int>& prompt,
                        const std::vector<uint64_t>& prefix_hash) const {
  if (k.pos > static_cast<int>(prompt.size()) || k.pos >= static_cast<int>(prefix_hash.size())) return false;
  if (prefix_hash[k.pos] != k.hash) return false;
  return std::equal(k.tokens.begin(), k.tokens.end(), prompt.begin());
}

void Engine::save_checkpoint(int pos, uint64_t epoch) {
  const uint64_t hash = fnv_tokens(s_->history.data(), pos);
  for (Checkpoint& k : ckpts_)
    if (k.pos == pos && k.hash == hash && std::equal(k.tokens.begin(), k.tokens.end(), s_->history.begin())) {
      k.used = ++ckpt_clock_;
      return;
    }
  int slot = -1;
  std::vector<bool> used(ckpt_slots_.size(), false);
  for (const Checkpoint& k : ckpts_) used[k.slot] = true;
  for (size_t i = 0; i < used.size(); ++i)
    if (!used[i]) slot = static_cast<int>(i);
  if (slot < 0) {
    // Full: the least recently used entry gives way (the owner's decision of 2026-10-12; stage 1 dropped the
    // latest position, so a store full of other prompts' earlier entries saved nothing new). Entries on this
    // prompt's own prefix were stamped at or after `epoch` and stay: a prompt longer than the store keeps its
    // earliest positions, which the sessions that share its system prompt resume from.
    int victim = -1;
    for (size_t i = 0; i < ckpts_.size(); ++i)
      if (ckpts_[i].used < epoch && (victim < 0 || ckpts_[i].used < ckpts_[victim].used)) victim = static_cast<int>(i);
    if (victim < 0) return;
    slot = ckpts_[victim].slot;
    ckpts_.erase(ckpts_.begin() + victim);
  }
  // Stream-ordered: the copy completes before any later restore reads the slot (one stream, M4 §1).
  copy_state(ckpt_slots_[slot], true);
  ckpts_.push_back({pos, slot, hash, std::vector<int>(s_->history.begin(), s_->history.begin() + pos), ++ckpt_clock_});
  std::sort(ckpts_.begin(), ckpts_.end(), [](const Checkpoint& a, const Checkpoint& b) { return a.pos < b.pos; });
}

void Engine::linear_fp4(const Fp4Weight& w, const float* x, int M, float* y, bool x_ready) {
  if (!quantized(M)) {  // the rows path: stream the weights once for every row
    if (!x_ready) kernels::to_half_rows(x, M, w.K, xh_, xinv_, stream_);
    kernels::stream_gemm_nvfp4(xh_, xinv_, M, w.w, w.table, w.scale2, y, w.N, w.K, stream_);
    return;
  }
  kernels::quant_act_nvfp4(x, M, w.K, w.in_scale, xq_, xsf_, stream_);
  set_xq(w.in_scale, true);
  kernels::prefill_gemm_nvfp4(xq_, xsf_, M, w.w, w.in_scale * w.scale2, y, w.N, w.N, w.K, false, stream_, w.scales);
}

void Engine::linear_fp4_multi(std::initializer_list<std::pair<const Fp4Weight*, float*>> ws, const float* x, int M,
                              bool x_ready) {
  if (quantized(M)) {  // one quantization of x for every matrix that shares its input scale
    for (const auto& [w, y] : ws) {
      if (!(x_ready && xq_is(w->in_scale, true))) kernels::quant_act_nvfp4(x, M, w->K, w->in_scale, xq_, xsf_, stream_);
      set_xq(w->in_scale, true);
      x_ready = true;
      kernels::prefill_gemm_nvfp4(xq_, xsf_, M, w->w, w->in_scale * w->scale2, y, w->N, w->N, w->K, false, stream_,
                                  w->scales);
    }
    return;
  }
  kernels::StreamTarget t[4];
  int n = 0, K = 0;
  for (const auto& [w, y] : ws) {
    t[n++] = {w->w, w->table, w->scale2, y, w->N};
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
  const Seg one{s_, 0, M};
  forward_rows(ids, M, pass, keep_at, std::span<const Seg>(&one, 1));
}

void Engine::forward_rows(const int* ids, int M, Pass pass, int keep_at, std::span<const Seg> segs) {
  join_commit();  // a no-op inside a verify's capture: speculate joins before the verify
  const ModelConfig& c = model_->config();
  const bool rows = M <= kernels::kMaxStreamRows, verify = pass == Pass::Verify;
  if (segs.empty() || (segs.size() > 1 && !verify)) throw std::runtime_error("forward: several sequences only in a verify");
  int seg_rows = 0;
  for (const Seg& g : segs) {
    if (g.row0 != seg_rows || g.rows < 1) throw std::runtime_error("forward: segments must tile the rows in order");
    if (g.s->pos + g.rows > opts_.max_context) throw std::runtime_error("context is longer than max_context");
    seg_rows += g.rows;
  }
  if (seg_rows != M) throw std::runtime_error("forward: segments must tile the rows");
  // A segment's position. A captured graph (one sequence, the active one) reads it from pos_dev_ (dpos); the
  // batch path runs eagerly at each session's own position.
  auto seg_pos = [&](const Seg& g, int off) { return use_dev_pos_ ? dpos(off) : kernels::DevPos(g.s->pos + off); };
  // Several sequences (M4 stage 4): the state kernels take them all in one launch, each row at its sequence's
  // position and in its sequence's caches and state; a captured graph reads the positions from pos_dev_[i].
  const bool multi = segs.size() > 1;
  if (segs.size() > static_cast<size_t>(kernels::kMaxSeqs)) throw std::runtime_error("forward: too many sequences");
  kernels::SeqBatch sb;
  sb.n = static_cast<int>(segs.size());
  for (int i = 0; i < sb.n; ++i) {
    sb.row0[i + 1] = segs[i].row0 + segs[i].rows;
    sb.pos[i] = segs[i].s->pos;
  }
  if (use_dev_pos_) sb.pos_dev = pos_dev_;
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
    // --prefill-bf16 (B34): a DeltaNet layer of a prefill takes the BF16 branch below, which runs its own norm.
    const bool bf16_gdn = !L.full && quantized_ && prefill_bf16_;
    // The prefill path quantizes the norm's output for the projections that follow; the DeltaNet's a and b
    // projections (BF16) also read it in FP32.
    // (FP32 too wherever the projections' input scales differ, so the others can quantize it themselves.)
    const bool xn_half =
        bf16_gdn ? false
        : L.full ? norm_rows(h_, L.input_norm, xn_, M, L.q.in_scale, false,
                             L.k.in_scale != L.q.in_scale || L.v.in_scale != L.q.in_scale)
                 : norm_rows(h_, L.input_norm, xn_, M, L.in_qkv.in_scale, false, true);
    if (L.full) {
      NvtxRange r("attention");
      linear_fp8_multi({{&L.q, q_gate_}, {&L.k, k_}, {&L.v, v_}}, xn_, M, xn_half);
      mark("attn_proj");
      // Per sequence (M4 §2): its rows' q/k norms and RoPE at its positions, their KV into its cache, and their
      // attention over it.
      const int kvsize = c.kv_heads * c.head_dim;
      if (multi) {
        kernels::SeqPtrs<__nv_bfloat16> kc, vc;
        kernels::SeqPtrs<const __nv_bfloat16> kcr, vcr;
        for (int i = 0; i < sb.n; ++i) {
          kcr.p[i] = kc.p[i] = segs[i].s->kcache[slot];
          vcr.p[i] = vc.p[i] = segs[i].s->vcache[slot];
        }
        kernels::attn_prepare_seqs(q_gate_, k_, v_, L.q_norm, L.k_norm, sb, c.heads, c.kv_heads, c.head_dim, c.rotary_dim,
                                   c.rope_theta, c.eps, q_, gate_, kc, vc, stream_);
        kernels::attention_rows_seqs(q_, kcr, vcr, sb, c.heads, c.kv_heads, c.head_dim, attn_scratch_, attn_, stream_,
                                     use_dev_pos_ ? opts_.max_context : 0);
      }
      for (const Seg& g : multi ? std::span<const Seg>() : segs) {
        const size_t r0 = g.row0;
        kernels::attn_prepare(q_gate_ + r0 * 2 * qsize, k_ + r0 * kvsize, v_ + r0 * kvsize, L.q_norm, L.k_norm,
                              seg_pos(g, 0), g.rows, c.heads, c.kv_heads, c.head_dim, c.rotary_dim, c.rope_theta, c.eps,
                              q_ + r0 * qsize, gate_ + r0 * qsize, g.s->kcache[slot], g.s->vcache[slot], stream_);
        // Fixed key ranges at fixed positions for every pass (row-invariant), in slices of queries that
        // bound the partial results' scratch. A captured graph launches the ranges of the whole context.
        for (int q0 = 0; q0 < g.rows; q0 += kAttnSlice) {
          const int n = std::min(kAttnSlice, g.rows - q0);
          kernels::attention_rows(q_ + (r0 + q0) * qsize, g.s->kcache[slot], g.s->vcache[slot], seg_pos(g, q0), n,
                                  c.heads, c.kv_heads, c.head_dim, attn_scratch_, attn_ + (r0 + q0) * qsize, stream_,
                                  use_dev_pos_ ? opts_.max_context : 0);
        }
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
    } else if (bf16_gdn) {
      // --prefill-bf16 (B34; off by default under MILESTONES §14's exactness policy): the DeltaNet layer of a
      // prefill with its intermediates in BF16 where production keeps BF16 (reports/M2-prefill.md §8 item 3,
      // cited as §7.3 in the backlog's B34 row), through the BF16 instantiations of the same kernels; the exact
      // FP32 branch below is untouched when the switch is off, and a verify never comes here. The tensors:
      //   xn_    the input norm's copy for the a/b projections (value-identical: narrow_bf16 rounds its FP32
      //          input to the same BF16 values; bandwidth only);
      //   t2_    the in_qkv projection, the conv's input and the source of its state window;
      //   z_     the in_z projection, the gated norm's gate;
      //   mixed_ the conv's output, the recurrence's input;
      //   core_  the recurrence's output, the gated norm's input.
      // The conv and recurrent state (and their checkpoints), a, b, g, beta and the residual stream stay FP32.
      NvtxRange r("deltanet");
      const int Vs = c.lin_value_size();
      kernels::rmsnorm_quant_bf16(h_, L.input_norm, xn_bf16_, M, H, c.eps, false, L.in_qkv.in_scale, xq_, xsf_, stream_);
      set_xq(L.in_qkv.in_scale, false);
      kernels::prefill_gemm_fp8_bf16out(xq_, M, L.in_qkv.w, L.in_qkv.in_scale * L.in_qkv.scale, t2_bf16_, C, C,
                                        L.in_qkv.K, stream_);
      if (!xq_is(L.in_z.in_scale, false)) {  // in_z quantizes differently (not in this checkpoint): from the copy
        kernels::quant_act_fp8_bf16in(xn_bf16_, M, L.in_z.K, L.in_z.in_scale, xq_, stream_);
        set_xq(L.in_z.in_scale, false);
      }
      kernels::prefill_gemm_fp8_bf16out(xq_, M, L.in_z.w, L.in_z.in_scale * L.in_z.scale, z_bf16_, Vs, Vs, L.in_z.K,
                                        stream_);
      kernels::narrow_bf16_bf16in(xn_bf16_, M, L.in_a.w, L.in_b.w, a_, b_, L.in_a.N, L.in_a.K, stream_);
      mark("gdn_proj");
      const bool keep = keep_at > 0 && keep_at < M;
      kernels::gdn_conv_prefill_bf16(t2_bf16_, mixed_bf16_, s_->conv[slot], L.conv, M, C, stream_,
                                     keep ? s_->snap.conv[slot] : nullptr, keep_at);
      kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g_, beta_, M, c.lin_v_heads, stream_);
      mark("gdn_conv");
      kernels::gdn_recurrent_prefill_bf16(mixed_bf16_, g_, beta_, s_->gdn[slot], core_bf16_, M, c.lin_k_heads,
                                          c.lin_v_heads, s_->pos, stream_, keep ? s_->snap.gdn[slot] : nullptr, keep_at);
      mark("gdn_recurrent");
      kernels::gated_rmsnorm_fp8_bf16(core_bf16_, z_bf16_, L.lin_norm, M * c.lin_v_heads, c.lin_v_dim, c.eps,
                                      L.out.in_scale, xq_, stream_);
      set_xq(L.out.in_scale, false);
      linear_fp8_residual(L.out, normed_, M, true);
      mark("gdn_proj");
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
      const bool keep = quantized_ && keep_at > 0 && keep_at < M;
      if (verify) {
        check(cudaMemcpyAsync(mixed, v_pre_[slot], size_t(M) * C * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
              "verify inputs");
        // Per sequence (M4 §2), from its own conv window, left unwritten.
        if (multi) {
          kernels::SeqPtrs<const float> st;
          for (int i = 0; i < sb.n; ++i) st.p[i] = segs[i].s->conv[slot];
          kernels::gdn_conv_seqs(mixed, st, L.conv, sb, C, stream_);
        }
        for (const Seg& sg : multi ? std::span<const Seg>() : segs) {
          check(cudaMemcpyAsync(conv_tmp_, sg.s->conv[slot], size_t(C) * (c.conv_kernel - 1) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_),
                "verify conv");
          kernels::gdn_conv(mixed + size_t(sg.row0) * C, conv_tmp_, L.conv, sg.rows, C, stream_);
        }
      } else if (quantized_) {
        kernels::gdn_conv_prefill(t2_, mixed, s_->conv[slot], L.conv, M, C, stream_,
                                  keep ? s_->snap.conv[slot] : nullptr, keep_at);
      } else {
        kernels::gdn_conv(mixed, s_->conv[slot], L.conv, M, C, stream_);
      }
      kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g, beta, M, c.lin_v_heads, stream_);
      mark("gdn_conv");
      if (quantized_)
        kernels::gdn_recurrent_prefill(mixed, g, beta, s_->gdn[slot], core_, M, c.lin_k_heads, c.lin_v_heads, s_->pos,
                                       stream_, keep ? s_->snap.gdn[slot] : nullptr, keep_at);
      else if (multi) {  // a verify: each sequence from its own state, left unwritten
        kernels::SeqPtrs<const float> st;
        for (int i = 0; i < sb.n; ++i) st.p[i] = segs[i].s->gdn[slot];
        kernels::gdn_recurrent_seqs(mixed, g, beta, st, core_, sb, c.lin_k_heads, c.lin_v_heads, stream_);
      } else
        for (const Seg& sg : segs) {  // per sequence, from its own state (a verify leaves it unwritten)
          const size_t r0 = sg.row0;
          kernels::gdn_recurrent(mixed + r0 * C, g + r0 * c.lin_v_heads, beta + r0 * c.lin_v_heads, sg.s->gdn[slot],
                                 verify ? nullptr : sg.s->gdn[slot], core_ + r0 * Vs, sg.rows, c.lin_k_heads,
                                 c.lin_v_heads, stream_);
        }
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
                                      L.up.in_scale * L.up.scale2, I, H, L.down.in_scale, xq2_, xsf2_, stream_,
                                      L.gate.scales, L.up.scales);
        kernels::prefill_gemm_nvfp4(xq2_, xsf2_, M, L.down.w, L.down.in_scale * L.down.scale2, h_, H, H, I, true,
                                    stream_, L.down.scales);
      } else {  // gate and up quantize differently: separate GEMMs, then silu(gate) * up straight into NVFP4
        linear_fp4_multi({{&L.gate, t1_}, {&L.up, t2_}}, xn_, M, post_half);
        kernels::silu_mul_quant_nvfp4(t1_, t2_, M, I, L.down.in_scale, xq_, xsf_, stream_);
        set_xq(0.f, true);  // not a plain quantization of any one input
        kernels::prefill_gemm_nvfp4(xq_, xsf_, M, L.down.w, L.down.in_scale * L.down.scale2, h_, H, H, I, true, stream_,
                                    L.down.scales);
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
  s_->pos += M;
}

Engine::PrefillCursor Engine::begin_prefill(const std::vector<int>& prompt, bool from_scratch) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");
  PrefillCursor cur;
  cur.t0 = std::chrono::steady_clock::now();
  cur.session = active_session();
  cur.prompt = prompt;
  const int L = static_cast<int>(prompt.size());
  // A session's caches hold the KV of its history; DeltaNet and conv state exist only where a state was
  // kept: the end of a session's last prompt (its snap) and the shared checkpoint store. The prompt resumes
  // from the furthest of these that lies inside the common prefix of the prompt with a session's history
  // (whose KV that session holds), so an agent's next turn (which extends its previous prompt) prefills only
  // its new tokens, and a new session whose system message (instructions and tool schemas) was already
  // processed starts after it. When the furthest such state is another session's, that session's KV for the
  // prefix is copied into the active one first (M4 §1); copies are exact, so the state is the same.
  // Another session may be mid-request (M4 §3): its history holds only tokens already processed, and its
  // snap and the store only states a finished pass left, so whatever is copied from it is complete.
  // The drafter attends to the last `window` positions: resuming at p > window start needs its context
  // KV from the window start up to p, which the session's ring holds only from dkv_from() on.
  const int window = draft_ ? std::max(0, L - draft_->config().sliding_window) : 0;
  std::vector<uint64_t> prefix_hash;  // of prompt[0, i), for the store's lookups (and its LRU stamps, below)
  if (!ckpts_.empty()) {
    const int n = std::min(L, opts_.checkpoint_limit);
    prefix_hash.resize(n + 1);
    prefix_hash[0] = kFnvBasis;
    for (int i = 0; i < n; ++i) prefix_hash[i + 1] = fnv_step(prefix_hash[i], prompt[i]);
  }
  struct Source {
    int reuse = 0;
    const StateSlot* from = nullptr;
  };
  // Only states a prefill left count, never a session's current one after decoding: a decoded token went
  // through the rows path, and a prompt's state must be the same however it was reached.
  auto best_in = [&](const Session& s) {
    const int n = std::min(static_cast<int>(s.history.size()), L);
    int lcp = 0;
    while (lcp < n && s.history[lcp] == prompt[lcp]) ++lcp;
    auto usable = [&](int p) { return p <= lcp && p < L && (!draft_ || p <= window || dkv_from(s) <= window); };
    Source r;
    if (s.snap_pos > 0 && usable(s.snap_pos)) r = {s.snap_pos, &s.snap};
    for (const Checkpoint& k : ckpts_)
      if (k.pos > r.reuse && usable(k.pos) && ckpt_match(k, prompt, prefix_hash)) r = {k.pos, &ckpt_slots_[k.slot]};
    return r;
  };
  Source src;
  const Session* donor = nullptr;
  if (!from_scratch) {
    src = best_in(*s_);
    for (const Session& s : sessions_) {
      if (&s == s_) continue;
      const Source d = best_in(s);
      if (d.reuse > src.reuse) {
        src = d;
        donor = &s;
      }
    }
  }
  // The store's clock (its LRU order): every entry on this prompt's prefix is used by this prefill, and the
  // entries it saves come after; save_checkpoint never evicts one of them for another.
  cur.ckpt_epoch = ++ckpt_clock_;
  if (!prefix_hash.empty())
    for (Checkpoint& k : ckpts_)
      if (ckpt_match(k, prompt, prefix_hash)) k.used = cur.ckpt_epoch;
  if (donor) adopt_prefix(*donor, src.reuse);
  if (src.from) copy_state(*src.from, false);
  else reset_state();
  const int reuse = src.reuse;
  s_->pos = reuse;
  truncate_history(reuse);
  if (draft_ && window > reuse) s_->dkv_lo = window;

  // Chunk ends: every prefill_chunk tokens, and the checkpoint positions (the end of each of the first
  // two messages, and chunk multiples up to checkpoint_limit), where the state is kept for later prompts.
  // Every kept state sits at a multiple of kResumeAlign: the prefill DeltaNet runs in 32-token chunks at
  // absolute positions, so a prompt resumed there computes exactly what a prefill from scratch does.
  auto align = [](int p) { return p - p % kResumeAlign; };
  if (!ckpt_slots_.empty()) {
    int messages = 0;
    for (int i = 0; i < L && i < opts_.checkpoint_limit && messages < 2; ++i)
      if (prompt[i] == opts_.boundary_token) {
        if (align(i + 1) > 0) cur.bounds.push_back(align(i + 1));
        ++messages;
      }
    for (int b = opts_.prefill_chunk; b < std::min(L, opts_.checkpoint_limit + 1); b += opts_.prefill_chunk)
      cur.bounds.push_back(b);
    std::sort(cur.bounds.begin(), cur.bounds.end());
  }
  cur.snap_at = align(L);  // the end-of-prompt state is kept here (the tokens after it are re-prefilled)
  cur.window = window;
  cur.next = reuse;
  cur.stats.reused_tokens = reuse;
  cur.stats.prefill_tokens = L - reuse;
  return cur;
}

bool Engine::prefill_chunk(PrefillCursor& cur, int max_tokens) {
  if (cur.done()) return true;
  select_session(cur.session);
  const auto t_chunk = std::chrono::steady_clock::now();
  const std::vector<int>& prompt = cur.prompt;
  const int L = static_cast<int>(prompt.size());
  int chunk = max_tokens > 0 ? std::min(max_tokens, opts_.prefill_chunk) : opts_.prefill_chunk;
  chunk = std::max(kResumeAlign, chunk - chunk % kResumeAlign);
  const int i = cur.next, snap_at = cur.snap_at, window = cur.window;
  int end = std::min(i + chunk, L);
  for (int b : cur.bounds)
    if (b > i && b < end) {
      end = b;
      break;
    }
  // The end-of-prompt state at snap_at < end is captured inside the pass (no separate pass for the tail).
  const int keep_at = snap_at > i && snap_at < end ? snap_at - i : -1;
  if (keep_at > 0) s_->snap_pos = -1;  // the snap is rewritten by the pass
  const int n = end - i, pos0 = s_->pos;
  forward(prompt.data() + i, n, Pass::Prefill, keep_at);
  // A pass whose logits are not finite left a broken state (a NaN anywhere in it reaches the last
  // token's logits). Keep nothing from it: the snap stays invalid if the pass wrote it, no checkpoint is
  // saved and its tokens do not join the history. A non-finite checkpoint would be reused by every later
  // prompt that shares the prefix. (A decode step's state is never kept, so only prefill needs this.)
  if (!all_finite(logits_)) {
    s_->pos = i;
    throw NonFiniteLogits();
  }
  if (keep_at > 0) s_->snap_pos = snap_at;
  if (draft_) {
    const int first = std::max(pos0, window);
    if (first < pos0 + n) draft_materialize(pos0 + n - first, first, first - pos0);  // still quantized_
  }
  s_->history.insert(s_->history.end(), prompt.begin() + i, prompt.begin() + end);
  cur.next = end;
  if (end == snap_at) {
    copy_state(s_->snap, true);
    s_->snap_pos = snap_at;
  }
  if (end < L && std::binary_search(cur.bounds.begin(), cur.bounds.end(), end)) save_checkpoint(end, cur.ckpt_epoch);
  cur.stats.chunks += 1;
  if (end < L) {
    cur.stats.compute_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_chunk).count();
    return false;
  }
  check(cudaStreamSynchronize(stream_), "prefill");
  s_->prompt_len = L;
  s_->turn_start = L;  // a raw prompt without message boundaries is bound only when extended whole
  for (int k = L - 1; k >= 0; --k)
    if (prompt[k] == opts_.boundary_token) {
      s_->turn_start = k + 1;
      break;
    }
  const auto now = std::chrono::steady_clock::now();
  cur.stats.compute_seconds += std::chrono::duration<double>(now - t_chunk).count();
  cur.stats.prefill_seconds = std::chrono::duration<double>(now - cur.t0).count();
  return true;
}

const std::vector<float>& Engine::prefill(const std::vector<int>& prompt, EngineStats* stats, bool from_scratch) {
  PrefillCursor cur = begin_prefill(prompt, from_scratch);
  while (!prefill_chunk(cur)) {
  }
  if (stats) *stats = cur.stats;
  return logits_;
}

const std::vector<float>& Engine::step(int token) {
  const int pos0 = s_->pos;
  forward(&token, 1, Pass::Decode);
  if (draft_) draft_materialize(1, pos0, 0);
  s_->history.push_back(token);
  return logits_;
}

int Engine::sample(const std::vector<float>& logits, const SamplingParams& p, const std::vector<int>& history) {
  if (p.seed != 0 && p.temperature > 0.f) s_->rng.seed(p.seed + history.size());
  const size_t output = std::min(history.size(), s_->prompt_len);  // where the request's output starts
  return sample_logits(logits, p, std::span<const int>(history).subspan(output), s_->rng);
}

}  // namespace ling
