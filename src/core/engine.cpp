#include "core/engine.hpp"

#include <cuda.h>
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

void check_cu(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* m = nullptr;
    cuGetErrorString(r, &m);
    throw std::runtime_error(std::string(what) + ": " + (m ? m : "?"));
  }
}

// The side pass's stream (EngineOptions::side_prefill): on a green context of `sms` SMs (its kernels then run on
// those SMs only, while the main stream's work takes any SM), or a plain non-blocking stream when sms < 0.
cudaStream_t make_side_stream(int sms, void** green) {
  if (sms < 0) {
    cudaStream_t st;
    check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "side stream");
    return st;
  }
  CUdevice dev;
  CUdevResource all, part, rest;
  unsigned groups = 1;
  CUdevResourceDesc desc;
  CUgreenCtx g;
  CUstream st;
  check_cu(cuDeviceGet(&dev, 0), "cuDeviceGet");
  check_cu(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
  check_cu(cuDevSmResourceSplitByCount(&part, &groups, &all, &rest, 0, static_cast<unsigned>(sms)),
           "cuDevSmResourceSplitByCount");
  check_cu(cuDevResourceGenerateDesc(&desc, &part, 1), "cuDevResourceGenerateDesc");
  check_cu(cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
  check_cu(cuGreenCtxStreamCreate(&st, g, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate");
  *green = g;
  std::fprintf(stderr, "side stream: green context of %u of %u SMs\n", part.sm.smCount, all.sm.smCount);
  return reinterpret_cast<cudaStream_t>(st);
}

void destroy_side_stream(cudaStream_t st, void* green) {
  cudaStreamDestroy(st);
  if (green) cuGreenCtxDestroy(static_cast<CUgreenCtx>(green));
}

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
template uint64_t* Engine::alloc<uint64_t>(size_t);

// The scratch one pass writes (the activations of a prefill chunk, the quantized copies, the attention partials, the
// last row's logits, the rows path's FP16 copies), into the members named now: the main set, or the side set while
// swapped (side_prefill).
void Engine::alloc_scratch() {
  const ModelConfig& c = model_->config();
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
}

Engine::Engine(const std::string& model_dir, EngineOptions opts) : opts_(opts) {
  // BACKLOG 34: the prefill chunk (tokens per pass alone, a multiple of kResumeAlign); every scratch buffer is sized by
  // it. A prompt's result does not depend on how its prefill is split (--chunk-check), so this changes no output bit.
  if (const char* e = std::getenv("LING_PREFILL_CHUNK"))
    opts_.prefill_chunk = std::max(kResumeAlign, std::atoi(e) - std::atoi(e) % kResumeAlign);
  model_ = std::make_unique<Model>(model_dir);
  const ModelConfig& c = model_->config();
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");

  const int C = c.lin_conv_channels(), Vs = c.lin_value_size();
  const int kvsize = c.kv_heads * c.head_dim;
  alloc_scratch();
  pos_dev_ = alloc<int>(kernels::kMaxSeqs);
  rdelta_dev_ = alloc<int>(kernels::kMaxSeqs);
  check(cudaMemset(rdelta_dev_, 0, kernels::kMaxSeqs * sizeof(int)), "rope deltas");
  mrope_dev_ = alloc<int>(size_t(std::max(opts_.prefill_chunk, kernels::kMaxStreamRows)) * 3);
  // The vision tower and the embedding cache's reserve (VISION §20.4), before the prefix KV store sizes itself.
  if (opts_.vision) {
    const int max_patches = static_cast<int>(std::min<long>(opts_.vision_max_pixels / 256, 1 << 20)) & ~3;
    vision_ = std::make_unique<VisionTower>(model_dir, max_patches);
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    check(cudaGetDevice(&props.location.id), "device");
    check(cudaMemPoolCreate(&emb_pool_, &props), "embedding cache");
    uint64_t keep_all = ~uint64_t(0);
    check(cudaMemPoolSetAttribute(emb_pool_, cudaMemPoolAttrReleaseThreshold, &keep_all), "embedding cache");
    emb_cap_ = size_t(std::max(opts_.vision_cache_mb, 0)) << 20;
    if (emb_cap_) {  // reserve it now: the pool keeps what was freed
      void* p = nullptr;
      check(cudaMallocFromPoolAsync(&p, emb_cap_, emb_pool_, stream_), "embedding cache");
      check(cudaFreeAsync(p, stream_), "embedding cache");
      check(cudaStreamSynchronize(stream_), "embedding cache");
    }
    std::fprintf(stderr, "vision: tower %zu MB, scratch %zu MB for %d patches, embedding cache %zu MB\n",
                 vision_->weight_bytes() >> 20, vision_->scratch_bytes() >> 20, max_patches, emb_cap_ >> 20);
  }
  check(cudaHostAlloc(reinterpret_cast<void**>(&pin_), sizeof(StepPinned), cudaHostAllocDefault), "pinned step buffers");
  kernels::prepare_kernels();  // attributes are set here, never inside a graph capture

  // The session slots (M4 §1), each allocated in full here: nothing is allocated while serving.
  if (opts_.sessions < 1) throw std::runtime_error("sessions must be at least 1");
  sessions_.resize(opts_.sessions);
  layer_slot_.resize(c.layers);
  int n_full = 0;
  for (int i = 0; i < c.layers; ++i) layer_slot_[i] = c.full_attention[i] ? n_full++ : n_linear_++;
  kv_pages_ = opts_.kv_pages;
  if (const char* e = std::getenv("LING_KV_PAGES")) kv_pages_ = std::atoi(e) != 0;
  constexpr int PT = kernels::kKvPageTokens;
  const int pages_per_session = (opts_.max_context + PT - 1) / PT;
  if (kv_pages_) {
    // P4 (BATCHING §5.3): one pool of pages, each [layer][K, V][kv head][PT][D]; inside a page a KV head's positions
    // are one contiguous run, as in the head-major slots. Page 0 is never mapped: a masked key's address lands there.
    // P4b: the pool holds opts_.kv_pool_pages pages (at most what every session at max_context could map), handed out
    // as passes reach them (kv_ensure). Zeroed here, so its memory is committed at start, not while serving.
    kernels::set_kv_layout(size_t(PT) * c.head_dim, c.head_dim);
    kv_page_elems_ = size_t(n_full) * 2 * kvsize * PT;
    const long cap = long(opts_.sessions) * pages_per_session;
    const int pool_pages = static_cast<int>(opts_.kv_pool_pages > 0 ? std::min<long>(opts_.kv_pool_pages, cap) : cap);
    kv_pool_ = alloc<__nv_bfloat16>(kv_page_elems_ * (size_t(pool_pages) + 1));
    check(cudaMemset(kv_pool_, 0, kv_page_elems_ * (size_t(pool_pages) + 1) * sizeof(__nv_bfloat16)), "KV pool");
    kv_book_ = std::make_unique<KvPool>(pool_pages + 1);
    for (cudaEvent_t& ev : kv_release_ev_) check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "KV release");
    std::fprintf(stderr,
                 "KV pool: %.2f GB, %d pages of %d positions (%.1f MB each), %ld positions; %d pages map one session at "
                 "max_context\n",
                 kv_page_elems_ * 2.0 * pool_pages / 1e9, pool_pages, PT, kv_page_elems_ * 2 / 1e6,
                 long(pool_pages) * PT, pages_per_session);
  } else {
    // Head-major KV caches: one KV head's keys are one contiguous stream for the attention kernels.
    kernels::set_kv_layout(size_t(opts_.max_context) * c.head_dim, c.head_dim);
  }
  std::random_device seed_source;
  for (Session& s : sessions_) {
    if (kv_pages_) {
      for (int i = 0; i < n_full; ++i) {
        s.kcache.push_back(kv_pool_ + size_t(i) * 2 * kvsize * PT);
        s.vcache.push_back(kv_pool_ + (size_t(i) * 2 + 1) * kvsize * PT);
      }
      // Every entry at page 0 until a pass maps the page (kv_ensure).
      s.page_host.assign(pages_per_session, 0);
      s.pages = alloc<uint64_t>(pages_per_session);
      check(cudaMemset(s.pages, 0, pages_per_session * sizeof(uint64_t)), "KV page table");
    } else {
      for (int i = 0; i < n_full; ++i) {
        s.kcache.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * kvsize));
        s.vcache.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * kvsize));
      }
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
    bool nvfp4 = opts_.draft_nvfp4;
    if (const char* e = std::getenv("LING_DRAFT_NVFP4")) nvfp4 = std::atoi(e) != 0;
    draft_ = std::make_unique<DraftModel>(opts_.draft_dir, nvfp4);
    alloc_drafter();
    alloc_sparse_head();
    std::fprintf(stderr, "drafter: %.2f GB, block %d, fc and conv projections %s\n", draft_->device_bytes() / 1e9,
                 opts_.draft_block,
                 draft_->nvfp4_copies()
                     ? ("NVFP4 (relative squared error " + std::to_string(draft_->nvfp4_rel_sq_error()) + ")").c_str()
                     : "BF16");
  }
  if (sessions_.size() > 1) std::fprintf(stderr, "sessions: %zu resident\n", sessions_.size());
  logits_.resize(c.vocab);
  logits_host_ = logits_.data();
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
  branch_checkpoints_ = opts_.branch_checkpoints;
  if (const char* e = std::getenv("LING_BRANCH_CKPT")) branch_checkpoints_ = std::atoi(e) != 0;
  inpass_ = opts_.inpass_checkpoints;
  if (const char* e = std::getenv("LING_CKPT_INPASS")) inpass_ = std::atoi(e) != 0;
  if (inpass_ && !ckpt_slots_.empty()) soft_slot_ = alloc_slot();
  else inpass_ = false;
  commit_overlap_ = opts_.commit_overlap && draft_ != nullptr;
  if (const char* e = std::getenv("LING_COMMIT_OVERLAP")) commit_overlap_ = draft_ != nullptr && std::atoi(e) != 0;
  if (commit_overlap_) {
    check(cudaStreamCreateWithFlags(&commit_stream_, cudaStreamNonBlocking), "commit stream");
    check(cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming), "commit event");
    commit_core_ = alloc<float>(size_t(kernels::kMaxStreamRows) * Vs);
    std::fprintf(stderr, "commit beside the next draft\n");
  }
  if (const char* e = std::getenv("LING_MK_FFN")) mk_ffn_ = std::atoi(e);
  if (const char* e = std::getenv("LING_MK_GDN")) mk_gdn_ = std::atoi(e);
  if (const char* e = std::getenv("LING_MK_ATTN")) mk_attn_ = std::atoi(e);
  if (const char* e = std::getenv("LING_MK_LAYERS")) mk_layers_ = std::atoi(e);
  if (const char* e = std::getenv("LING_MK_SPAN")) mk_span_ = std::atoi(e);
  if (mk_span_ > 0) {  // the spanning kernel's own FP16 copies (see kernels::gdn_ffn_persistent)
    xh3_ = alloc<__half>(size_t(kernels::kMaxStreamRows) * c.hidden);
    xinv3_ = alloc<float>(kernels::kMaxStreamRows);
    xh4_ = alloc<__half>(size_t(kernels::kMaxStreamRows) * c.intermediate);
    xinv4_ = alloc<float>(kernels::kMaxStreamRows);
    std::fprintf(stderr, "megakernel stage 4: a verify's DeltaNet mixer and FFN in one launch (LING_MK_SPAN=%d)\n",
                 mk_span_);
  }
  if (mk_layers_ > 0)
    std::fprintf(stderr, "megakernel stage 4: the residual adds and the next norms inside the persistent kernels "
                         "(LING_MK_LAYERS=%d)\n", mk_layers_);
  if (mk_ffn_ > 0 || mk_gdn_ > 0 || mk_attn_ > 0) {
    xh2_ = alloc<__half>(size_t(kernels::kMaxStreamRows) * std::max(c.intermediate, Vs));
    xinv2_ = alloc<float>(kernels::kMaxStreamRows);
  }
  if (mk_ffn_ > 0) {
    const size_t words = size_t(c.layers) * kernels::kMkFfnWords;
    mk_counters_ = reinterpret_cast<unsigned*>(alloc<int>(words));
    check(cudaMemset(mk_counters_, 0, words * sizeof(unsigned)), "megakernel counters");
    std::fprintf(stderr, "megakernel stage 1: the rows path's FFN as one persistent kernel (LING_MK_FFN=%d)\n", mk_ffn_);
  }
  if ((mk_ffn_ > 0 || mk_gdn_ > 0) && std::getenv("LING_MK_TIMES") && std::atoi(std::getenv("LING_MK_TIMES")) != 0) {
    const size_t n = size_t(c.layers) * kMkTimesBlocks * kernels::kMkTimesPerBlock;
    for (unsigned long long** t : {&mk_ffn_times_, &mk_gdn_times_}) {
      *t = reinterpret_cast<unsigned long long*>(alloc<uint8_t>(n * sizeof(unsigned long long)));
      check(cudaMemset(*t, 0, n * sizeof(unsigned long long)), "megakernel timings");
    }
  }
  if (mk_attn_ > 0) {
    const size_t words = size_t(c.layers) * kernels::kMkAttnWords;
    mk_attn_counters_ = reinterpret_cast<unsigned*>(alloc<int>(words));
    check(cudaMemset(mk_attn_counters_, 0, words * sizeof(unsigned)), "megakernel counters");
    kernels::attn_persistent_prepare();
    if (std::getenv("LING_MK_TIMES") && std::atoi(std::getenv("LING_MK_TIMES")) != 0) {
      const size_t n = size_t(c.layers) * kernels::kMkAttnEvents;
      mk_attn_times_ = reinterpret_cast<unsigned long long*>(alloc<uint8_t>(n * sizeof(unsigned long long)));
      check(cudaMemset(mk_attn_times_, 0, n * sizeof(unsigned long long)), "megakernel timings");
    }
    std::fprintf(stderr, "megakernel stage 3: a verify's attention layer as one persistent kernel (LING_MK_ATTN=%d)\n",
                 mk_attn_);
  }
  if (mk_gdn_ > 0) {
    const size_t words = size_t(c.layers) * kernels::kMkGdnWords;
    mk_gdn_counters_ = reinterpret_cast<unsigned*>(alloc<int>(words));
    check(cudaMemset(mk_gdn_counters_, 0, words * sizeof(unsigned)), "megakernel counters");
    std::fprintf(stderr, "megakernel stage 2: a verify's DeltaNet mixer as one persistent kernel (LING_MK_GDN=%d)\n", mk_gdn_);
  }
  if (const char* e = std::getenv("LING_GDN_FORK")) fork_ = std::atoi(e);
  if (fork_ > 0) {
    int lo = 0, hi = 0;
    check(cudaDeviceGetStreamPriorityRange(&lo, &hi), "priority range");
    check(cudaStreamCreateWithPriority(&fork_stream_, cudaStreamNonBlocking, fork_ >= 2 ? hi : lo), "fork stream");
    for (cudaEvent_t* ev : {&fork_xn_, &fork_qkv_, &fork_done_})
      check(cudaEventCreateWithFlags(ev, cudaEventDisableTiming), "fork event");
    std::fprintf(stderr, "probe: the DeltaNet chain beside the in_z projection (LING_GDN_FORK=%d)\n", fork_);
  }
  if (const char* e = std::getenv("LING_SIDE_PREFILL")) opts_.side_prefill = std::atoi(e);
  if (opts_.side_prefill != 0) {
    // The side set: its own scratch (as the main set's), the drafter's feature and context rows, a pinned buffer for
    // its logits, and its stream. Allocated with the side names in place, then swapped back.
    swap_side();
    alloc_scratch();
    if (draft_) {
      const DraftConfig& d = draft_->config();
      const size_t R = std::max(opts_.prefill_chunk, kernels::kMaxStreamRows);
      cap_ = alloc<__nv_bfloat16>(R * cap_stride_);
      dctx_ = alloc<float>(R * d.hidden);
      dk_ = alloc<float>(R * d.kv_heads * d.head_dim);
      dv_ = alloc<float>(R * d.kv_heads * d.head_dim);
      conv_tmp_ = alloc<float>(size_t(C) * (c.conv_kernel - 1));
      if (draft_->nvfp4_copies()) xh_fc_ = alloc<__half>(size_t(kernels::kMaxStreamRows) * draft_->fc4().K);
    }
    check(cudaHostAlloc(reinterpret_cast<void**>(&logits_host_), size_t(c.vocab) * sizeof(float), cudaHostAllocDefault),
          "side logits");
    s_ = &sessions_[0];
    stream_ = make_side_stream(opts_.side_prefill, &side_green_);
    swap_side();
    check(cudaEventCreateWithFlags(&side_done_, cudaEventDisableTiming), "side event");
    check(cudaEventCreateWithFlags(&side_after_main_, cudaEventDisableTiming), "side event");
    std::fprintf(stderr, "prefill beside a decode: a second stream%s\n",
                 opts_.side_prefill > 0 ? (" on a green context of " + std::to_string(opts_.side_prefill) + " SMs").c_str() : "");
  }
  // The prefix KV store (BACKLOG 35) last, from what the engine's own allocations left: one arena allocated now, so the
  // footprint is fixed at start and never grows while serving. It takes at most prefix_kv_mb and leaves
  // LING_PREFIX_KV_RESERVE_MB (default 16 GB) of the device's free memory untouched (the host's safety margin and what
  // the step graphs and the stream pool still take); below 512 MB the store is off.
  int mb = opts_.prefix_kv_mb;
  if (const char* e = std::getenv("LING_PREFIX_KV_MB")) mb = std::atoi(e);
  if (mb > 0 && opts_.prefix_checkpoints > 0) {
    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "memory info");
    size_t reserve = size_t(16384) << 20;
    if (const char* e = std::getenv("LING_PREFIX_KV_RESERVE_MB")) reserve = size_t(std::atoll(e)) << 20;
    const size_t want = size_t(mb) << 20, avail = free_b > reserve ? free_b - reserve : 0;
    size_t size = std::min(want, avail);
    size -= size % (size_t(2) << 20);
    if (size >= (size_t(512) << 20) && cudaMalloc(&kv_arena_base_, size) == cudaSuccess) {
      kv_arena_ = std::make_unique<ArenaAlloc>(size, size_t(2) << 20);
      const size_t per_token = size_t(c.layers - n_linear_) * 2 * c.kv_heads * c.head_dim * sizeof(__nv_bfloat16);
      const size_t state = size_t(n_linear_) * (size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim +
                                               size_t(c.lin_conv_channels()) * (c.conv_kernel - 1)) * sizeof(float);
      PrefixKvTree::Ops ops;
      ops.alloc = [this](size_t n) -> void* {
        size_t off = 0;
        return kv_arena_->alloc(n, &off) ? static_cast<char*>(kv_arena_base_) + off : nullptr;
      };
      // Stream-ordered reuse: every copy into or out of a range is queued on stream_ (the main or, swapped, the side
      // stream, which starts after the main stream's work and is joined before any other prefill), so a range given
      // out again is written only after the copies that read it.
      ops.release = [this](void* p) { kv_arena_->release(static_cast<char*>(p) - static_cast<char*>(kv_arena_base_)); };
      ops.store = [this](void* buf, int buf_start, int buf_len, int from, int to) {
        kv_copy(buf, buf_start, buf_len, from, to, true);
      };
      if (kv_book_) {
        // P4c paged mode: a node holds a reference to each pool page of the active session its run overlaps (the
        // session that just prefilled the prompt, kv_insert), no copy; the arena holds the states only.
        ops.hold = [this](int from, int to) -> std::vector<int> {
          const int a = KvPool::index_of(from), b = KvPool::index_of(to - 1);
          if (b >= static_cast<int>(s_->kv_ids.size())) return {};
          for (int j = a; j <= b; ++j)
            if (s_->kv_ids[j] == 0) return {};
          std::vector<int> ids;
          for (int j = a; j <= b; ++j) {
            kv_book_->retain(s_->kv_ids[j]);
            ids.push_back(s_->kv_ids[j]);
          }
          return ids;
        };
        ops.retain = [this](int page) { kv_book_->retain(page); };
        ops.drop = [this](int page) { kv_book_->release(page); };
      }
      kv_tree_ = std::make_unique<PrefixKvTree>(size, kv_book_ ? 0 : per_token, state, std::move(ops));
      std::fprintf(stderr, "prefix KV store: %zu MB of %d asked (%zu MB free after the engine's own memory, %zu kept free; %zu KB a token, %zu MB a state)\n",
                   size >> 20, mb, free_b >> 20, reserve >> 20, per_token >> 10, state >> 20);
    } else {
      cudaGetLastError();
      std::fprintf(stderr, "prefix KV store: off (%zu MB free after the engine's own memory, %zu kept free, %d asked)\n",
                   free_b >> 20, reserve >> 20, mb);
    }
  }
}

void Engine::swap_side() {
#define LING_SIDE_SWAP(m) std::swap(m, side_.m);
  LING_SIDE_SET(LING_SIDE_SWAP)
#undef LING_SIDE_SWAP
  side_active_ = !side_active_;
}

void Engine::join_commit() {
  // A side pass belongs to another session than the step whose commit is pending: it neither reads nor writes
  // that state, so it does not wait (and the main stream's join stays pending).
  if (!commit_pending_ || side_active_) return;
  // A wait on an event recorded outside a capture cannot be captured: every captured body runs with nothing
  // pending (speculate joins before the verify; the draft and the drafter's context KV never join).
  if (use_dev_pos_) throw std::runtime_error("join_commit inside a step graph's capture");
  check(cudaStreamWaitEvent(stream_, commit_done_, 0), "join commit");
  commit_pending_ = false;
}

// The timings store's summary: per kernel kind, the mean over layers of each stamp's offset from the layer's first
// block start, for the chain's blocks and the others (stage 2), in microseconds.
void Engine::mk_times_report() {
  const ModelConfig& c = model_->config();
  const int S = kernels::kMkTimesPerBlock;
  std::vector<unsigned long long> h(size_t(c.layers) * kMkTimesBlocks * S);
  for (int kind = 0; kind < 2; ++kind) {
    unsigned long long* d = kind == 0 ? mk_ffn_times_ : mk_gdn_times_;
    if (!d || cudaMemcpy(h.data(), d, h.size() * sizeof(h[0]), cudaMemcpyDeviceToHost) != cudaSuccess) continue;
    const int chain = kind == 1 ? (std::getenv("LING_MK_CHAIN") ? std::atoi(std::getenv("LING_MK_CHAIN")) : 24) : 0;
    double sum[2][16] = {}, cnt[2][16] = {}, endmax = 0, nl = 0;
    for (int l = 0; l < c.layers; ++l) {
      const unsigned long long* L = h.data() + size_t(l) * kMkTimesBlocks * S;
      unsigned long long t0 = ~0ull, tend = 0;
      for (int b = 0; b < kMkTimesBlocks; ++b)
        if (L[b * S]) t0 = std::min(t0, L[b * S]);
      if (t0 == ~0ull) continue;
      for (int b = 0; b < kMkTimesBlocks; ++b) {
        if (!L[b * S]) continue;
        const int role = b < chain ? 0 : 1;
        for (int k = 0; k < S; ++k)
          if (L[b * S + k] >= t0 && L[b * S + k] - t0 < 100'000'000ull) {
            sum[role][k] += double(L[b * S + k] - t0) / 1e3;
            cnt[role][k] += 1;
            tend = std::max(tend, L[b * S + k]);
          }
      }
      endmax += double(tend - t0) / 1e3;
      nl += 1;
    }
    if (nl == 0) continue;
    std::fprintf(stderr, "megakernel timings (%s, %d layers, last step; us from the first block's start; layer span %.1f):\n",
                 kind == 0 ? "ffn_persistent" : "gdn_persistent", int(nl), endmax / nl);
    for (int role = 0; role < 2; ++role) {
      if (kind == 0 && role == 0) continue;
      std::fprintf(stderr, "  %s:", kind == 0 ? "all blocks" : role == 0 ? "chain blocks" : "streaming blocks");
      for (int k = 0; k < S; ++k)
        if (cnt[role][k] > 0) std::fprintf(stderr, " p%d %.1f", k, sum[role][k] / cnt[role][k]);
      std::fprintf(stderr, "\n");
    }
  }
  // Stage 3: each counter's completion, the mean over the attention layers of its offset from the start, in us.
  std::vector<unsigned long long> e(size_t(c.layers) * kernels::kMkAttnEvents);
  if (mk_attn_times_ &&
      cudaMemcpy(e.data(), mk_attn_times_, e.size() * sizeof(e[0]), cudaMemcpyDeviceToHost) == cudaSuccess) {
    constexpr int E = kernels::kMkAttnEvents;
    double sum[E] = {}, cnt[E] = {};
    int nl = 0;
    for (int l = 0; l < c.layers; ++l) {
      const unsigned long long* L = e.data() + size_t(l) * E;
      if (!L[0]) continue;
      ++nl;
      for (int k = 1; k < E; ++k)
        if (L[k] >= L[0] && L[k] - L[0] < 100'000'000ull) {
          sum[k] += double(L[k] - L[0]) / 1e3;
          cnt[k] += 1;
        }
    }
    static const char* names[E] = {"start", "qkv0", "qkv1", "qkv2", "qkv3", "prep0", "prep1", "prep2", "prep3",
                                   "attn0", "attn1", "attn2", "attn3", "combine", "half", "out"};
    if (nl > 0) {
      std::fprintf(stderr, "megakernel timings (attn_persistent, %d layers, last step; us from the start):", nl);
      for (int k = 1; k < E; ++k)
        if (cnt[k] > 0) std::fprintf(stderr, " %s %.1f", names[k], sum[k] / cnt[k]);
      std::fprintf(stderr, "\n");
    }
  }
}

Engine::~Engine() {
  if (mk_ffn_times_ || mk_gdn_times_ || mk_attn_times_) {
    cudaDeviceSynchronize();
    mk_times_report();
  }
  if (commit_stream_) cudaStreamSynchronize(commit_stream_);
  if (stream_) cudaStreamSynchronize(stream_);
  kv_tree_.reset();
  if (kv_arena_base_) cudaFree(kv_arena_base_);
  for (auto& e : emb_) cudaFreeAsync(e.second.p, stream_);
  emb_.clear();
  if (stream_) cudaStreamSynchronize(stream_);
  if (emb_pool_) cudaMemPoolDestroy(emb_pool_);
  vision_.reset();
  if (side_.stream_) {
    cudaStreamSynchronize(side_.stream_);
    if (side_.logits_host_) cudaFreeHost(side_.logits_host_);
    destroy_side_stream(side_.stream_, side_green_);
    if (side_done_) cudaEventDestroy(side_done_);
    if (side_after_main_) cudaEventDestroy(side_after_main_);
  }
  clear_graphs();
  if (pin_) cudaFreeHost(pin_);
  if (profile_) {
    for (const auto& [phase, sec] : phase_seconds_) std::fprintf(stderr, "profile %-24s %8.3f s\n", phase.c_str(), sec);
  }
  for (void* p : buffers_) cudaFree(p);
  if (commit_done_) cudaEventDestroy(commit_done_);
  for (cudaEvent_t ev : kv_release_ev_)
    if (ev) cudaEventDestroy(ev);
  if (commit_stream_) cudaStreamDestroy(commit_stream_);
  for (cudaEvent_t ev : {fork_xn_, fork_qkv_, fork_done_})
    if (ev) cudaEventDestroy(ev);
  if (fork_stream_) cudaStreamDestroy(fork_stream_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Engine::reset() {
  reset_session();
  ckpts_.clear();
  if (kv_tree_) kv_store_dropped([&] { kv_tree_->clear(); });
}

void Engine::reset_session() {
  reset_state();
  check(cudaStreamSynchronize(stream_), "reset");
  s_->history.clear();
  s_->sh_dirty = true;  // the sparse head's context vocabulary goes with the history (BACKLOG 19)
  s_->prompt_len = 0;
  s_->turn_start = 0;
  s_->snap_pos = -1;
  s_->dkv_lo = 0;
  s_->pos = 0;
  s_->shadow.clear();
  if (kv_pages_) kv_release(*s_, 0);  // P4b: the session's pages back to the pool
  s_->guard.reset();  // the drafter-off guard's mean belongs to the forgotten history (BACKLOG 10)
}

int Engine::match_session(const std::vector<int>& prompt, const std::string& key, uint64_t scope) const {
  int best = -1, best_lcp = 0;
  for (size_t i = 0; i < sessions_.size(); ++i) {
    const Session& s = sessions_[i];
    if (!s.owned || s.history.empty() || s.prompt_len == 0 || s.scope != scope) continue;
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
    if (sessions_[i].owned && sessions_[i].key == key && sessions_[i].scope == scope) return static_cast<int>(i);
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

void Engine::activate_session(int i, const std::string& key, bool fresh, uint64_t scope) {
  if (i < 0 || i >= static_cast<int>(sessions_.size())) throw std::runtime_error("no such session");
  side_join();  // a side chunk's session, store entries and donor KV are complete before anything binds or copies
  s_ = &sessions_[i];
  if (s_->scope != scope) {
    // P4c: another scope's history is never resumed from; its pages stay (begin_prefill rewrites, unshares or gives
    // them back) and so does its state memory, which nothing reads before a prefill rewrites it.
    s_->history.clear();
    s_->sh_dirty = true;
    s_->prompt_len = 0;
    s_->turn_start = 0;
    s_->snap_pos = -1;
    s_->dkv_lo = 0;
    s_->pos = 0;
    s_->shadow.clear();
    s_->scope = scope;
  }
  if (fresh) {
    if (s_->owned) ++evictions_;
    s_->owned = true;
    s_->key = key;
    s_->shadow.clear();  // the lookup's records belong to the previous session's output
    s_->guard.reset();   // and the drafter-off guard's mean to its steps (BACKLOG 10)
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
  side_join();
  for (Session& s : sessions_) {
    s_ = &s;
    reset_session();
    s.spec = SpecStats{};
  }
  s_ = &sessions_[0];
  ckpts_.clear();
  if (kv_tree_) kv_store_dropped([&] { kv_tree_->clear(); });
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
    t.batched_steps += a.batched_steps;
    t.tree_steps += a.tree_steps;
    t.tree_branch_steps += a.tree_branch_steps;
    t.tree_off_chain += a.tree_off_chain;
    t.sparse_steps += a.sparse_steps;
    t.sparse_fallbacks += a.sparse_fallbacks;
    t.drafter_off_steps += a.drafter_off_steps;
    t.drafter_switches += a.drafter_switches;
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

void Engine::kv_pages_adopt(const Session& from, int n) {
  // P4c: the active session maps `from`'s pages for [0, n) (a reference each), no copy. The last of them may be only
  // partly inside [0, n): whichever of the two writes it next gets a private copy first (kv_ensure's copy-on-write;
  // begin_prefill's own kv_ensure from n on does it for the active session before its first chunk).
  const int pending = kv_book_->pending_pages();
  kv_book_->share(s_->kv_ids, from.kv_ids, n);
  kv_table_sync(*s_, stream_);
  if (kv_book_->pending_pages() > pending) kv_record_release();  // pages the active session mapped there before
}

// ---- The KV page pool (P4b, BATCHING §5.3). ----

int Engine::kv_target(size_t prompt_len) const {
  return static_cast<int>(
      std::min<size_t>(prompt_len + size_t(kKvGrowthPages) * kernels::kKvPageTokens, size_t(opts_.max_context)));
}

int Engine::kv_step_rows(const Session& s, bool spec) const {
  const int room = opts_.max_context - s.pos;
  const int B = std::min(opts_.draft_block, room);
  if (!spec || !draft_ || B < 2) return std::max(0, std::min(1, room));
  // A tree verify's rows (speculate(): the chain's first tree_chain drafts and the anchor, then tree_nodes nodes), or
  // the block; never more than the rows path takes.
  const int tree = opts_.tree_nodes > 0 ? std::min(B, opts_.tree_chain + 1) + opts_.tree_nodes : 0;
  return std::min({std::max(B, tree), int(kernels::kMaxStreamRows), room});
}

int Engine::kv_pages_free() {
  if (!kv_book_) return 0;
  kv_retire(false);
  return kv_book_->free_pages();
}

int Engine::kv_session_pages(int session) const { return kv_reclaimable({session}, false); }

int Engine::kv_reclaimable(const std::vector<int>& sessions, bool store, const std::vector<int>& store_keep) const {
  if (!kv_book_) return 0;
  std::vector<std::vector<int>> tree;
  KvPool::Refs refs;
  kv_gone_refs(KvGone{sessions, store, store_keep}, tree, refs);
  return kv_book_->reclaimable(refs);
}

void Engine::kv_gone_refs(const KvGone& gone, std::vector<std::vector<int>>& tree, KvPool::Refs& refs) const {
  for (int i : gone.sessions) refs.push_back(&sessions_.at(i).kv_ids);
  if (gone.store && kv_tree_) tree = kv_tree_->page_lists(gone.store_keep);
  for (const std::vector<int>& t : tree) refs.push_back(&t);
}

int Engine::kv_store_pages() const { return kv_reclaimable({}, true); }

bool Engine::kv_store_evict(const std::vector<int>& keep) {
  if (!kv_tree_ || !kv_book_) return false;
  bool evicted = false;
  kv_store_dropped([&] { evicted = kv_tree_->evict_lru_leaf(keep); });
  return evicted;
}

void Engine::kv_store_dropped(const std::function<void()>& op) {
  // A store call that may drop page references (an eviction, a clear, an insert making room): the pages it lets go
  // of are freed only once every stream has passed this point, as any release.
  const int pending = kv_book_ ? kv_book_->pending_pages() : 0;
  op();
  if (kv_book_ && kv_book_->pending_pages() > pending) kv_record_release();
}

int Engine::kv_admit_need(int session, const std::vector<int>& prompt, uint64_t scope, KvSource* src,
                          const KvGone* gone, bool from_scratch) const {
  if (!kv_book_ || prompt.empty()) return 0;
  const Session& s = sessions_.at(session);
  const ResumePlan plan = plan_resume(s, prompt, from_scratch, scope);
  const int need = kv_resume_need(s, plan, prompt, gone);
  if (src) {
    *src = KvSource{};
    if (plan.from_kv) src->store_path = plan.kvm.path;
    else if (plan.donor) src->donor = static_cast<int>(plan.donor - sessions_.data());
    src->cold = std::max(need, kv_resume_need(s, plan_resume(s, prompt, true, scope), prompt));
  }
  return need;
}

int Engine::kv_step_need(int session, bool spec, const KvGone* gone) const {
  if (!kv_book_) return 0;
  const Session& s = sessions_.at(session);
  std::vector<std::vector<int>> tree;
  KvPool::Refs refs;
  if (gone) kv_gone_refs(*gone, tree, refs);
  return kv_book_->needed(s.kv_ids, s.pos, s.pos + kv_step_rows(s, spec), gone ? &refs : nullptr);
}

void Engine::kv_record_release() {
  // Every stream that could still read a released page: whatever is queued on it now finishes before retire().
  const cudaStream_t streams[3] = {main_stream(), side_stream(), commit_stream_};
  for (int i = 0; i < 3; ++i)
    if (streams[i]) {
      check(cudaEventRecord(kv_release_ev_[i], streams[i]), "KV release");
      kv_release_rec_[i] = true;
    }
}

bool Engine::kv_retire(bool wait) {
  if (!kv_book_ || kv_book_->pending_pages() == 0) return true;
  for (int i = 0; i < 3; ++i) {
    if (!kv_release_rec_[i]) continue;
    if (wait) {
      check(cudaEventSynchronize(kv_release_ev_[i]), "KV release");
    } else {
      const cudaError_t e = cudaEventQuery(kv_release_ev_[i]);
      if (e == cudaErrorNotReady) return false;
      check(e, "KV release");
    }
  }
  kv_book_->retire();
  return true;
}

void Engine::kv_table_sync(Session& s, cudaStream_t st) {
  std::vector<int> idx;
  std::vector<uint64_t> val;
  for (size_t j = 0; j < s.page_host.size(); ++j) {
    const uint64_t want = j < s.kv_ids.size() ? uint64_t(s.kv_ids[j]) * kv_page_elems_ : 0;
    if (want == s.page_host[j]) continue;
    idx.push_back(static_cast<int>(j));
    val.push_back(want);
    s.page_host[j] = want;
  }
  if (!idx.empty()) kernels::kv_table_set(s.pages, idx.data(), val.data(), static_cast<int>(idx.size()), st);
}

void Engine::kv_ensure(Session& s, int from, int to, cudaStream_t st) {
  if (!kv_book_) return;
  to = std::min(to, opts_.max_context);
  if (to <= from) return;
  int need = kv_book_->needed(s.kv_ids, from, to);
  if (need == 0) return;
  // A captured pass replays without its host code: its pages are mapped before the capture or the replay
  // (speculate, speculate_batch), never here.
  if (use_dev_pos_) throw std::logic_error("KV pages to map inside a graph capture");
  if (need > kv_book_->free_pages()) kv_retire(false);
  if (need > kv_book_->free_pages()) kv_retire(true);
  const int pending = kv_book_->pending_pages();
  std::vector<KvPool::Copy> copies;
  if (!kv_book_->ensure(s.kv_ids, from, to, &copies))
    throw KvPoolExhausted("KV page pool exhausted: " + std::to_string(need) + " pages needed, " +
                          std::to_string(kv_book_->free_pages()) + " free of " +
                          std::to_string(kv_book_->capacity()));
  // Copy-on-write (P4c; no page is shared in P4b): each copy on the writer's stream before its table entry changes.
  const size_t page_bytes = kv_page_elems_ * sizeof(__nv_bfloat16);
  for (const KvPool::Copy& c : copies)
    check(cudaMemcpyAsync(kv_pool_ + size_t(c.dst) * kv_page_elems_, kv_pool_ + size_t(c.src) * kv_page_elems_,
                          page_bytes, cudaMemcpyDeviceToDevice, st),
          "KV page copy");
  kv_table_sync(s, st);
  if (kv_book_->pending_pages() > pending) kv_record_release();  // a copied page this holder let go of
}

void Engine::kv_release(Session& s, int n) {
  if (!kv_book_) return;
  const int pending = kv_book_->pending_pages();
  kv_book_->truncate(s.kv_ids, n);
  // Entries past the session's pages back to page 0, on the main stream (no other stream runs this session's work
  // here: the side chunk was joined), after whatever it queued that reads them.
  kv_table_sync(s, main_stream());
  if (kv_book_->pending_pages() > pending) kv_record_release();
}

void Engine::kv_evict(int session) {
  if (!kv_book_) return;
  if (side_cur_ && side_cur_->session == session) throw std::logic_error("kv_evict: the session's chunk is in flight");
  Session& s = sessions_.at(session);
  if (s.kv_ids.empty() && s.history.empty()) return;
  kv_release(s, 0);
  s.history.clear();
  s.sh_dirty = true;
  s.prompt_len = 0;
  s.turn_start = 0;
  s.snap_pos = -1;
  s.dkv_lo = 0;
  s.pos = 0;
  s.shadow.clear();
  if (s.owned) ++kv_evictions_;
  s.owned = false;  // free_session hands the slot out first; match_session never binds an empty history
  s.key.clear();
}

int Engine::kv_lru_idle(const std::vector<bool>& busy) const {
  int best = -1;
  for (size_t i = 0; i < sessions_.size(); ++i) {
    const Session& s = sessions_[i];
    if ((i < busy.size() && busy[i]) || (side_cur_ && side_cur_->session == static_cast<int>(i))) continue;
    bool holds = false;
    for (int p : s.kv_ids) holds = holds || p != 0;
    if (holds && (best < 0 || s.last_used < sessions_[best].last_used)) best = static_cast<int>(i);
  }
  return best;
}

void Engine::adopt_prefix(const Session& from, int n) {
  join_commit();
  const ModelConfig& c = model_->config();
  // Head-major caches: each KV head's positions [0, n) are one contiguous run, max_context positions apart.
  const size_t row = size_t(c.head_dim) * sizeof(__nv_bfloat16), pitch = size_t(opts_.max_context) * row;
  if (kv_pages_) kv_pages_adopt(from, n);
  for (size_t l = 0; l < (kv_pages_ ? 0 : s_->kcache.size()); ++l) {
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
  s_->sh_dirty = true;  // the sparse head's context vocabulary is rebuilt from the adopted history (BACKLOG 19)
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

uint64_t Engine::draft_hash() {
  if (!draft_) return 0;
  const DraftConfig& d = draft_->config();
  const size_t dkv = size_t(d.kv_heads) * d.head_dim;
  join_commit();
  check(cudaStreamSynchronize(stream_), "draft_hash");
  std::vector<__nv_bfloat16> host(size_t(ring_) * dkv);
  uint64_t h = 1469598103934665603ull;
  const int hi = static_cast<int>(s_->history.size());
  for (int l = 0; l < d.layers; ++l)
    for (const __nv_bfloat16* ring : {s_->dkc[l], s_->dvc[l]}) {
      check(cudaMemcpy(host.data(), ring, host.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "draft_hash");
      for (int p = dkv_from(*s_); p < hi; ++p) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(host.data() + size_t(p % ring_) * dkv);
        for (size_t i = 0; i < dkv * sizeof(__nv_bfloat16); ++i) h = (h ^ b[i]) * 1099511628211ull;
      }
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
  // The sparse head's context keeps the forgotten tokens' rows (a superset only costs rows) and reads the tokens
  // that replace them (BACKLOG 19).
  if (static_cast<size_t>(std::max(n, 0)) < s_->sh_synced) s_->sh_rewin = true;
  s_->sh_synced = std::min(s_->sh_synced, static_cast<size_t>(std::max(n, 0)));
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
                        const std::vector<uint64_t>& prefix_hash, uint64_t scope) const {
  if (k.scope != scope) return false;
  if (k.pos > static_cast<int>(prompt.size()) || k.pos >= static_cast<int>(prefix_hash.size())) return false;
  if (prefix_hash[k.pos] != k.hash) return false;
  return std::equal(k.tokens.begin(), k.tokens.end(), prompt.begin());
}

// The prefix KV store's device side (BACKLOG 35). A state buffer is every linear layer's recurrent state then its conv
// window, back to back; a KV buffer of n positions is [layer][K, V][kv head][n][head_dim] in BF16.
Engine::StateSlot Engine::state_view(const void* p) const {
  const ModelConfig& c = model_->config();
  const size_t gdn = size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim, conv = size_t(c.lin_conv_channels()) * (c.conv_kernel - 1);
  StateSlot v;
  float* f = static_cast<float*>(const_cast<void*>(p));
  for (int i = 0; i < n_linear_; ++i) {
    v.gdn.push_back(f);
    f += gdn;
    v.conv.push_back(f);
    f += conv;
  }
  return v;
}

void Engine::kv_copy(void* buf, int buf_start, int buf_len, int from, int to, bool to_buf) {
  const ModelConfig& c = model_->config();
  const size_t D = c.head_dim, row = D * sizeof(__nv_bfloat16), cache_pitch = size_t(opts_.max_context) * row;
  const size_t buf_pitch = size_t(buf_len) * row, width = size_t(to - from) * row;
  auto* b = static_cast<__nv_bfloat16*>(buf);
  if (kv_pages_) {  // P4: one launch over the positions, through the page table
    kernels::kv_pages_copy(kv_pool_, s_->pages, static_cast<int>(s_->kcache.size()) * 2, c.kv_heads, c.head_dim, b,
                           buf_start, buf_len, from, to, to_buf, stream_);
    return;
  }
  for (size_t l = 0; l < s_->kcache.size(); ++l)
    for (int kv = 0; kv < 2; ++kv) {
      __nv_bfloat16* cache = (kv == 0 ? s_->kcache : s_->vcache)[l] + size_t(from) * D;
      __nv_bfloat16* run = b + (l * 2 + kv) * size_t(c.kv_heads) * buf_len * D + size_t(from - buf_start) * D;
      check(to_buf ? cudaMemcpy2DAsync(run, buf_pitch, cache, cache_pitch, width, c.kv_heads, cudaMemcpyDeviceToDevice, stream_)
                   : cudaMemcpy2DAsync(cache, cache_pitch, run, buf_pitch, width, c.kv_heads, cudaMemcpyDeviceToDevice, stream_),
            "prefix KV copy");
    }
}

void Engine::kv_restore(const PrefixKvTree::Match& m, int n) {
  join_commit();
  if (kv_tree_->paged()) {
    // P4c: the store's pages for [0, n) mapped into the active session (a reference each), no copy; the partly used
    // last one is copied by begin_prefill's kv_ensure before the session writes it.
    const int pending = kv_book_->pending_pages();
    kv_book_->share(s_->kv_ids, kv_tree_->pages(m, n), n);
    kv_table_sync(*s_, stream_);
    if (kv_book_->pending_pages() > pending) kv_record_release();
    return;
  }
  for (const PrefixKvTree::Segment& g : kv_tree_->segments(m, n))
    kv_copy(const_cast<void*>(g.buf), g.buf_start, g.buf_len, g.from, g.to, false);
}

void Engine::kv_insert(const PrefillCursor& cur) {
  // Paged (P4c): the new nodes hold the active session's pages; making room in the budget may drop older nodes'.
  kv_store_dropped([&] {
    const std::vector<std::pair<int, int>> want =
        kv_tree_->insert(cur.prompt, cur.kv_q, cur.kv_cuts, cur.kv_epoch, s_->scope);
    // Each cut's state from the checkpoint store (this prefill saved them; one the store had no room for is left out).
    for (const auto& [node, pos] : want)
      for (const Checkpoint& k : ckpts_)
        if (k.pos == pos && k.scope == s_->scope && std::equal(k.tokens.begin(), k.tokens.end(), cur.prompt.begin())) {
          if (void* st = kv_tree_->add_state(node, cur.kv_epoch)) copy_slot(state_view(st), ckpt_slots_[k.slot]);
          break;
        }
  });
}

void Engine::copy_slot(const StateSlot& dst, const StateSlot& src) {
  const ModelConfig& c = model_->config();
  const size_t gdn_bytes = size_t(c.lin_v_heads) * c.lin_k_dim * c.lin_v_dim * sizeof(float);
  const size_t conv_bytes = size_t(c.lin_conv_channels()) * (c.conv_kernel - 1) * sizeof(float);
  for (size_t i = 0; i < src.gdn.size(); ++i) {
    check(cudaMemcpyAsync(dst.gdn[i], src.gdn[i], gdn_bytes, cudaMemcpyDeviceToDevice, stream_), "slot copy");
    check(cudaMemcpyAsync(dst.conv[i], src.conv[i], conv_bytes, cudaMemcpyDeviceToDevice, stream_), "slot copy");
  }
}

void Engine::save_checkpoint(int pos, uint64_t epoch, const StateSlot* from, bool soft) {
  const uint64_t hash = fnv_tokens(s_->history.data(), pos);
  for (Checkpoint& k : ckpts_)
    if (k.pos == pos && k.hash == hash && k.scope == s_->scope &&
        std::equal(k.tokens.begin(), k.tokens.end(), s_->history.begin())) {
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
    // A message end or a branch point (BACKLOG 35) is what other conversations resume from: when every entry is this
    // prompt's own, it replaces the furthest chunk multiple among them instead of being dropped.
    if (victim < 0 && soft)
      for (size_t i = 0; i < ckpts_.size(); ++i)
        if (!ckpts_[i].soft && (victim < 0 || ckpts_[i].pos > ckpts_[victim].pos)) victim = static_cast<int>(i);
    if (victim < 0) return;
    slot = ckpts_[victim].slot;
    ckpts_.erase(ckpts_.begin() + victim);
  }
  // Stream-ordered: the copy completes before any later restore reads the slot (one stream, M4 §1).
  if (from) copy_slot(ckpt_slots_[slot], *from);
  else copy_state(ckpt_slots_[slot], true);
  ckpts_.push_back({pos, slot, hash, std::vector<int>(s_->history.begin(), s_->history.begin() + pos), ++ckpt_clock_, soft,
                    s_->scope});
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
  // No step matrix takes this path (it was cuBLAS's, which a graph capture could not take).
  if (use_dev_pos_) throw std::runtime_error("the BF16 GEMM inside a step graph");
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16(x_bf16_, M, w.w, y, w.N, w.K, stream_);
}

void Engine::linear_bf16_drafter(const Bf16Weight& w, const float* x, int M, float* y) {
  if (M <= kernels::kMaxStreamRows && w.K % 1024 == 0) {  // stream the weights once
    kernels::bf16_rows(x, M, w.w, y, w.N, w.K, stream_);
    return;
  }
  // No step matrix takes this path (it was cuBLAS's, which a graph capture could not take).
  if (use_dev_pos_) throw std::runtime_error("the BF16 GEMM inside a step graph");
  kernels::to_bf16(x, x_bf16_, M * w.K, stream_);
  kernels::gemm_bf16(x_bf16_, M, w.w, y, w.N, w.K, stream_);
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
  // P4b: every segment's pages for the rows it writes, on this pass's stream (the side stream for a side chunk) before
  // the pass. Inside a capture kv_ensure only checks that they are mapped: speculate and speculate_batch map a step's
  // pages before its graphs run.
  if (kv_book_)
    for (const Seg& g : segs) kv_ensure(*g.s, g.s->pos, g.s->pos + g.rows, stream_);
  // A segment's position. A captured graph (one sequence, the active one) reads it from pos_dev_ (dpos); the
  // batch path runs eagerly at each session's own position.
  auto seg_pos = [&](const Seg& g, int off) {
    return use_dev_pos_ ? dpos(off) : kernels::DevPos(g.s->pos + off).with_rope(g.s->rope_delta, nullptr);
  };
  // Several sequences (M4 stage 4): the state kernels take them all in one launch, each row at its sequence's
  // position and in its sequence's caches and state; a captured graph reads the positions from pos_dev_[i].
  const bool multi = segs.size() > 1;
  if (segs.size() > static_cast<size_t>(kernels::kMaxSeqs)) throw std::runtime_error("forward: too many sequences");
  kernels::SeqBatch sb;
  sb.n = static_cast<int>(segs.size());
  for (int i = 0; i < sb.n; ++i) {
    sb.row0[i + 1] = segs[i].row0 + segs[i].rows;
    sb.pos[i] = segs[i].s->pos;
    sb.rdelta[i] = segs[i].s->rope_delta;
  }
  if (use_dev_pos_) {
    sb.pos_dev = pos_dev_;
    sb.rdelta_dev = rdelta_dev_;
  }
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
  // The megakernel probe (BACKLOG 28): LING_L2_HINT_GDN, _MLP and _ATTN (MB, default 0: off) prefetch that much of the
  // next projection's weights into L2 right after the GEMM before a latency-bound stretch (the DeltaNet chain before
  // the out-projection, silu and the FP16 copy before the down projection, attention before its out-projection).
  static const size_t hint_gdn = std::getenv("LING_L2_HINT_GDN") ? size_t(std::atof(std::getenv("LING_L2_HINT_GDN")) * 1e6) : 0;
  static const size_t hint_mlp = std::getenv("LING_L2_HINT_MLP") ? size_t(std::atof(std::getenv("LING_L2_HINT_MLP")) * 1e6) : 0;
  static const size_t hint_attn = std::getenv("LING_L2_HINT_ATTN") ? size_t(std::atof(std::getenv("LING_L2_HINT_ATTN")) * 1e6) : 0;
  check(cudaMemcpyAsync(ids_dev_, ids, M * sizeof(int), cudaMemcpyHostToDevice, stream_), "ids");
  kernels::embed(model_->embed(), ids_dev_, M, H, h_, stream_);
  if (pass == Pass::Prefill)  // an image's rows: its embedding, BF16 to FP32 (VISION §20.3 (b))
    for (const ImgRows& g : chunk_img_)
      vision_kernels::embed_rows(g.src, g.count, H, h_ + size_t(g.row) * H, stream_);
  const int* mrope = pass == Pass::Prefill && chunk_mrope_ ? mrope_dev_ : nullptr;

  // The megakernel's stage 4 (LING_MK_LAYERS): a persistent kernel adds its output into h_ itself and computes the next
  // norm (rmsnorm_half's, exactly) into xn_, xh_ and xinv_, so the add and the norm kernels are skipped. pre_normed: the
  // previous layer's FFN left this layer's input norm; post_normed: this layer's mixer left the post-attention norm.
  const bool tails = mk_layers_ > 0 && !quantized_ && kernels::mk_tail_rows(M);
  const bool span = mk_span_ > 0 && !quantized_ && kernels::mk_tail_rows(M);
  bool pre_normed = false;
  for (int li = 0; li < c.layers; ++li) {
    const LayerWeights& L = model_->layers()[li];
    const int slot = layer_slot_[li];
    const kernels::MkTail post_tail{h_, L.post_norm, xn_, xh_, xinv_, c.eps};
    bool post_normed = false;
    bool span_done = false, pre_normed_next = false;  // stage 4 (LING_MK_SPAN): this layer's mixer and FFN in one launch
    // --prefill-bf16 (B34): a DeltaNet layer of a prefill takes the BF16 branch below, which runs its own norm.
    const bool bf16_gdn = !L.full && quantized_ && prefill_bf16_;
    // The prefill path quantizes the norm's output for the projections that follow; the DeltaNet's a and b
    // projections (BF16) also read it in FP32.
    // (FP32 too wherever the projections' input scales differ, so the others can quantize it themselves.)
    const bool xn_half =
        pre_normed ? true
        : bf16_gdn ? false
        : L.full   ? norm_rows(h_, L.input_norm, xn_, M, L.q.in_scale, false,
                               L.k.in_scale != L.q.in_scale || L.v.in_scale != L.q.in_scale)
                   : norm_rows(h_, L.input_norm, xn_, M, L.in_qkv.in_scale, false, true);
    pre_normed = false;
    // The megakernel's stage 3 (LING_MK_ATTN): the whole attention layer of one chain verify as one persistent kernel.
    bool mk_attn = L.full && mk_attn_ > 0 && verify && !multi && !quantized_ && xn_half && L.q.K == H &&
                   L.k.K == H && L.v.K == H && L.o.K == qsize && L.o.N == H;
    if (mk_attn) {
      const Seg& g = segs[0];
      kernels::MkAttnParams p{};
      p.xh = xh_;
      p.xinv = xinv_;
      p.M = M;
      p.K = H;
      p.q = {L.q.w, nullptr, L.q.scale, q_gate_, L.q.N};
      p.k = {L.k.w, nullptr, L.k.scale, k_, L.k.N};
      p.v = {L.v.w, nullptr, L.v.scale, v_, L.v.N};
      p.o = {L.o.w, nullptr, L.o.scale, t1_, L.o.N};
      p.Hq = c.heads;
      p.Hkv = c.kv_heads;
      p.D = c.head_dim;
      p.rot = c.rotary_dim;
      p.theta = c.rope_theta;
      p.eps = c.eps;
      p.q_norm = L.q_norm;
      p.k_norm = L.k_norm;
      p.q_out = q_;
      p.gate = gate_;
      p.kcache = g.s->kcache[slot];
      p.vcache = g.s->vcache[slot];
      p.pages = g.s->pages;
      p.pos0 = seg_pos(g, 0);
      p.td = tree_;  // a tree verify's rows (BACKLOG 26), or nullptr
      p.scratch = attn_scratch_;
      p.attn = attn_;
      p.xh2 = xh2_;
      p.xinv2 = xinv2_;
      p.counters = mk_attn_counters_ + size_t(li) * kernels::kMkAttnWords;
      if (mk_attn_times_) p.times = mk_attn_times_ + size_t(li) * kernels::kMkAttnEvents;
      if (tails) p.tail = &post_tail;
      NvtxRange r("attention");
      mk_attn = kernels::attn_persistent(p, stream_);
      if (mk_attn) {
        if (tails) post_normed = true;
        else kernels::add_inplace(h_, t1_, M * H, stream_);
        mark("attn_proj");
      }
    }
    if (mk_attn) {
    } else if (L.full) {
      NvtxRange r("attention");
      linear_fp8_multi({{&L.q, q_gate_}, {&L.k, k_}, {&L.v, v_}}, xn_, M, xn_half);
      if (!quantized_) kernels::l2_prefetch_weights(L.o.w, false, false, L.o.N, L.o.K, hint_attn, stream_);
      mark("attn_proj");
      // Per sequence (M4 §2): its rows' q/k norms and RoPE at its positions, their KV into its cache, and their
      // attention over it.
      const int kvsize = c.kv_heads * c.head_dim;
      if (multi) {
        kernels::SeqPtrs<__nv_bfloat16> kc, vc;
        kernels::SeqPtrs<const __nv_bfloat16> kcr, vcr;
        kernels::SeqPtrs<const uint64_t> pg;
        for (int i = 0; i < sb.n; ++i) {
          kcr.p[i] = kc.p[i] = segs[i].s->kcache[slot];
          vcr.p[i] = vc.p[i] = segs[i].s->vcache[slot];
          pg.p[i] = segs[i].s->pages;
        }
        kernels::attn_prepare_seqs(q_gate_, k_, v_, L.q_norm, L.k_norm, sb, c.heads, c.kv_heads, c.head_dim, c.rotary_dim,
                                   c.rope_theta, c.eps, q_, gate_, kc, vc, stream_, pg);
        kernels::attention_rows_seqs(q_, kcr, vcr, sb, c.heads, c.kv_heads, c.head_dim, attn_scratch_, attn_, stream_,
                                     use_dev_pos_ ? opts_.max_context : 0, pg);
      }
      for (const Seg& g : multi ? std::span<const Seg>() : segs) {
        const size_t r0 = g.row0;
        kernels::attn_prepare(q_gate_ + r0 * 2 * qsize, k_ + r0 * kvsize, v_ + r0 * kvsize, L.q_norm, L.k_norm,
                              seg_pos(g, 0), g.rows, c.heads, c.kv_heads, c.head_dim, c.rotary_dim, c.rope_theta, c.eps,
                              q_ + r0 * qsize, gate_ + r0 * qsize, g.s->kcache[slot], g.s->vcache[slot], stream_,
                              verify ? tree_ : nullptr, mrope, g.s->pages);
        // Fixed key ranges at fixed positions for every pass (row-invariant), in slices of queries that
        // bound the partial results' scratch. A captured graph launches the ranges of the whole context.
        for (int q0 = 0; q0 < g.rows; q0 += kAttnSlice) {
          const int n = std::min(kAttnSlice, g.rows - q0);
          kernels::attention_rows(q_ + (r0 + q0) * qsize, g.s->kcache[slot], g.s->vcache[slot], seg_pos(g, q0), n,
                                  c.heads, c.kv_heads, c.head_dim, attn_scratch_, attn_ + (r0 + q0) * qsize, stream_,
                                  use_dev_pos_ ? opts_.max_context : 0, verify ? tree_ : nullptr, g.s->pages);
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
                                     keep ? keep_target().conv[slot] : nullptr, keep_at);
      kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g_, beta_, M, c.lin_v_heads, stream_);
      mark("gdn_conv");
      kernels::gdn_recurrent_prefill_bf16(mixed_bf16_, g_, beta_, s_->gdn[slot], core_bf16_, M, c.lin_k_heads,
                                          c.lin_v_heads, s_->pos, stream_, keep ? keep_target().gdn[slot] : nullptr, keep_at);
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
      const bool fork = fork_ > 0 && verify && !tree_ && !multi && !quantized_ && L.in_a.N == L.in_b.N;
      // The megakernel's stage 2 (LING_MK_GDN): the whole mixer of one verify sequence as one persistent kernel; a tree
      // verify (BACKLOG 26) too, with the tree conv and the per-path recurrence (on by default since 2026-10-20:
      // -2.2 to -2.6 ms per step with trees at 24K and 59K; LING_MK_GDN_TREE=0 gives a tree the graph path's mixer).
      static const bool gdn_tree = !std::getenv("LING_MK_GDN_TREE") || std::atoi(std::getenv("LING_MK_GDN_TREE")) != 0;
      bool mk_gdn = mk_gdn_ > 0 && verify && (!tree_ || gdn_tree) && !multi && !quantized_ && xn_half && L.in_a.N == L.in_b.N &&
                    L.in_a.N == c.lin_v_heads;
      if (mk_gdn) {
        const Seg& sg = segs[0];
        kernels::MkGdnParams p{};
        p.xh = xh_;
        p.xinv = xinv_;
        p.xn = xn_;
        p.M = M;
        p.K = H;
        p.in_qkv = {L.in_qkv.w, nullptr, L.in_qkv.scale, v_pre_[slot], L.in_qkv.N};
        p.in_z = {L.in_z.w, nullptr, L.in_z.scale, z_, L.in_z.N};
        p.out = {L.out.w, nullptr, L.out.scale, t1_, L.out.N};
        p.wa = L.in_a.w;
        p.wb = L.in_b.w;
        p.a = a_;
        p.b = b_;
        p.g = g;
        p.beta = beta;
        p.A_log = L.A_log;
        p.dt_bias = L.dt_bias;
        p.mixed = mixed;
        p.conv_state = sg.s->conv[slot];
        p.conv_w = L.conv;
        p.state = sg.s->gdn[slot];
        p.core = core_;
        p.norm_w = L.lin_norm;
        p.eps = c.eps;
        p.normed = normed_;
        p.xh2 = xh2_;
        p.xinv2 = xinv2_;
        p.C = C;
        p.H = c.lin_k_heads;
        p.HV = c.lin_v_heads;
        p.Dk = c.lin_k_dim;
        p.Dv = c.lin_v_dim;
        p.Vs = Vs;
        p.td = tree_;
        p.counters = mk_gdn_counters_ + size_t(slot) * kernels::kMkGdnWords;
        if (mk_gdn_times_) p.times = mk_gdn_times_ + size_t(li) * kMkTimesBlocks * kernels::kMkTimesPerBlock;
        if (tails) p.tail = &post_tail;
        const bool shapes = L.in_qkv.K == H && L.in_z.K == H && L.out.K == Vs && L.in_a.K == H;
        if (span && shapes && mk_ffn_ > 0) {  // stage 4 (LING_MK_SPAN): the mixer and the FFN in one launch
          const kernels::MkTail span_post{h_, L.post_norm, xn_, xh3_, xinv3_, c.eps};
          const kernels::MkTail span_next{h_, li + 1 < c.layers ? model_->layers()[li + 1].input_norm : nullptr, xn_,
                                          xh_, xinv_, c.eps};
          const kernels::StreamTarget sg_g{L.gate.w, L.gate.table, L.gate.scale2, t1_, L.gate.N},
              sg_u{L.up.w, L.up.table, L.up.scale2, t2_, L.up.N},
              sg_d{L.down.w, L.down.table, L.down.scale2, t2_, L.down.N};
          p.tail = &span_post;
          span_done = kernels::gdn_ffn_persistent(
              p, sg_g, sg_u, sg_d, I, t1_, t2_, xh4_, xinv4_, mk_counters_ + size_t(li) * kernels::kMkFfnWords,
              mk_ffn_times_ ? mk_ffn_times_ + size_t(li) * kMkTimesBlocks * kernels::kMkTimesPerBlock : nullptr,
              &span_next, stream_);
          p.tail = tails ? &post_tail : nullptr;
          if (span_done) pre_normed_next = span_next.norm_w != nullptr;
        }
        mk_gdn = span_done || (shapes && kernels::gdn_persistent(p, stream_));
      }
      if (span_done) {
        mark("gdn_proj");
      } else if (mk_gdn) {
        if (tails) post_normed = true;
        else kernels::add_inplace(h_, t1_, M * H, stream_);
        mark("gdn_proj");
      } else if (fork) {
        // The overlap probe (LING_GDN_FORK): in_qkv alone, then in_z on stream_ while the chain runs on fork_stream_
        // (a matrix's result does not depend on its launch companions, so splitting the launch changes no bit).
        const Seg& sg = segs[0];
        cudaStream_t fs = fork_stream_;
        check(cudaEventRecord(fork_xn_, stream_), "fork");
        check(cudaStreamWaitEvent(fs, fork_xn_), "fork");
        kernels::bf16_rows(xn_, M, L.in_a.w, a_, L.in_a.N, L.in_a.K, fs, L.in_b.w, b_);
        kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g, beta, M, c.lin_v_heads, fs);
        linear_fp8_multi({{&L.in_qkv, v_pre_[slot]}}, xn_, M, xn_half);
        check(cudaEventRecord(fork_qkv_, stream_), "fork");
        linear_fp8_multi({{&L.in_z, z_}}, xn_, M, true);
        kernels::l2_prefetch_weights(L.out.w, false, false, L.out.N, L.out.K, hint_gdn, stream_);
        check(cudaStreamWaitEvent(fs, fork_qkv_), "fork");
        check(cudaMemcpyAsync(mixed, v_pre_[slot], size_t(M) * C * sizeof(float), cudaMemcpyDeviceToDevice, fs),
              "verify inputs");
        check(cudaMemcpyAsync(conv_tmp_, sg.s->conv[slot], size_t(C) * (c.conv_kernel - 1) * sizeof(float),
                              cudaMemcpyDeviceToDevice, fs),
              "verify conv");
        kernels::gdn_conv(mixed, conv_tmp_, L.conv, M, C, fs);
        kernels::gdn_recurrent(mixed, g, beta, sg.s->gdn[slot], nullptr, core_, M, c.lin_k_heads, c.lin_v_heads, fs);
        check(cudaEventRecord(fork_done_, fs), "fork");
        check(cudaStreamWaitEvent(stream_, fork_done_), "fork");
        kernels::gated_rmsnorm(core_, z_, L.lin_norm, normed_, M * c.lin_v_heads, c.lin_v_dim, c.eps, stream_);
        linear_fp8_residual(L.out, normed_, M, false);
        mark("gdn_proj");
      } else {
      // The prefill path convolves out of place: the projection goes to t2_ (free until the MLP).
      linear_fp8_multi({{&L.in_qkv, verify ? v_pre_[slot] : quantized_ ? t2_ : mixed}, {&L.in_z, z_}}, xn_, M, xn_half);
      if (!quantized_) kernels::l2_prefetch_weights(L.out.w, false, false, L.out.N, L.out.K, hint_gdn, stream_);
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
      if (verify && tree_) {  // a tree (BACKLOG 26): each row's taps are its ancestors', the window left unwritten
        kernels::gdn_conv_tree(v_pre_[slot], mixed, s_->conv[slot], L.conv, M, C, tree_, stream_);
      } else if (verify) {
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
                                  keep ? keep_target().conv[slot] : nullptr, keep_at);
      } else {
        kernels::gdn_conv(mixed, s_->conv[slot], L.conv, M, C, stream_);
      }
      kernels::gdn_gating(a_, b_, L.A_log, L.dt_bias, g, beta, M, c.lin_v_heads, stream_);
      mark("gdn_conv");
      if (quantized_)
        kernels::gdn_recurrent_prefill(mixed, g, beta, s_->gdn[slot], core_, M, c.lin_k_heads, c.lin_v_heads, s_->pos,
                                       stream_, keep ? keep_target().gdn[slot] : nullptr, keep_at);
      else if (verify && tree_)  // a tree (BACKLOG 26), from the state, left unwritten
        kernels::gdn_recurrent_tree(mixed, g, beta, s_->gdn[slot], core_, c.lin_k_heads, c.lin_v_heads, tree_,
                                    tree_grid_, stream_);
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
      }  // not forked
    }
    if (span_done) {
      pre_normed = pre_normed_next;  // the spanning kernel ran the FFN and left the next layer's input norm
    } else {
      NvtxRange r("mlp");
      const bool post_half =
          post_normed || norm_rows(h_, L.post_norm, xn_, M, L.gate.in_scale, true, L.up.in_scale != L.gate.in_scale);
      const kernels::MkTail next_tail{h_, li + 1 < c.layers ? model_->layers()[li + 1].input_norm : nullptr, xn_, xh_,
                                      xinv_, c.eps};
      const kernels::StreamTarget mk_g{L.gate.w, L.gate.table, L.gate.scale2, t1_, L.gate.N},
          mk_u{L.up.w, L.up.table, L.up.scale2, t2_, L.up.N}, mk_d{L.down.w, L.down.table, L.down.scale2, t2_, L.down.N};
      if (!quantized_ && mk_ffn_ > 0 && post_half &&
          kernels::ffn_persistent(xh_, xinv_, M, mk_g, mk_u, mk_d, H, I, t1_, t2_, xh2_, xinv2_,
                                  mk_counters_ + size_t(li) * kernels::kMkFfnWords, stream_,
                                  mk_ffn_times_ ? mk_ffn_times_ + size_t(li) * kMkTimesBlocks * kernels::kMkTimesPerBlock
                                                : nullptr,
                                  tails ? &next_tail : nullptr)) {
        if (tails) pre_normed = next_tail.norm_w != nullptr;
        else kernels::add_inplace(h_, t2_, M * H, stream_);
      } else if (!quantized_) {
        linear_fp4_multi({{&L.gate, t1_}, {&L.up, t2_}}, xn_, M, post_half);
        kernels::l2_prefetch_weights(L.down.w, true, L.down.table != nullptr, L.down.N, L.down.K, hint_mlp, stream_);
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
  check(cudaMemcpyAsync(logits_host_, logits_dev_, logits_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_),
        "logits");
  if (!side_active_) check(cudaStreamSynchronize(stream_), "forward");  // a side pass is waited for by its event
  mark("lm_head");
  s_->pos += M;
}

Engine::ResumePlan Engine::plan_resume(const Session& self, const std::vector<int>& prompt, bool from_scratch,
                                       uint64_t scope) const {
  // (begin_prefill's comment above its call says what is chosen and why.) Share scopes (P4c): only the sessions,
  // checkpoints and store entries of `scope` count; self's history only when it is that scope's (activate_session
  // clears it otherwise).
  const int L = static_cast<int>(prompt.size());
  ResumePlan plan;
  const int window = plan.window = draft_ ? std::max(0, L - draft_->config().sliding_window) : 0;
  std::vector<uint64_t>& prefix_hash = plan.prefix_hash;  // of prompt[0, i), for the store's lookups and LRU stamps
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
    if (s.scope != scope) return Source{};
    const int n = std::min(static_cast<int>(s.history.size()), L);
    int lcp = 0;
    while (lcp < n && s.history[lcp] == prompt[lcp]) ++lcp;
    auto usable = [&](int p) { return p <= lcp && p < L && (!draft_ || p <= window || dkv_from(s) <= window); };
    Source r;
    if (s.snap_pos > 0 && usable(s.snap_pos)) r = {s.snap_pos, &s.snap};
    for (const Checkpoint& k : ckpts_)
      if (k.pos > r.reuse && usable(k.pos) && ckpt_match(k, prompt, prefix_hash, scope)) r = {k.pos, &ckpt_slots_[k.slot]};
    return r;
  };
  Source src;
  const Session* donor = nullptr;
  if (!from_scratch) {
    src = best_in(self);
    for (const Session& s : sessions_) {
      if (&s == &self) continue;
      const Source d = best_in(s);
      if (d.reuse > src.reuse) {
        src = d;
        donor = &s;
      }
    }
  }
  // The prefix KV store (BACKLOG 35): a donor beside the sessions. Its KV runs cover the prompt's first kvm.len tokens,
  // and a state there (its own at a node's end, or the checkpoint store's) lets the prompt resume, up to the drafter's
  // window (the store holds no drafter context) and short of the prompt's last token.
  PrefixKvTree::Match& kvm = plan.kvm;
  int& kv_reuse = plan.kv_reuse;
  const void*& kv_state = plan.kv_state;
  const StateSlot*& kv_ckpt = plan.kv_ckpt;
  if (kv_tree_ && !from_scratch) {
    kvm = kv_tree_->match(prompt, scope);
    const int lim = std::min({kvm.len, draft_ ? window : L, L - 1});
    kv_reuse = kv_tree_->best_state(kvm, lim, &kv_state);
    for (const Checkpoint& k : ckpts_)
      if (k.pos > kv_reuse && k.pos <= lim && ckpt_match(k, prompt, prefix_hash, scope)) {
        kv_reuse = k.pos;
        kv_state = nullptr;
        kv_ckpt = &ckpt_slots_[k.slot];
      }
  }
  plan.from_kv = kv_reuse > src.reuse;
  plan.src_reuse = src.reuse;
  plan.src_from = src.from;
  plan.donor = donor;
  plan.reuse = plan.from_kv ? kv_reuse : src.reuse;
  // The branch point (BACKLOG 34; the hybrid models' radix rule): where this prompt leaves the longest prefix it
  // shares with another conversation, a resident session's history or a stored entry, taken before this prefill
  // truncates or replaces the active session's history. A checkpoint there lets the next prompt of either branch
  // resume at it instead of at the last chunk multiple before it (agents' system prompts share ~5-11K tokens and
  // differ after). The active session's own history counts only when the prompt does not extend it (its snap
  // covers that).
  int& branch_at = plan.branch_at;
  if (branch_checkpoints_ && !ckpt_slots_.empty()) {
    auto lcp_of = [&](const std::vector<int>& h) {
      const size_t n = std::min(h.size(), prompt.size());
      size_t i = 0;
      while (i < n && h[i] == prompt[i]) ++i;
      return static_cast<int>(i);
    };
    for (const Session& s : sessions_) {
      if (s.history.empty() || s.scope != scope) continue;
      const int l = lcp_of(s.history);
      if (&s == &self && s.prompt_len > 0 && l >= s.turn_start) continue;
      branch_at = std::max(branch_at, l);
    }
    for (const Checkpoint& k : ckpts_)
      if (k.scope == scope) branch_at = std::max(branch_at, lcp_of(k.tokens));
    branch_at = std::max(branch_at, kvm.len);
    branch_at -= branch_at % kResumeAlign;
  }
  return plan;
}

int Engine::kv_resume_need(const Session& self, const ResumePlan& plan, const std::vector<int>& prompt,
                           const KvGone* gone) const {
  std::vector<int> tree;
  const std::vector<int>* donor = nullptr;
  if (plan.from_kv) {
    tree = kv_tree_->pages(plan.kvm, plan.kv_reuse);
    donor = &tree;
  } else if (plan.donor) {
    donor = &plan.donor->kv_ids;
  }
  const int L = static_cast<int>(prompt.size());
  const int pages = kv_admit_pages(prompt.size());
  std::vector<std::vector<int>> gone_tree;
  KvPool::Refs gone_refs;
  if (gone) kv_gone_refs(*gone, gone_tree, gone_refs);
  int need = kv_book_->resume_need(self.kv_ids, plan.reuse, pages, donor, gone ? &gone_refs : nullptr);
  // The store's hold covers the pages up to its last cut: when that cut falls in the page position L is in, the store
  // shares the page the first decode step writes, which copies it first (one page more; counted even when the store
  // holds that run already, and only when the pool could hold it at all).
  if (kv_tree_ && kv_tree_->paged() && L < opts_.max_context && pages < kv_book_->capacity()) {
    const std::vector<int> cuts = kv_store_cuts(prompt, plan.branch_at);
    if (!cuts.empty() && KvPool::index_of(cuts.back() - 1) == KvPool::index_of(L)) need += 1;
  }
  return need;
}

std::vector<int> Engine::kv_store_cuts(const std::vector<int>& prompt, int branch_at) const {
  const int L = static_cast<int>(prompt.size());
  auto align = [](int p) { return p - p % kResumeAlign; };
  std::vector<int> cuts;
  for (int i = 0, messages = 0; i < L && messages < 2; ++i)
    if (prompt[i] == opts_.boundary_token) {
      if (align(i + 1) > 0) cuts.push_back(align(i + 1));
      ++messages;
    }
  if (branch_at > 0) cuts.push_back(branch_at);
  std::erase_if(cuts, [&](int c) { return c >= L || c > opts_.checkpoint_limit; });
  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  return cuts;
}

Engine::PrefillCursor Engine::begin_prefill(const std::vector<int>& prompt, bool from_scratch,
                                            const std::vector<PromptImage>* images) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");
  side_join();
  // Where the prompt resumes, planned before anything changes (P4c: the pool's need depends on the source's pages).
  // The truncation below does not change the choice: it keeps every position below the prompt's end, and the drafter
  // ring's valid start only moves to the cut, past any usable resume point.
  const ResumePlan plan0 = plan_resume(*s_, prompt, from_scratch, s_->scope);
  // P4b: the prompt's pages and kKvGrowthPages more are taken below, whatever the prompt resumes from, after the
  // session's pages past them are given back (their frees deferred as every release's: kv_ensure waits for them when
  // it needs them). Checked first, so a short pool changes nothing (ling-serve's scheduler admits a request only when
  // kv_admit_need pages are free).
  if (kv_book_) {
    const int need = kv_resume_need(*s_, plan0, prompt);
    if (need > kv_book_->free_pages()) kv_retire(false);
    if (need > kv_book_->free_pages()) kv_retire(true);
    if (need > kv_book_->free_pages())
      throw KvPoolExhausted("KV page pool exhausted: the prompt needs " + std::to_string(need) + " pages, " +
                            std::to_string(kv_book_->free_pages()) + " free of " +
                            std::to_string(kv_book_->capacity()));
    const int target = kv_target(prompt.size());
    kv_release(*s_, target);
    // The history past those pages goes with them, so nothing (a failed begin_prefill included) reads them.
    if (static_cast<int>(s_->history.size()) > target) truncate_history(target);
    s_->pos = std::min(s_->pos, target);
  }
  PrefillCursor cur;
  // Images (VISION §20.3 (b)): each run of an image's ids is one image; its grid gives its RoPE positions.
  {
    const int vocab = model_->config().vocab;
    if (images) cur.images = *images;
    const int L0 = static_cast<int>(prompt.size());
    int next_pos = 0;
    for (int i = 0; i < L0;) {
      const int t = prompt[i];
      if (t >= 0 && t < vocab) {
        ++i;
        ++next_pos;
        continue;
      }
      if (t < kImageTokenBase) throw std::runtime_error("prompt token " + std::to_string(t) + " is outside the vocabulary");
      if (!vision_) throw std::runtime_error("the prompt holds an image and the vision tower is off");
      int k = 0;
      while (k < static_cast<int>(cur.images.size()) && cur.images[k].token != t) ++k;
      if (k == static_cast<int>(cur.images.size())) throw std::runtime_error("the prompt holds an image id with no image");
      const PromptImage& im = cur.images[k];
      PrefillCursor::Span sp;
      sp.start = i;
      sp.image = k;
      sp.rows = im.grid_h / 2;
      sp.cols = im.grid_w / 2;
      sp.base = next_pos;
      int j = i;
      while (j < L0 && prompt[j] == t && j - i < im.tokens()) ++j;
      sp.len = j - i;
      if (sp.len != im.tokens())
        throw std::runtime_error("an image's run of ids is " + std::to_string(sp.len) + " tokens, its grid needs " +
                                 std::to_string(im.tokens()));
      cur.spans.push_back(sp);
      next_pos += std::max(sp.rows, sp.cols);
      i = j;
    }
    // Every pass after this prompt turns at slot + delta; this prompt's own chunks read their table (chunk_begin).
    s_->rope_delta = rope_delta_after(cur);
  }
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
  const ResumePlan& plan = plan0;
  const int window = plan.window;
  const std::vector<uint64_t>& prefix_hash = plan.prefix_hash;
  const Session* donor = plan.donor;
  const PrefixKvTree::Match& kvm = plan.kvm;
  const int kv_reuse = plan.kv_reuse;
  const bool from_kv = plan.from_kv;
  const int branch_at = plan.branch_at;
  cur.kv_epoch = ++kv_epoch_;
  if (kv_tree_ && !from_scratch) kv_tree_->touch(kvm, cur.kv_epoch);
  // The store's clock (its LRU order): every entry on this prompt's prefix is used by this prefill, and the
  // entries it saves come after; save_checkpoint never evicts one of them for another.
  cur.ckpt_epoch = ++ckpt_clock_;
  if (!prefix_hash.empty())
    for (Checkpoint& k : ckpts_)
      if (ckpt_match(k, prompt, prefix_hash, s_->scope)) k.used = cur.ckpt_epoch;
  if (from_kv) {  // the KV of [0, kv_reuse) out of the store, the state from the store or the checkpoint store
    kv_restore(kvm, kv_reuse);
    s_->history.assign(prompt.begin(), prompt.begin() + kv_reuse);
    s_->sh_dirty = true;    // the sparse head's context vocabulary is rebuilt from the new history (BACKLOG 19)
    s_->dkv_lo = kv_reuse;  // no drafter context before kv_reuse (kv_reuse <= window)
    s_->snap_pos = -1;
    s_->shadow.clear();
    copy_state(plan.kv_state ? state_view(plan.kv_state) : *plan.kv_ckpt, false);
    cur.stats.store_tokens = kv_reuse;
  } else {
    if (donor) adopt_prefix(*donor, plan.src_reuse);
    if (plan.src_from) copy_state(*plan.src_from, false);
    else reset_state();
  }
  const int reuse = plan.reuse;
  s_->pos = reuse;
  truncate_history(reuse);
  if (draft_ && window > reuse) s_->dkv_lo = window;
  if (kv_book_) {
    // P4b: the pages up to the prompt's end and kKvGrowthPages more, mapped now (taken at admission, so a prefill never
    // stops half way for memory); the session's pages past them were released above. The session's own pages below
    // that point stay and are rewritten in place: they are private, and every position past `reuse` is written by this
    // prefill or a later pass before anything reads it, as in a flat slot (a masked key is never read).
    // P4c: the pages it shares (with the prefix KV store or another session) from the first whole page past `reuse`
    // on are dropped instead (fresh pages, no copy of what is overwritten); kv_ensure copies the one `reuse` falls in.
    const int pending = kv_book_->pending_pages();
    kv_book_->unshare(s_->kv_ids, reuse);
    kv_table_sync(*s_, stream_);
    if (kv_book_->pending_pages() > pending) kv_record_release();
    kv_ensure(*s_, reuse, kv_target(prompt.size()), stream_);
  }

  // Chunk ends: every prefill_chunk tokens, and the checkpoint positions (the end of each of the first
  // two messages, and chunk multiples up to checkpoint_limit), where the state is kept for later prompts.
  // Every kept state sits at a multiple of kResumeAlign: the prefill DeltaNet runs in 32-token chunks at
  // absolute positions, so a prompt resumed there computes exactly what a prefill from scratch does.
  auto align = [](int p) { return p - p % kResumeAlign; };
  if (!ckpt_slots_.empty()) {
    // With inpass_ (BACKLOG 35) the message ends and the branch point are soft: their state is captured inside the
    // pass that crosses them (as the end-of-prompt snap is), so they do not cut a chunk short.
    std::vector<int>& at = inpass_ ? cur.soft : cur.bounds;
    int messages = 0;
    for (int i = 0; i < L && i < opts_.checkpoint_limit && messages < 2; ++i)
      if (prompt[i] == opts_.boundary_token) {
        if (align(i + 1) > 0) at.push_back(align(i + 1));
        ++messages;
      }
    for (int b = opts_.prefill_chunk; b < std::min(L, opts_.checkpoint_limit + 1); b += opts_.prefill_chunk)
      cur.bounds.push_back(b);
    if (branch_at > reuse && branch_at < L && branch_at <= opts_.checkpoint_limit) at.push_back(branch_at);
    // LING_CKPT_TRIM=1 (BACKLOG 34, measured before any default): no chunk-multiple checkpoint past both the branch
    // point and the message ends. Those positions serve only this conversation, whose own snap covers its next turn,
    // and each one cuts a chunk short off the 2,048-token grid that the other bounds shift (one more pass).
    static const bool trim = [] {
      const char* e = std::getenv("LING_CKPT_TRIM");
      return e && std::atoi(e) != 0;
    }();
    if (trim) {
      int keep_to = branch_at;
      for (int i = 0, messages = 0; i < L && i < opts_.checkpoint_limit && messages < 2; ++i)
        if (prompt[i] == opts_.boundary_token) {
          keep_to = std::max(keep_to, align(i + 1));
          ++messages;
        }
      std::erase_if(cur.bounds, [&](int b) { return b > keep_to && b % opts_.prefill_chunk == 0; });
    }
    std::sort(cur.bounds.begin(), cur.bounds.end());
    cur.bounds.erase(std::unique(cur.bounds.begin(), cur.bounds.end()), cur.bounds.end());
    // A soft point that is also a chunk end needs no capture; the others stay sorted and unique.
    std::erase_if(cur.soft, [&](int b) { return b <= reuse || b >= L || std::binary_search(cur.bounds.begin(), cur.bounds.end(), b); });
    std::sort(cur.soft.begin(), cur.soft.end());
    cur.soft.erase(std::unique(cur.soft.begin(), cur.soft.end()), cur.soft.end());
  }
  // What this prompt adds to the prefix KV store at its end: its first tokens up to the furthest of its first two
  // message ends and its branch point (the part another conversation can share), with states at those positions.
  if (kv_tree_) {
    cur.kv_cuts = kv_store_cuts(prompt, branch_at);
    cur.kv_q = cur.kv_cuts.empty() ? 0 : cur.kv_cuts.back();
  }
  cur.snap_at = align(L);  // the end-of-prompt state is kept here (the tokens after it are re-prefilled)
  cur.window = window;
  cur.next = reuse;
  cur.stats.reused_tokens = reuse;
  cur.stats.prefill_tokens = L - reuse;
  for (const PrefillCursor::Span& sp : cur.spans)
    cur.stats.image_tokens += std::max(0, sp.start + sp.len - std::max(sp.start, reuse));
  return cur;
}

void Engine::rope_rows(const PrefillCursor& cur, int from, int to, std::vector<int>& out) {
  out.resize(size_t(3) * std::max(0, to - from));
  // Text before the first span sits at its index; each span moves the text after it by max(rows, cols) - len.
  int delta = 0;
  size_t k = 0;
  for (int i = from; i < to;) {
    while (k < cur.spans.size() && cur.spans[k].start + cur.spans[k].len <= i) {
      const PrefillCursor::Span& sp = cur.spans[k];
      delta += std::max(sp.rows, sp.cols) - sp.len;
      ++k;
    }
    int* o = out.data() + size_t(3) * (i - from);
    if (k < cur.spans.size() && i >= cur.spans[k].start) {
      const PrefillCursor::Span& sp = cur.spans[k];
      const int j = i - sp.start;
      o[0] = sp.base;
      o[1] = sp.base + j / sp.cols;
      o[2] = sp.base + j % sp.cols;
    } else {
      o[0] = o[1] = o[2] = i + delta;
    }
    ++i;
  }
}

int Engine::rope_delta_after(const PrefillCursor& cur) {
  int delta = 0;
  for (const PrefillCursor::Span& sp : cur.spans) delta += std::max(sp.rows, sp.cols) - sp.len;
  return delta;
}

const __nv_bfloat16* Engine::image_embedding(const PromptImage& im, const std::vector<int>& keep, EngineStats& stats) {
  auto it = emb_.find(im.token);
  if (it != emb_.end()) {
    it->second.used = ++emb_clock_;
    return it->second.p;
  }
  if (!vision_) throw std::runtime_error("the vision tower is off");
  if (!im.patches || im.patches->size() != size_t(im.grid_h) * im.grid_w * vision_->config().patch_dim())
    throw std::runtime_error("an image's patches are missing or the wrong size");
  const size_t bytes = size_t(im.tokens()) * vision_->config().out * sizeof(__nv_bfloat16);
  // Make room: least recently used first, never an image of the current prompt. The stream finishes what reads the
  // evicted entries before they are freed.
  while (emb_bytes_ + bytes > emb_cap_) {
    auto victim = emb_.end();
    for (auto e = emb_.begin(); e != emb_.end(); ++e)
      if (std::find(keep.begin(), keep.end(), e->first) == keep.end() &&
          (victim == emb_.end() || e->second.used < victim->second.used))
        victim = e;
    if (victim == emb_.end()) break;  // everything cached is this prompt's: the pool grows past its reserve
    check(cudaStreamSynchronize(stream_), "embedding cache");
    check(cudaFreeAsync(victim->second.p, stream_), "embedding cache");
    emb_bytes_ -= victim->second.bytes;
    emb_.erase(victim);
  }
  void* p = nullptr;
  check(cudaMallocFromPoolAsync(&p, bytes, emb_pool_, stream_), "embedding cache");
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  check(cudaEventCreate(&e0), "vision timing");
  check(cudaEventCreate(&e1), "vision timing");
  check(cudaEventRecord(e0, stream_), "vision timing");
  vision_->forward(im.patches->data(), im.grid_h, im.grid_w, static_cast<__nv_bfloat16*>(p), stream_);
  check(cudaEventRecord(e1, stream_), "vision timing");
  check(cudaEventSynchronize(e1), "vision");
  float ms = 0;
  check(cudaEventElapsedTime(&ms, e0, e1), "vision timing");
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  stats.vision_seconds += ms / 1e3;
  stats.images_encoded += 1;
  emb_[im.token] = EmbEntry{static_cast<__nv_bfloat16*>(p), bytes, ++emb_clock_};
  emb_bytes_ += bytes;
  return static_cast<const __nv_bfloat16*>(p);
}

bool Engine::prefill_chunk(PrefillCursor& cur, int max_tokens) {
  if (cur.done()) return true;
  if (side_cur_) throw std::runtime_error("prefill_chunk while a side chunk is in flight");
  // A prompt whose earlier chunk ran on the side stream: that chunk's finish queued the snap and checkpoint copies of
  // this session's state and its drafter context there, which this pass must not overwrite before they are done.
  side_join();
  select_session(cur.session);
  const ChunkRun run = chunk_begin(cur, max_tokens);
  return chunk_end(cur, run);
}

Engine::ChunkRun Engine::chunk_begin(PrefillCursor& cur, int max_tokens) {
  ChunkRun r;
  r.t = std::chrono::steady_clock::now();
  const std::vector<int>& prompt = cur.prompt;
  const int L = static_cast<int>(prompt.size());
  int chunk = max_tokens > 0 ? std::min(max_tokens, opts_.prefill_chunk) : opts_.prefill_chunk;
  chunk = std::max(kResumeAlign, chunk - chunk % kResumeAlign);
  const int i = cur.next, snap_at = cur.snap_at;
  int end = std::min(i + chunk, L);
  for (int b : cur.bounds)
    if (b > i && b < end) {
      end = b;
      break;
    }
  // The soft points inside the chunk (BACKLOG 35): the pass captures one state; the snap takes precedence (the first
  // soft point then ends the chunk), and a second soft point ends it too.
  std::vector<int> caught;
  for (int b : cur.soft)
    if (b > i && b < end) caught.push_back(b);
  if (snap_at > i && snap_at < end && !caught.empty()) {
    end = caught[0];
    caught.clear();
  }
  if (caught.size() > 1) {
    end = caught[1];
    caught.resize(1);
  }
  // The end-of-prompt state at snap_at < end is captured inside the pass (no separate pass for the tail).
  int keep_at = snap_at > i && snap_at < end ? snap_at - i : -1;
  if (keep_at > 0) s_->snap_pos = -1;  // the snap is rewritten by the pass
  r.soft_at = -1;
  if (keep_at < 0 && !caught.empty()) {  // a soft point's state, into the soft slot
    if ((caught[0] - i) % kResumeAlign != 0 || end - i <= kRowsMax)
      throw std::runtime_error("prefill: a soft checkpoint off the resume grid or on the rows path");
    keep_at = caught[0] - i;
    r.soft_at = caught[0];
    keep_slot_ = &soft_slot_;
  }
  r.i = i;
  r.end = end;
  r.keep_at = keep_at;
  r.n = end - i;
  r.pos0 = s_->pos;
  // Images (VISION §20.3 (b)): the chunk's rows of each image take the image's embedding (the tower runs here when
  // the cache does not hold it), the ids are replaced by 0 for the embedding lookup, and every row of a prompt with
  // images reads its RoPE positions from the table.
  std::vector<int> ids;
  const int* chunk_ids = prompt.data() + i;
  chunk_img_.clear();
  chunk_mrope_ = cur.has_images();
  try {
    if (chunk_mrope_) {
      std::vector<int> keep;
      for (const PromptImage& im : cur.images) keep.push_back(im.token);
      for (const PrefillCursor::Span& sp : cur.spans) {
        const int a = std::max(sp.start, i), b = std::min(sp.start + sp.len, end);
        if (a >= b) continue;
        const __nv_bfloat16* emb = image_embedding(cur.images[sp.image], keep, cur.stats);
        chunk_img_.push_back({a - i, b - a, emb + size_t(a - sp.start) * model_->config().hidden});
      }
      ids.assign(prompt.begin() + i, prompt.begin() + end);
      for (int& t : ids)
        if (t >= kImageTokenBase) t = 0;
      chunk_ids = ids.data();
      std::vector<int> table;
      rope_rows(cur, i, end, table);
      check(cudaMemcpyAsync(mrope_dev_, table.data(), table.size() * sizeof(int), cudaMemcpyHostToDevice, stream_),
            "rope table");
    }
    forward(chunk_ids, r.n, Pass::Prefill, keep_at);
  } catch (...) {
    keep_slot_ = nullptr;
    chunk_img_.clear();
    chunk_mrope_ = false;
    throw;
  }
  keep_slot_ = nullptr;
  chunk_img_.clear();
  chunk_mrope_ = false;
  return r;
}

bool Engine::chunk_end(PrefillCursor& cur, const ChunkRun& run) {
  const auto t_chunk = run.t;
  const std::vector<int>& prompt = cur.prompt;
  const int L = static_cast<int>(prompt.size());
  const int i = run.i, end = run.end, keep_at = run.keep_at, n = run.n, pos0 = run.pos0;
  const int snap_at = cur.snap_at, window = cur.window;
  // A pass whose logits are not finite left a broken state (a NaN anywhere in it reaches the last
  // token's logits). Keep nothing from it: the snap stays invalid if the pass wrote it, no checkpoint is
  // saved and its tokens do not join the history. A non-finite checkpoint would be reused by every later
  // prompt that shares the prefix. (A decode step's state is never kept, so only prefill needs this.)
  if (!all_finite(logits_)) {
    s_->pos = i;
    throw NonFiniteLogits();
  }
  if (keep_at > 0 && run.soft_at < 0) s_->snap_pos = snap_at;
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
  if (run.soft_at > 0) save_checkpoint(run.soft_at, cur.ckpt_epoch, &soft_slot_, true);  // captured inside the pass
  const bool soft_end = std::binary_search(cur.soft.begin(), cur.soft.end(), end);
  if (end < L && (soft_end || std::binary_search(cur.bounds.begin(), cur.bounds.end(), end)))
    save_checkpoint(end, cur.ckpt_epoch, nullptr, soft_end);
  cur.stats.chunks += 1;
  if (end < L) {
    cur.stats.compute_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_chunk).count();
    return false;
  }
  if (kv_tree_ && cur.kv_q > 0) kv_insert(cur);
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

void Engine::prefill_chunk_launch(PrefillCursor& cur, int max_tokens) {
  if (!side_.stream_) throw std::runtime_error("prefill_chunk_launch without side_prefill");
  if (side_cur_) throw std::runtime_error("a side chunk is already in flight");
  if (cur.done()) throw std::runtime_error("prefill_chunk_launch on a finished prompt");
  Session* main_s = s_;
  // The chunk starts after everything queued on the main stream so far: begin_prefill's reset, state copy and
  // donor KV for this session ran there, and the previous chunk's finish may have been followed by main-stream work.
  check(cudaEventRecord(side_after_main_, stream_), "side order");
  swap_side();
  try {
    check(cudaStreamWaitEvent(stream_, side_after_main_, 0), "side order");
    select_session(cur.session);
    side_run_ = chunk_begin(cur, max_tokens);
    check(cudaEventRecord(side_done_, stream_), "side chunk");
  } catch (...) {
    swap_side();
    s_ = main_s;
    throw;
  }
  swap_side();
  s_ = main_s;
  side_cur_ = &cur;
}

bool Engine::side_ready() {
  if (!side_cur_) return true;
  const cudaError_t e = cudaEventQuery(side_done_);
  if (e == cudaErrorNotReady) return false;
  check(e, "side chunk");
  return true;
}

bool Engine::prefill_chunk_finish() {
  if (!side_cur_) throw std::runtime_error("prefill_chunk_finish without a side chunk");
  PrefillCursor& cur = *side_cur_;
  side_cur_ = nullptr;
  check(cudaEventSynchronize(side_done_), "side chunk");
  bool last = false;
  swap_side();
  try {
    std::copy(logits_host_, logits_host_ + logits_.size(), logits_.begin());
    last = chunk_end(cur, side_run_);
  } catch (...) {
    swap_side();
    select_session(cur.session);
    throw;
  }
  swap_side();
  select_session(cur.session);  // as prefill_chunk leaves it
  return last;
}

void Engine::side_join() {
  if (side_cur_) check(cudaEventSynchronize(side_done_), "side chunk");
  if (side_.stream_ && !side_active_) check(cudaStreamSynchronize(side_.stream_), "side stream");
}

void Engine::side_abandon() {
  if (!side_cur_) return;
  check(cudaEventSynchronize(side_done_), "side chunk");
  side_cur_ = nullptr;
}

const std::vector<float>& Engine::prefill(const std::vector<int>& prompt, EngineStats* stats, bool from_scratch,
                                          const std::vector<PromptImage>* images) {
  PrefillCursor cur = begin_prefill(prompt, from_scratch, images);
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
