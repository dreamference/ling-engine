// Speculative decoding with the DFlash2 drafter (specs/DREAMFERENCE_LING_ENGINE_SPECULATION.md §10, lever 2), as SGLang's DFLASH path runs it
// (models/dflash.py, speculative/dflash_worker_v2.py).
//
// One step from an anchor (the token chosen from the last logits, not yet processed by the target):
//   1. draft: the drafter denoises [anchor, mask x (B - 1)] at positions pos .. pos + B - 1, attending
//      bidirectionally within the block and to its context KV over the last 2048 positions; the target's
//      LM head gives each drafted position its top-16 candidates, and DFlash2's selector walks one path
//      through the 16 x 16 transition scores (greedy: argmax; sampling: a draw at the request's
//      temperature, whose 16-way distribution is the draft's q);
//   2. verify: one target pass over the B rows [anchor, d1 .. d(B-1)] on the rows path, which writes the
//      rows' KV but leaves the DeltaNet and conv state alone and keeps each layer's inputs;
//   3. accept: greedily (a draft survives while it equals the target's argmax) or by exact rejection
//      sampling, min(1, p/q) and a residual draw from max(0, p - q);
//   4. commit: the DeltaNet and conv states are advanced over the anchor and the accepted drafts by
//      replaying those rows' saved inputs through the same kernels, and their target features become
//      the drafter's context KV.
// The rows path is row-invariant, so greedy speculation reproduces plain greedy decoding token for token.
//
// CUDA graphs (M3, EngineOptions::step_graphs). Each of the three GPU parts of a step (draft, verify with
// its top-K, commit) is a fixed launch sequence with no host synchronization inside, captured once into a
// graph and replayed every step; the host keeps the two decisions between them (the lattice walk and the
// accept rule, exactly as without graphs), which cost a few microseconds each. What changes from step to step
// is handled so that a graph stays valid:
//   - the position: the kernels that depend on it read it from pos_dev_ (kernels::DevPos), which the host
//     writes before each step's graphs run;
//   - the anchor and the drafted tokens: page-locked host buffers (pin_) copied by the graphs' memcpy nodes;
//   - the accepted count, which only the commit depends on: one commit graph per count (at most 32);
//   - the attention's key ranges (its grid): the verify graph launches every range up to max_context, and a
//     range past the rows' last key exits at once with an empty partial the combine skips;
//   - sampling parameters: the verify graph is keyed by top-K, everything else is on the host.
// Every kernel computes exactly what its eager launch computes, so graphs change no output bit
// (--spec-check compares the tokens and the recurrent state with plain decoding).
//
// Batches (M4 stages 3 and 4, speculate_batch): several sessions' steps share one verify pass over their concatenated
// rows; the drafter runs over every sequence's block at once and the state kernels take all sequences in one launch
// (kernels::SeqBatch), each row computed as its sequence alone computes it (--batch-check). A batch's graphs are keyed
// by the order of its sessions (they bake each one's pointers and the row split) and read each sequence's position
// from pos_dev_[i]; each sequence's commit graph is its session's, keyed by its slot and accepted count.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <nvtx3/nvToolsExt.h>

#include "core/engine.hpp"
#include "core/kernels.cuh"
#include "core/accept.hpp"
#include "core/lookup.hpp"
#include "core/safetensors.hpp"

namespace ling {
namespace {

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// The target's sampling distribution for one row from its top-K logits (sorted, descending): the chain
// of Engine::sample (temperature, top-k, top-p, min-p), as (token, probability) pairs.
Dist target_dist(const float* vals, const int* ids, int K, const SamplingParams& p) {
  const int k = (p.top_k > 0 && p.top_k < K) ? p.top_k : K;
  std::vector<double> probs(k);
  double sum = 0;
  for (int i = 0; i < k; ++i) sum += probs[i] = std::exp((double(vals[i]) - vals[0]) / p.temperature);
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
  double kept = 0;
  for (int i = 0; i < keep; ++i) kept += probs[i];
  Dist d(keep);
  for (int i = 0; i < keep; ++i) d[i] = {ids[i], probs[i] / kept};
  return d;
}

// LING_DUMP_STEPS=FILE (a measurement aid for tree verification, BACKLOG 26): every one-stream speculative step appended
// to FILE as int32 [magic 0x54524545, pos, anchor, E, K16, K, accepted, next, greedy], the drafter's candidates
// [E][K16] (int32) and lattice scores [E][K16][K16] (float32), the verified rows [E + 1] (int32), and the verify's
// top K per row, values [E + 1][K] (float32) and ids [E + 1][K] (int32). Host-side only: the step is unchanged.
std::FILE* dump_file() {
  static std::FILE* f = [] {
    const char* p = std::getenv("LING_DUMP_STEPS");
    return p && *p ? std::fopen(p, "ab") : nullptr;
  }();
  return f;
}

// LING_DUMP_HIDDEN=FILE (with LING_DUMP_STEPS; the acceptance probe of BACKLOG 26, follow-up 2b): for every step written to
// the step dump, in the same order, int32 [magic 0x48494444, pos, E, hidden, rank], the drafted positions' final normed
// hidden states [E][hidden] and their selector projections [E][rank] (float32). A copy to the host after the draft.
std::FILE* hidden_dump_file() {
  static std::FILE* f = [] {
    const char* p = std::getenv("LING_DUMP_HIDDEN");
    return p && *p ? std::fopen(p, "ab") : nullptr;
  }();
  return f;
}

}  // namespace

// One sequence's part of a speculative step on the host: what speculate() kept in locals, so a batch can hold
// one per sequence.
struct Engine::StepPlan {
  Session* s = nullptr;
  const SamplingParams* p = nullptr;
  int anchor = 0;
  bool greedy = true, use_lookup = false;
  int E = 0, V = 0, K = 1, row0 = 0;
  LookupProposal lp;
  std::vector<int> rows;                // the anchor, then the drafted (or looked-up) tokens
  std::vector<std::vector<double>> q;   // the draft's distribution over its candidates, per drafted position
  std::vector<float> hidden, sel;       // LING_DUMP_HIDDEN: the drafted positions' hidden states and selector projections
  std::vector<int> cand;                // the drafter's candidates [E][K16] (pin_ holds one sequence's at a time)
  std::vector<int> picks;               // the chain's candidate index per drafted position
  bool tree = false;                    // a tree verify (BACKLOG 26): tr's rows, then path its accepted rows
  DraftTree tr;
  std::vector<int> path;
  int accepted = 0, next = 0;
  bool failed = false;
};

void Engine::alloc_drafter() {
  const ModelConfig& c = model_->config();
  const DraftConfig& d = draft_->config();
  if (d.hidden != c.hidden) throw std::runtime_error("drafter hidden size does not match the target");
  const int H = c.hidden, R = std::max(opts_.prefill_chunk, kernels::kMaxStreamRows), B = kernels::kMaxStreamRows;
  const int dq = d.heads * d.head_dim, dkv = d.kv_heads * d.head_dim, nfeat = static_cast<int>(d.target_layers.size());
  feature_layer_.assign(c.layers, -1);
  for (int i = 0; i < nfeat; ++i) {
    const int li = d.target_layers[i];
    if (li < 0 || li >= c.layers) throw std::runtime_error("drafter target layer out of range");
    feature_layer_[li] = i;
  }
  cap_stride_ = nfeat * H;
  cap_ = alloc<__nv_bfloat16>(size_t(R) * cap_stride_);
  // Each session's context KV is a ring (M4 §1): the window, kRingSlack positions of slack, and a commit's rows.
  constexpr int kRingSlack = 6144;
  ring_ = d.sliding_window + kRingSlack + kernels::kMaxStreamRows;
  for (Session& s : sessions_)
    for (int l = 0; l < d.layers; ++l) {
      s.dkc.push_back(alloc<__nv_bfloat16>(size_t(ring_) * dkv));
      s.dvc.push_back(alloc<__nv_bfloat16>(size_t(ring_) * dkv));
    }
  dctx_ = alloc<float>(size_t(R) * H);
  if (draft_->nvfp4_copies()) xh_fc_ = alloc<__half>(size_t(B) * draft_->fc4().K);
  dk_ = alloc<float>(size_t(R) * dkv);
  dv_ = alloc<float>(size_t(R) * dkv);
  dres_ = alloc<float>(size_t(B) * H);
  dh_ = alloc<float>(size_t(B) * H);
  dh2_ = alloc<float>(size_t(B) * H);
  dout_ = alloc<float>(size_t(B) * H);
  dcoef_ = alloc<float>(size_t(B) * 2 * d.conv_taps * d.conv_groups);
  dq_ = alloc<float>(size_t(B) * dq);
  dattn_ = alloc<float>(size_t(B) * dq);
  dg_ = alloc<float>(size_t(B) * d.intermediate);
  du_ = alloc<float>(size_t(B) * d.intermediate);
  dlogits_ = alloc<float>(size_t(B) * c.vocab);
  dunary_ = alloc<float>(size_t(B) * d.selector_top_k);
  dcand_ = alloc<int>(size_t(B) * d.selector_top_k);
  dhp_ = alloc<float>(size_t(B) * d.selector_rank);
  dscores_ = alloc<float>(size_t(B) * d.selector_top_k * d.selector_top_k);
  topk_scratch_ = alloc<float>(kernels::topk_scratch_floats(B, 64));
  vals_dev_ = alloc<float>(size_t(B) * 64);
  ids_out_dev_ = alloc<int>(size_t(B) * 64);
  if (d.selector_top_k > 16) throw std::runtime_error("drafter selector_top_k > 16");
  const int C = c.lin_conv_channels();
  for (size_t i = 0; i < s_->gdn.size(); ++i) {
    v_pre_.push_back(alloc<float>(size_t(B) * C));
    v_post_.push_back(alloc<float>(size_t(B) * C));
    v_g_.push_back(alloc<float>(size_t(B) * c.lin_v_heads));
    v_beta_.push_back(alloc<float>(size_t(B) * c.lin_v_heads));
  }
  conv_tmp_ = alloc<float>(size_t(C) * (c.conv_kernel - 1));
  set_draft_block(opts_.draft_block);
  // Tree verification (BACKLOG 26).
  if (const char* e = std::getenv("LING_TREE")) opts_.tree_nodes = std::atoi(e);
  if (const char* e = std::getenv("LING_TREE_T")) opts_.tree_temp = std::atof(e);
  if (const char* e = std::getenv("LING_TREE_W")) opts_.tree_width = std::atoi(e);
  if (const char* e = std::getenv("LING_TREE_DEPTH")) opts_.tree_max_depth = std::atoi(e);
  if (const char* e = std::getenv("LING_TREE_CHAIN")) opts_.tree_chain = std::atoi(e);
  if (const char* e = std::getenv("LING_TREE_MIN")) opts_.tree_min_mass = std::atof(e);
  tree_dev_ = reinterpret_cast<kernels::TreeDesc*>(alloc<uint8_t>(sizeof(kernels::TreeDesc)));
  gather_rows_dev_ = alloc<int>(kernels::kMaxStreamRows);
  tree_grid_ = std::min(kernels::kMaxTreeVariants, 1 + std::max(0, opts_.tree_nodes));
  // Each session's buffers a tree's accepted path is gathered in: every full-attention layer's K and V per KV head
  // (head-major, slot pos + row), and the verify's DeltaNet inputs and target features (row).
  const size_t D = c.head_dim;
  std::vector<kernels::GatherBuf> shared;
  for (size_t i = 0; i < v_pre_.size(); ++i) {
    shared.push_back({reinterpret_cast<char*>(v_pre_[i]), size_t(C) * 4, C * 4, 0});
    shared.push_back({reinterpret_cast<char*>(v_post_[i]), size_t(C) * 4, C * 4, 0});
    shared.push_back({reinterpret_cast<char*>(v_g_[i]), size_t(c.lin_v_heads) * 4, c.lin_v_heads * 4, 0});
    shared.push_back({reinterpret_cast<char*>(v_beta_[i]), size_t(c.lin_v_heads) * 4, c.lin_v_heads * 4, 0});
  }
  shared.push_back({reinterpret_cast<char*>(cap_), size_t(cap_stride_) * 2, cap_stride_ * 2, 0});
  for (Session& s : sessions_) {
    std::vector<kernels::GatherBuf> g = shared;
    for (size_t l = 0; l < s.kcache.size(); ++l)
      for (int h = 0; h < c.kv_heads; ++h) {
        // P4: a head's run inside each page, rows through the session's page table (in element units)
        const size_t off = size_t(h) * (s.pages ? kernels::kKvPageTokens : opts_.max_context) * D;
        const int unit = s.pages ? int(sizeof(__nv_bfloat16)) : 0;
        g.push_back({reinterpret_cast<char*>(s.kcache[l] + off), D * 2, int(D * 2), 1, s.pages, unit});
        g.push_back({reinterpret_cast<char*>(s.vcache[l] + off), D * 2, int(D * 2), 1, s.pages, unit});
      }
    for (const kernels::GatherBuf& b : g)
      if (b.row_bytes % 16 != 0) throw std::runtime_error("tree gather: a row is not a multiple of 16 bytes");
    s.gather = reinterpret_cast<kernels::GatherBuf*>(alloc<uint8_t>(g.size() * sizeof(kernels::GatherBuf)));
    s.ngather = static_cast<int>(g.size());
    check(cudaMemcpy(s.gather, g.data(), g.size() * sizeof(kernels::GatherBuf), cudaMemcpyHostToDevice), "tree gather");
  }
  if (opts_.tree_nodes > 0)
    std::fprintf(stderr, "tree verification: %d nodes beyond the chain (chain %d), temperature %.2f, width %d, depth %d\n",
                 opts_.tree_nodes, opts_.tree_chain, opts_.tree_temp, opts_.tree_width, opts_.tree_max_depth);
}

void Engine::alloc_sparse_head() {
  // BACKLOG 19: the drafter's context-aware sparse head, on by default where its file is beside the drafter.
  if (const char* e = std::getenv("LING_DRAFT_HEAD")) opts_.draft_head = e;
  if (const char* e = std::getenv("LING_SPARSE_HEAD_DIR")) opts_.sparse_head_dir = e;
  if (opts_.draft_head == "full") return;
  if (opts_.draft_head != "sparse") throw std::runtime_error("the drafter's head is full or sparse, not " + opts_.draft_head);
  auto env_i = [](const char* n, int& v) { if (const char* e = std::getenv(n)) v = std::atoi(e); };
  auto env_d = [](const char* n, double& v) { if (const char* e = std::getenv(n)) v = std::atof(e); };
  env_i("LING_SH_DYNAMIC", opts_.sparse_head_dynamic);
  env_i("LING_SH_CONTEXT", opts_.sparse_head_context);
  env_i("LING_SH_WINDOW", opts_.sparse_head_window);
  env_d("LING_SH_HI", opts_.sparse_head_hi);
  env_d("LING_SH_LO", opts_.sparse_head_lo);
  env_d("LING_SH_ACCEPT_FLOOR", opts_.sparse_head_accept_floor);
  opts_.sparse_head_window = std::clamp(opts_.sparse_head_window, 1, 4096);
  opts_.sparse_head_context = std::max(0, opts_.sparse_head_context);
  opts_.sparse_head_dynamic = std::max(0, opts_.sparse_head_dynamic);
  const ModelConfig& c = model_->config();
  const int V = c.vocab, H = c.hidden;
  const std::string dir = opts_.sparse_head_dir.empty() ? opts_.draft_dir + "/sparse-head" : opts_.sparse_head_dir;
  // A missing or unusable file never stops the engine: the drafter then keeps the full head, said in one line.
  std::unique_ptr<ling::Checkpoint> ck;
  const TensorView *usp = nullptr, *vtp = nullptr, *fixedp = nullptr;
  try {
    ck = std::make_unique<ling::Checkpoint>(dir);
    usp = &ck->get("us"), vtp = &ck->get("vt"), fixedp = &ck->get("fixed");
    const TensorView &us = *usp, &vt = *vtp, &fixed = *fixedp;
    if (us.dtype != "BF16" || us.shape.size() != 2 || us.shape[0] != V || us.shape[1] != 256 || vt.dtype != "BF16" ||
        vt.shape.size() != 2 || vt.shape[1] != H || vt.shape[0] != us.shape[1] || fixed.dtype != "I32")
      throw std::runtime_error("unexpected tensors (us [vocab][256] and vt [256][hidden] BF16, fixed I32)");
    if (model_->lm_head().N != V || model_->lm_head().K != H) throw std::runtime_error("the LM head's shape");
    if (H % 1024 != 0) throw std::runtime_error("a hidden size that is not a multiple of 1,024 (bf16_rows)");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "drafter head: full (no sparse head from %s: %s)\n", dir.c_str(), e.what());
    return;
  }
  const TensorView &us = *usp, &vt = *vtp, &fixed = *fixedp;
  sh_rank_ = static_cast<int>(us.shape[1]);
  auto put = [&](const TensorView& t) {
    uint8_t* d = alloc<uint8_t>(t.bytes);
    check(cudaMemcpy(d, t.data, t.bytes, cudaMemcpyHostToDevice), "sparse head upload");
    return d;
  };
  sh_us_ = reinterpret_cast<__nv_bfloat16*>(put(us));
  sh_vt_ = reinterpret_cast<__nv_bfloat16*>(put(vt));
  sh_fixed_.assign(V, 0);
  const int nf = static_cast<int>(fixed.numel());
  for (int i = 0; i < nf; ++i) {
    int id;
    std::memcpy(&id, fixed.data + 4 * size_t(i), 4);
    if (id >= 0 && id < V && !sh_fixed_[id]) sh_fixed_[id] = 1, ++sh_nfixed_;
  }
  // The capacity: the fixed rows, the session's context rows (at most sparse_head_context), the chosen ones and their
  // ties, so the clamp (which drops the highest ids) never bites in practice. Columns past a step's count cost little:
  // their GEMM blocks leave at once and the top 16 reads them as -inf. A batched draft selects per sequence.
  const long cap = long(sh_nfixed_) + long(opts_.sparse_head_context) + opts_.sparse_head_dynamic + 256;
  sh_cap_ = static_cast<int>(std::min<long>((cap + 31) / 32 * 32, (V + 31) / 32 * 32));
  sh_p_ = alloc<float>(size_t(kernels::kMaxStreamRows) * sh_rank_);
  sh_scratch_ = alloc<uint8_t>(kernels::sparse_head_scratch_bytes(V));
  sh_rows_ = alloc<int>(V);
  sh_nrows_ = alloc<int>(1);
  sh_ids_dev_ = alloc<int>(kShMarkMax);
  sh_xtmp_ = alloc<float>(size_t(kernels::kMaxStreamRows) * H);
  for (Session& s : sessions_) {
    s.sh_mask = alloc<uint8_t>(V);
    s.sh_dirty = true;
  }
  sparse_head_ = true;
  std::fprintf(stderr,
               "drafter head: sparse, %d fixed + up to %d context + %d chosen rows per step (rank %d, at most %d); full "
               "head above %.3f of the last %d tokens outside them, sparse again below %.3f%s\n",
               sh_nfixed_, opts_.sparse_head_context, opts_.sparse_head_dynamic, sh_rank_, sh_cap_, opts_.sparse_head_hi,
               opts_.sparse_head_window, opts_.sparse_head_lo,
               opts_.sparse_head_accept_floor > 0
                   ? (", or below " + std::to_string(opts_.sparse_head_accept_floor) + " tokens per step").c_str()
                   : "");
}

bool Engine::sparse_head_sync(Session& s) {
  if (!sparse_head_) return false;
  const int V = model_->config().vocab, W = opts_.sparse_head_window;
  const std::vector<int>& h = s.history;
  bool upload = false;
  if (s.sh_dirty || h.size() < s.sh_synced) {  // a new history: the fixed list alone, the window empty
    s.sh_seen = sh_fixed_;
    s.sh_ctx = 0;
    s.sh_synced = 0;
    s.sh_win.assign(W, 0);
    s.sh_win_pos = s.sh_win_n = s.sh_novel = 0;
    s.sh_full = true;
    s.sh_cool_until = 0;
    s.sh_acc_ema = -1;
    s.sh_dirty = false;
    upload = true;
  } else if (s.sh_rewin) {  // a new turn: the window and the head decided again from the tokens that follow the cut
    s.sh_win.assign(W, 0);
    s.sh_win_pos = s.sh_win_n = s.sh_novel = 0;
    s.sh_full = true;
    s.sh_cool_until = 0;
    s.sh_acc_ema = -1;
  }
  s.sh_rewin = false;
  // The acceptance floor's input: the last step's tokens, if it drafted with the sparse head.
  if (s.sh_last_sparse && s.spec.steps == s.sh_last_steps + 1) {
    const double a = double(s.spec.accepted - s.sh_last_acc) + 1;
    s.sh_acc_ema = s.sh_acc_ema < 0 ? a : s.sh_acc_ema * (15.0 / 16) + a / 16;
  }
  s.sh_last_steps = s.spec.steps;
  s.sh_last_acc = s.spec.accepted;
  // The tokens since the last draft (the prompt's at the first): each new one joins the context (up to its cap) and
  // counts as outside the selection in the window.
  std::vector<int> fresh;
  for (size_t i = s.sh_synced; i < h.size(); ++i) {
    const int t = h[i];
    if (t < 0 || t >= V) continue;
    const uint8_t novel = s.sh_seen[t] ? 0 : 1;
    if (novel && s.sh_ctx < opts_.sparse_head_context) {
      s.sh_seen[t] = 1;
      ++s.sh_ctx;
      if (!upload) fresh.push_back(t);
    }
    s.sh_novel += novel - s.sh_win[s.sh_win_pos];
    s.sh_win[s.sh_win_pos] = novel;
    s.sh_win_pos = (s.sh_win_pos + 1) % W;
    s.sh_win_n = std::min(s.sh_win_n + 1, W);
  }
  s.sh_synced = h.size();
  if (upload || fresh.size() > size_t(kShMarkMax)) {
    check(cudaMemcpyAsync(s.sh_mask, s.sh_seen.data(), V, cudaMemcpyHostToDevice, stream_), "sparse head mask");
  } else if (!fresh.empty()) {
    check(cudaMemcpyAsync(sh_ids_dev_, fresh.data(), fresh.size() * sizeof(int), cudaMemcpyHostToDevice, stream_),
          "sparse head ids");
    kernels::sparse_head_mark(s.sh_mask, sh_ids_dev_, static_cast<int>(fresh.size()), stream_);
  }
  // The fallback, with hysteresis: the full head while the window holds too few tokens or too many new ones.
  const double share = s.sh_win_n ? double(s.sh_novel) / s.sh_win_n : 1.0;
  const bool was_full = s.sh_full;
  if (s.sh_win_n < std::min(W, 16)) s.sh_full = true;
  else if (!s.sh_full && share > opts_.sparse_head_hi) s.sh_full = true;
  else if (s.sh_full && share < opts_.sparse_head_lo && h.size() >= s.sh_cool_until) s.sh_full = false;
  if (!s.sh_full && opts_.sparse_head_accept_floor > 0 && s.sh_acc_ema >= 0 &&
      s.sh_acc_ema < opts_.sparse_head_accept_floor) {
    s.sh_full = true;
    s.sh_cool_until = h.size() + W;
    s.sh_acc_ema = -1;
  }
  if (s.sh_full && !was_full) ++s.spec.sparse_fallbacks;
  static const bool verbose = [] { const char* e = std::getenv("LING_SH_VERBOSE"); return e && std::atoi(e); }();
  if (verbose && s.sh_full != was_full)
    std::fprintf(stderr, "drafter head: %s at token %zu (%.3f of the last %d new, %d context rows, ema %.2f)\n",
                 s.sh_full ? "full" : "sparse", h.size(), share, s.sh_win_n, s.sh_ctx, s.sh_acc_ema);
  return !s.sh_full;
}

void Engine::draft_head(int n, int E, const uint8_t* const* masks) {
  const ModelConfig& c = model_->config();
  const DraftConfig& d = draft_->config();
  const int K16 = d.selector_top_k;
  const Fp4Weight& lm = model_->lm_head();
  bool any_sparse = false;
  for (int i = 0; i < n; ++i) any_sparse |= masks && masks[i];
  if (!any_sparse) {  // the full head over every row, as before the sparse head (BACKLOG 19)
    linear_fp4(lm, dh_, n * E, dlogits_);
    kernels::topk_rows(dlogits_, n * E, c.vocab, K16, topk_scratch_, dunary_, dcand_, stream_);
    return;
  }
  // Each sequence as it is drafted alone: its own selection, or the full head for a session in fallback.
  kernels::SparseHeadBufs b;
  b.w = lm.w, b.table = lm.table, b.raw = lm.scales, b.scale2 = lm.scale2, b.N = c.vocab, b.K = d.hidden;
  b.us = sh_us_, b.vt = sh_vt_, b.R = sh_rank_, b.dynamic = opts_.sparse_head_dynamic, b.cap = sh_cap_;
  b.p = sh_p_, b.scratch = sh_scratch_, b.rows = sh_rows_, b.nrows = sh_nrows_, b.xtmp = sh_xtmp_;
  b.xh = xh_, b.xinv = xinv_, b.logits = dlogits_, b.topk_scratch = topk_scratch_;
  kernels::draft_head_rows(b, dh_, n, E, masks, K16, dunary_, dcand_, stream_);
}

bool Engine::tree_step(const SamplingParams& p, bool use_lookup, bool want_logits, int B) const {
  (void)p;
  const int rows = std::min(B, opts_.tree_chain + 1) + opts_.tree_nodes;
  return opts_.tree_nodes > 0 && !use_lookup && !want_logits && B == opts_.draft_block &&
         rows <= kernels::kMaxStreamRows && s_->pos + rows <= opts_.max_context;
}

bool Engine::can_speculate(const SamplingParams& p) const {
  // Penalties are applied over the full vocabulary on the host; the accept step sees only each row's
  // top-K. Requests that set them decode without speculation.
  if (!draft_ || p.presence_penalty != 0.f || p.repetition_penalty != 1.f) return false;
  // Sampling needs the target's whole truncated distribution from the top 64 logits.
  return p.temperature <= 0.f || (p.top_k > 0 && p.top_k <= 64);
}

void Engine::set_step_graphs(bool on) {
  if (!on) clear_graphs();
  graphs_ = on && graphs_allowed_;
}

void Engine::clear_graphs() {
  if (commit_stream_) cudaStreamSynchronize(commit_stream_);  // a replay may still run there
  for (Session& s : sessions_) {
    if (s.draft_graph) cudaGraphExecDestroy(s.draft_graph);
    s.draft_graph = nullptr;
    if (s.draft_graph_sparse) cudaGraphExecDestroy(s.draft_graph_sparse);
    s.draft_graph_sparse = nullptr;
    for (auto& [k, g] : s.verify_graphs) cudaGraphExecDestroy(g);
    s.verify_graphs.clear();
    for (auto& [k, g] : s.tree_verify_graphs) cudaGraphExecDestroy(g);
    s.tree_verify_graphs.clear();
    for (cudaGraphExec_t g : s.commit_graphs)
      if (g) cudaGraphExecDestroy(g);
    s.commit_graphs.clear();
    for (auto& [k, g] : s.batch_commit_graphs) cudaGraphExecDestroy(g);
    s.batch_commit_graphs.clear();
    for (auto& [k, g] : s.state_commit_graphs) cudaGraphExecDestroy(g);
    s.state_commit_graphs.clear();
  }
  for (auto& [order, bg] : batch_graphs_) {
    if (bg.draft) cudaGraphExecDestroy(bg.draft);
    for (auto& [k, g] : bg.draft_sparse) cudaGraphExecDestroy(g);
    for (auto& [k, g] : bg.verify) cudaGraphExecDestroy(g);
  }
  batch_graphs_.clear();
}

cudaGraphExec_t Engine::capture(const std::function<void()>& body, cudaStream_t st) {
  cudaGraph_t graph = nullptr;
  // Thread-local: ling-serve's other threads may make CUDA calls while the worker captures.
  check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
  use_dev_pos_ = true;
  try {
    body();
  } catch (...) {
    use_dev_pos_ = false;
    cudaStreamEndCapture(st, &graph);  // ends the invalidated capture
    if (graph) cudaGraphDestroy(graph);
    cudaGetLastError();
    throw;
  }
  use_dev_pos_ = false;
  check(cudaStreamEndCapture(st, &graph), "end capture");
  cudaGraphExec_t exec = nullptr;
  const cudaError_t e = cudaGraphInstantiate(&exec, graph, fork_ >= 2 ? cudaGraphInstantiateFlagUseNodePriority : 0);
  cudaGraphDestroy(graph);
  check(e, "graph instantiate");
  return exec;
}

void Engine::run_graph(cudaGraphExec_t& slot, const char* what, const std::function<void()>& body,
                       cudaStream_t st) {
  if (!st) st = stream_;
  if (graphs_ && !slot) {
    try {
      slot = capture(body, st);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "step graphs off: capturing the %s failed: %s\n", what, e.what());
      clear_graphs();
      graphs_ = false;
    }
  }
  if (graphs_ && slot) {
    NvtxRange range(what);  // the same names as the eager ranges, so a trace attributes graph kernels alike
    check(cudaGraphLaunch(slot, st), what);
    return;
  }
  body();
}

void Engine::commit(int row0, int n, cudaStream_t st, float* out) {
  const ModelConfig& c = model_->config();
  const size_t C = c.lin_conv_channels(), r0 = row0;
  for (int li = 0; li < c.layers; ++li) {
    const LayerWeights& L = model_->layers()[li];
    if (L.full) continue;  // the verify already wrote the KV; rows past pos + n are overwritten later
    const int slot = layer_slot_[li];
    // The conv window advances over the accepted rows' raw inputs, the recurrence over their conv outputs.
    // In a batch the sequence's rows start at row0; the conv runs in place over its own rows only.
    kernels::gdn_conv(v_pre_[slot] + r0 * C, s_->conv[slot], L.conv, n, c.lin_conv_channels(), st);
    kernels::gdn_recurrent(v_post_[slot] + r0 * C, v_g_[slot] + r0 * c.lin_v_heads, v_beta_[slot] + r0 * c.lin_v_heads,
                           s_->gdn[slot], s_->gdn[slot], out, n, c.lin_k_heads, c.lin_v_heads, st);
  }
}

void Engine::draft_materialize(int rows, kernels::DevPos pos0, int cap_row0) {
  if (rows <= 0) return;
  const DraftConfig& d = draft_->config();
  const int dkv = d.kv_heads * d.head_dim;
  // fc over the concatenated target features (already BF16), then the hidden norm. The context goes
  // straight to each layer's K and V: no input norm, no convolution (SGLang's prepare_context_hidden_for_kv).
  // A prefill's rows (quantized_) always take the BF16 GEMM, as the target's projections take the prefill path
  // whatever the chunk's size: a prompt's last few rows (a chunk of <= 32) must give the drafter the context a larger
  // chunk gives them, or the speculative continuation depends on the chunking (--chunk-check).
  const Fp4Weight& fc4 = draft_->fc4();
  if (!quantized(rows) && fc4.w) {  // BACKLOG 33: the NVFP4 copy on the rows path
    kernels::bf16_to_half_rows(cap_ + size_t(cap_row0) * cap_stride_, rows, fc4.K, xh_fc_, xinv_, stream_);
    kernels::stream_gemm_nvfp4(xh_fc_, xinv_, rows, fc4.w, fc4.table, fc4.scale2, dctx_, fc4.N, fc4.K, stream_);
  } else if (!quantized(rows))
    kernels::bf16_rows_bf16in(cap_ + size_t(cap_row0) * cap_stride_, rows, draft_->fc().w, dctx_, d.hidden, cap_stride_,
                              stream_);
  else
    kernels::gemm_bf16(cap_ + size_t(cap_row0) * cap_stride_, rows, draft_->fc().w, dctx_, d.hidden, cap_stride_,
                       stream_);
  kernels::rmsnorm(dctx_, draft_->hidden_norm(), dctx_, rows, d.hidden, d.eps, false, stream_);
  for (int l = 0; l < d.layers; ++l) {
    const DraftLayer& L = draft_->layers()[l];
    linear_fp4_multi({{&L.k, dk_}, {&L.v, dv_}}, dctx_, rows);
    kernels::draft_qk_rope(dk_, rows, d.kv_heads, dkv, L.k_norm, pos0, d.rope_theta, d.eps, stream_);
    kernels::draft_store_kv(dk_, dv_, rows, dkv, pos0, s_->dkc[l], s_->dvc[l], ring_, stream_);
  }
}

void Engine::linear_draft_kproj(const Bf16Weight& w, const Fp4Weight& w4, const float* x, int M, float* y) {
  if (!w4.w) return linear_bf16_drafter(w, x, M, y);
  // Without the BF16 original only the rows path is there; a drafter pass never has more rows (pin_->draft_ids).
  if (M > kernels::kMaxStreamRows) throw std::runtime_error("the drafter's NVFP4 conv projection over > 32 rows");
  kernels::to_half_rows(x, M, w4.K, xh_, xinv_, stream_);
  kernels::stream_gemm_nvfp4(xh_, xinv_, M, w4.w, w4.table, w4.scale2, y, w4.N, w4.K, stream_);
}

void Engine::draft_launch(int B, bool sparse) {
  NvtxRange range("draft");
  // The drafter's block always takes the rows path. quantized_ is whatever the last forward() left: after a
  // prefill it is set, which made the first draft after each prompt take the quantized prefill path (and a
  // draft graph captured on that step would have kept it for every later step).
  quantized_ = false;
  const ModelConfig& c = model_->config();
  const DraftConfig& d = draft_->config();
  const int H = d.hidden, E = B - 1, dq = d.heads * d.head_dim, dkv = d.kv_heads * d.head_dim;
  const int K16 = d.selector_top_k;
  check(cudaMemcpyAsync(ids_dev_, pin_->draft_ids, B * sizeof(int), cudaMemcpyHostToDevice, stream_), "draft ids");
  kernels::embed(model_->embed(), ids_dev_, B, H, dres_, stream_);
  for (int l = 0; l < d.layers; ++l) {
    const DraftLayer& L = draft_->layers()[l];
    // Pre-norm with the residual added in (SGLang's fused add + RMSNorm); the drafter's norms are plain.
    if (l > 0) kernels::add_inplace(dres_, dout_, B * H, stream_);
    kernels::rmsnorm(dres_, L.input_norm, dh_, B, H, d.eps, false, stream_);
    // Attention wrapped in the grouped conv: input side 0, output side 1, one kernel projection.
    linear_draft_kproj(L.attn_kproj, L.attn_kproj4, dh_, B, dcoef_);
    kernels::grouped_conv(dh_, dcoef_, L.attn_base, 0, dh2_, B, H, d.conv_groups, B, stream_);
    linear_fp4_multi({{&L.q, dq_}, {&L.k, dk_}, {&L.v, dv_}}, dh2_, B);
    kernels::draft_qk_rope(dq_, B, d.heads, dq, L.q_norm, dpos(), d.rope_theta, d.eps, stream_);
    kernels::draft_qk_rope(dk_, B, d.kv_heads, dkv, L.k_norm, dpos(), d.rope_theta, d.eps, stream_);
    kernels::draft_attention(dq_, s_->dkc[l], s_->dvc[l], ring_, dk_, dv_, dpos(), B, d.sliding_window - 1, d.heads,
                             d.kv_heads, dattn_, stream_);
    linear_fp4(L.o, dattn_, B, dh2_);
    kernels::grouped_conv(dh2_, dcoef_, L.attn_base, 1, dout_, B, H, d.conv_groups, B, stream_);
    kernels::add_inplace(dres_, dout_, B * H, stream_);
    kernels::rmsnorm(dres_, L.post_norm, dh_, B, H, d.eps, false, stream_);
    // The MLP, wrapped the same way.
    linear_draft_kproj(L.mlp_kproj, L.mlp_kproj4, dh_, B, dcoef_);
    kernels::grouped_conv(dh_, dcoef_, L.mlp_base, 0, dh2_, B, H, d.conv_groups, B, stream_);
    linear_fp4_multi({{&L.gate, dg_}, {&L.up, du_}}, dh2_, B);
    kernels::silu_mul(dg_, du_, dg_, B * d.intermediate, stream_);
    linear_fp4(L.down, dg_, B, dh2_);
    kernels::grouped_conv(dh2_, dcoef_, L.mlp_base, 1, dout_, B, H, d.conv_groups, B, stream_);
  }
  kernels::add_inplace(dres_, dout_, B * H, stream_);
  // Drafted positions 1 .. B - 1: the final norm, the target's LM head (no target final norm), the top 16,
  // the selector's projection and lattice (its first predecessor, the anchor, read from ids_dev_[0]).
  kernels::rmsnorm(dres_ + H, draft_->norm(), dh_, E, H, d.eps, false, stream_);
  const uint8_t* mask = sparse ? s_->sh_mask : nullptr;
  draft_head(1, E, &mask);
  linear_bf16_drafter(draft_->selector_projection(), dh_, E, dhp_);
  kernels::selector_lattice(dhp_, dcand_, dunary_, draft_->predecessor_codebook(), draft_->successor_codebook(),
                            pin_->draft_ids[0], E, dscores_, stream_, ids_dev_);
  check(cudaMemcpyAsync(pin_->cand, dcand_, size_t(E) * K16 * sizeof(int), cudaMemcpyDeviceToHost, stream_),
        "draft candidates");
  check(cudaMemcpyAsync(pin_->scores, dscores_, size_t(E) * K16 * K16 * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_),
        "draft scores");
}

void Engine::draft_batch_launch(std::span<StepPlan> plans, int B, unsigned sparse) {
  NvtxRange range("draft");
  quantized_ = false;  // the rows path (see draft_launch)
  const ModelConfig& c = model_->config();
  const DraftConfig& d = draft_->config();
  const int n = static_cast<int>(plans.size()), R = n * B, H = d.hidden, E = B - 1, dq = d.heads * d.head_dim;
  const int dkv = d.kv_heads * d.head_dim, K16 = d.selector_top_k;
  // Every sequence's block of B rows, one after the other; sequence i at its own position (pos_dev_[i] in a graph).
  kernels::SeqBatch sb;
  sb.n = n;
  for (int i = 0; i < n; ++i) {
    sb.row0[i + 1] = (i + 1) * B;
    sb.pos[i] = plans[i].s->pos;
  }
  if (use_dev_pos_) sb.pos_dev = pos_dev_;
  check(cudaMemcpyAsync(ids_dev_, pin_->draft_ids, R * sizeof(int), cudaMemcpyHostToDevice, stream_), "draft ids");
  kernels::embed(model_->embed(), ids_dev_, R, H, dres_, stream_);
  for (int l = 0; l < d.layers; ++l) {
    const DraftLayer& L = draft_->layers()[l];
    kernels::SeqPtrs<const __nv_bfloat16> kc, vc;
    for (int i = 0; i < n; ++i) {
      kc.p[i] = plans[i].s->dkc[l];
      vc.p[i] = plans[i].s->dvc[l];
    }
    // draft_launch's layer over R rows: the per-row kernels once, the grouped conv block-wise (its second tap
    // never crosses a block), RoPE and attention per sequence through sb.
    if (l > 0) kernels::add_inplace(dres_, dout_, R * H, stream_);
    kernels::rmsnorm(dres_, L.input_norm, dh_, R, H, d.eps, false, stream_);
    linear_draft_kproj(L.attn_kproj, L.attn_kproj4, dh_, R, dcoef_);
    kernels::grouped_conv(dh_, dcoef_, L.attn_base, 0, dh2_, R, H, d.conv_groups, B, stream_);
    linear_fp4_multi({{&L.q, dq_}, {&L.k, dk_}, {&L.v, dv_}}, dh2_, R);
    kernels::draft_qk_rope_seqs(dq_, d.heads, dq, L.q_norm, sb, d.rope_theta, d.eps, stream_);
    kernels::draft_qk_rope_seqs(dk_, d.kv_heads, dkv, L.k_norm, sb, d.rope_theta, d.eps, stream_);
    kernels::draft_attention_seqs(dq_, kc, vc, ring_, dk_, dv_, sb, d.sliding_window - 1, d.heads, d.kv_heads, dattn_,
                                  stream_);
    linear_fp4(L.o, dattn_, R, dh2_);
    kernels::grouped_conv(dh2_, dcoef_, L.attn_base, 1, dout_, R, H, d.conv_groups, B, stream_);
    kernels::add_inplace(dres_, dout_, R * H, stream_);
    kernels::rmsnorm(dres_, L.post_norm, dh_, R, H, d.eps, false, stream_);
    linear_draft_kproj(L.mlp_kproj, L.mlp_kproj4, dh_, R, dcoef_);
    kernels::grouped_conv(dh_, dcoef_, L.mlp_base, 0, dh2_, R, H, d.conv_groups, B, stream_);
    linear_fp4_multi({{&L.gate, dg_}, {&L.up, du_}}, dh2_, R);
    kernels::silu_mul(dg_, du_, dg_, R * d.intermediate, stream_);
    linear_fp4(L.down, dg_, R, dh2_);
    kernels::grouped_conv(dh2_, dcoef_, L.mlp_base, 1, dout_, R, H, d.conv_groups, B, stream_);
  }
  kernels::add_inplace(dres_, dout_, R * H, stream_);
  // Each sequence's drafted positions 1 .. B - 1, gathered to rows i * E .. of dh_ by its final norm; then the LM
  // head, the top 16 and the selector's projection over all n * E rows, and each sequence's lattice from its anchor.
  for (int i = 0; i < n; ++i)
    kernels::rmsnorm(dres_ + (size_t(i) * B + 1) * H, draft_->norm(), dh_ + size_t(i) * E * H, E, H, d.eps, false,
                     stream_);
  // BACKLOG 19: bit `i` of `sparse` set: sequence i drafts with the sparse head over its own session's mask, exactly
  // as it does alone; clear: the full head.
  const uint8_t* masks[kernels::kMaxSeqs] = {};
  for (int i = 0; i < n; ++i) masks[i] = (sparse >> i) & 1u ? plans[i].s->sh_mask : nullptr;
  draft_head(n, E, masks);
  linear_bf16_drafter(draft_->selector_projection(), dh_, n * E, dhp_);
  const int rank = d.selector_rank;
  for (int i = 0; i < n; ++i)
    kernels::selector_lattice(dhp_ + size_t(i) * E * rank, dcand_ + size_t(i) * E * K16, dunary_ + size_t(i) * E * K16,
                              draft_->predecessor_codebook(), draft_->successor_codebook(), pin_->draft_ids[i * B], E,
                              dscores_ + size_t(i) * E * K16 * K16, stream_, ids_dev_ + i * B);
  check(cudaMemcpyAsync(pin_->cand, dcand_, size_t(n) * E * K16 * sizeof(int), cudaMemcpyDeviceToHost, stream_),
        "draft candidates");
  check(cudaMemcpyAsync(pin_->scores, dscores_, size_t(n) * E * K16 * K16 * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_),
        "draft scores");
}

void Engine::draft_propose(int anchor, int B, bool graph) {
  pin_->draft_ids[0] = anchor;
  for (int i = 1; i < B; ++i) pin_->draft_ids[i] = draft_->config().mask_token;
  const bool sparse = sparse_head_sync(*s_);  // BACKLOG 19: the head for this draft
  if (graph) run_graph(sparse ? s_->draft_graph_sparse : s_->draft_graph, "draft", [&] { draft_launch(B, sparse); });
  else draft_launch(B, sparse);
  s_->sh_last_sparse = sparse;
  if (sparse) ++s_->spec.sparse_steps;
  check(cudaStreamSynchronize(stream_), "draft");
  // BACKLOG 19's offline input (bench/sparse_head): LING_DUMP_DRAFT_H=FILE appends each one-stream draft's rows' final
  // hidden states (E x hidden floats, the LM head's input) and their top-16 candidates (E x 16 ints).
  static FILE* dump = [] { const char* f = std::getenv("LING_DUMP_DRAFT_H"); return f ? std::fopen(f, "ab") : nullptr; }();
  static long dump_left = [] { const char* n = std::getenv("LING_DUMP_DRAFT_H_STEPS"); return n ? std::atol(n) : 1L << 40; }();
  if (dump && dump_left-- > 0) {
    const int E = B - 1, H = draft_->config().hidden, K16 = draft_->config().selector_top_k;
    std::vector<float> h(size_t(E) * H);
    check(cudaMemcpy(h.data(), dh_, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "dump");
    const int hdr[3] = {E, H, K16};
    std::fwrite(hdr, sizeof(int), 3, dump);
    std::fwrite(h.data(), sizeof(float), h.size(), dump);
    std::fwrite(pin_->cand, sizeof(int), size_t(E) * K16, dump);
    std::fflush(dump);
  }
}

void Engine::verify_launch(int V, int K, bool tree) {
  const ModelConfig& c = model_->config();
  if (tree) {  // the tree's description (BACKLOG 26), read by the attention and DeltaNet kernels of this pass
    check(cudaMemcpyAsync(tree_dev_, &pin_->tree, sizeof(kernels::TreeDesc), cudaMemcpyHostToDevice, stream_), "tree");
    tree_ = tree_dev_;
  }
  try {
    forward(pin_->rows, V, Pass::Verify);
  } catch (...) {
    tree_ = nullptr;
    throw;
  }
  tree_ = nullptr;
  kernels::topk_rows(logits_dev_, V, c.vocab, K, topk_scratch_, vals_dev_, ids_out_dev_, stream_);
  // The values too when greedy (V floats): the host checks them for NaN and inf after the step.
  check(cudaMemcpyAsync(pin_->topk_vals, vals_dev_, size_t(V) * K * sizeof(float), cudaMemcpyDeviceToHost, stream_),
        "topk");
  check(cudaMemcpyAsync(pin_->topk_ids, ids_out_dev_, size_t(V) * K * sizeof(int), cudaMemcpyDeviceToHost, stream_),
        "topk");
}

void Engine::commit_launch(int row0, int n) {
  NvtxRange range("commit");
  quantized_ = false;  // at most 32 rows: the rows path, whatever the last forward() left (see draft_launch)
  if (!commit_overlap_) commit(row0, n, stream_, core_);
  draft_materialize(n, dpos(), row0);  // the verify's target features of those rows
}

void Engine::commit_state_launch(int row0, int n, cudaStream_t st) {
  NvtxRange range("commit state");
  commit(row0, n, st, commit_core_);
}

// The DeltaNet replay of the active session's rows row0 .. row0 + n - 1 on commit_stream_ (graph or not), ordered
// after the verify that wrote their inputs (the host has synchronized stream_ since) and before the next pass.
void Engine::commit_state_async(int row0, int n, bool graph) {
  // Every replay queues on commit_stream_, so a batch's sequences replay in order and commit_done_ (recorded after
  // the last) covers them all.
  if (graph) run_graph(s_->state_commit_graphs[{row0, n}], "commit state", [&] { commit_state_launch(row0, n, commit_stream_); },
                       commit_stream_);
  else commit_state_launch(row0, n, commit_stream_);
  check(cudaEventRecord(commit_done_, commit_stream_), "commit event");
  commit_pending_ = true;
}

void Engine::plan_begin(StepPlan& sp, int B) {
  Session& s = *sp.s;
  const SamplingParams& p = *sp.p;
  sp.greedy = p.temperature <= 0.f;
  sp.E = B - 1;
  if (p.seed != 0) s.rng.seed(p.seed + s.history.size());
  // 0. The context lookup: verified instead of the drafter's chain on a long verbatim match (auto), or
  // only recorded and scored later (shadow).
  if (opts_.lookup_mode > 0) {
    std::vector<int> seq(s.history);
    seq.push_back(sp.anchor);
    sp.lp = lookup_propose(seq, sp.E);
    // An image's ids (VISION §20.7) are never proposed: the proposal stops before the first.
    const int V = model_->config().vocab;
    auto bad = std::find_if(sp.lp.tokens.begin(), sp.lp.tokens.end(), [V](int t) { return t < 0 || t >= V; });
    sp.lp.tokens.erase(bad, sp.lp.tokens.end());
    if (sp.lp.tokens.empty()) sp.lp.match = 0;
  }
  sp.use_lookup = opts_.lookup_mode == 2 && !sp.lp.tokens.empty() && sp.lp.match >= opts_.lookup_min_match;
  sp.rows.assign(1, sp.anchor);
  if (sp.use_lookup) {
    sp.rows.insert(sp.rows.end(), sp.lp.tokens.begin(), sp.lp.tokens.end());
    sp.E = static_cast<int>(sp.lp.tokens.size());
  }
  sp.V = sp.E + 1;
  sp.K = sp.greedy ? 1 : p.top_k;
}

void Engine::plan_draft(StepPlan& sp, int B, bool graph) {
  // 1. Draft (the active session must be sp.s), and walk the selector's lattice (SGLang's sample_path).
  draft_propose(sp.anchor, B, graph);
  if (hidden_dump_file() && dump_file()) {  // LING_DUMP_HIDDEN: the drafted positions' normed hidden states and selector
    const auto& d = draft_->config();  // projections, for the acceptance probe (BACKLOG 26, follow-up 2b)
    const int E = B - 1;
    sp.hidden.resize(size_t(E) * d.hidden);
    sp.sel.resize(size_t(E) * d.selector_rank);
    check(cudaMemcpy(sp.hidden.data(), dh_, sp.hidden.size() * sizeof(float), cudaMemcpyDeviceToHost), "hidden dump");
    check(cudaMemcpy(sp.sel.data(), dhp_, sp.sel.size() * sizeof(float), cudaMemcpyDeviceToHost), "selector dump");
  }
  plan_walk(sp, B, pin_->cand, pin_->scores);
}

void Engine::plan_walk(StepPlan& sp, int B, const int* cand, const float* scores_host) {
  const SamplingParams& p = *sp.p;
  const int K16 = draft_->config().selector_top_k, E = sp.E;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  sp.cand.assign(cand, cand + size_t(E) * K16);
  sp.rows.resize(B);
  sp.q.assign(E, std::vector<double>(K16, 0.0));
  sp.picks.assign(E, 0);
  int prev = 0;
  for (int e = 0; e < E; ++e) {
    const float* sc = scores_host + (size_t(e) * K16 + (e == 0 ? 0 : prev)) * K16;
    std::vector<double>& q = sp.q[e];
    int pick = 0;
    if (sp.greedy) {
      for (int k = 1; k < K16; ++k)
        if (sc[k] > sc[pick]) pick = k;  // ties to the lower index
      q[pick] = 1.0;
    } else {
      const double T = std::max(double(p.temperature), 1e-5);
      double mx = sc[0];
      for (int k = 1; k < K16; ++k) mx = std::max(mx, double(sc[k]));
      double sum = 0;
      for (int k = 0; k < K16; ++k) sum += q[k] = std::exp((sc[k] - mx) / T);
      for (double& v : q) v /= sum;
      const double u = uni(sp.s->rng);
      double cum = 0;
      pick = K16 - 1;
      for (int k = 0; k < K16; ++k) {
        cum += q[k];
        if (u < cum) {
          pick = k;
          break;
        }
      }
    }
    sp.rows[e + 1] = sp.cand[size_t(e) * K16 + pick];
    sp.picks[e] = pick;
    prev = pick;
  }
}

void Engine::plan_accept(StepPlan& sp, const float* vals, const int* ids) {
  // On the host, after the step (graph or not): no verified row may be NaN or inf (core/sampling.hpp).
  const int V = sp.V, K = sp.K, E = sp.E;
  require_finite_topk({vals, size_t(V) * K}, {ids, size_t(V) * K}, model_->config().vocab);
  if (sp.tree) {  // BACKLOG 26: the accepted path through the tree (core/tree.hpp)
    if (sp.greedy) {
      sp.path = accept_tree_greedy(sp.tr, ids, sp.next);
    } else {
      const int K16 = draft_->config().selector_top_k;
      auto q_of = [&](int e, int token) {  // the chain's draft probability of `token` at drafted position e
        double qt = 0;
        const int* cand = sp.cand.data() + size_t(e) * K16;
        for (int k = 0; k < K16; ++k)
          if (cand[k] == token) qt += sp.q[e][k];
        return qt;
      };
      std::vector<Dist> P(V);
      for (int i = 0; i < V; ++i) P[i] = target_dist(vals + size_t(i) * K, ids + size_t(i) * K, K, *sp.p);
      sp.path = accept_tree_sampled(sp.tr, P, q_of, sp.s->rng, sp.next);
    }
    sp.accepted = static_cast<int>(sp.path.size()) - 1;
    return;
  }
  // 3. Accept.
  if (sp.greedy) {  // K = 1: ids holds each row's argmax
    int accepted = 0;
    while (accepted < E && sp.rows[accepted + 1] == ids[accepted]) ++accepted;
    sp.accepted = accepted;
    sp.next = ids[accepted];
    return;
  }
  const int K16 = draft_->config().selector_top_k;
  auto q_of = [&](int e, int token) {  // the draft's probability of `token` at drafted position e
    if (sp.use_lookup) return token == sp.rows[e + 1] ? 1.0 : 0.0;  // a deterministic proposal
    double qt = 0;
    const int* cand = sp.cand.data() + size_t(e) * K16;
    for (int k = 0; k < K16; ++k)
      if (cand[k] == token) qt += sp.q[e][k];
    return qt;
  };
  std::vector<Dist> P(V);
  for (int i = 0; i < V; ++i) P[i] = target_dist(vals + size_t(i) * K, ids + size_t(i) * K, K, *sp.p);
  const auto [acc, tok] = accept_sampled(P, std::vector<int>(sp.rows.begin() + 1, sp.rows.end()), q_of, sp.s->rng);
  sp.accepted = acc;
  sp.next = tok;
}

void Engine::plan_count(StepPlan& sp) {
  Session& s = *sp.s;
  const int accepted = sp.accepted, n = accepted + 1;
  s.spec.steps += 1;
  s.spec.drafted += sp.V - 1;
  if (sp.tree) {
    int off = 0;
    for (size_t j = 0; j < sp.path.size(); ++j) off += sp.path[j] != static_cast<int>(j);
    s.spec.tree_steps += 1;
    s.spec.tree_branch_steps += off > 0;
    s.spec.tree_off_chain += off;
  }
  s.spec.accepted += accepted;
  s.spec.accept_hist[accepted] += 1;
  if (sp.use_lookup) {
    s.spec.lookup_steps += 1;
    s.spec.lookup_accepted += accepted;
  } else if (opts_.lookup_mode == 1 && !sp.lp.tokens.empty()) {
    const int m = sp.lp.match;
    const int bucket = m >= 32 ? 4 : m >= 16 ? 3 : m >= 8 ? 2 : m >= 4 ? 1 : 0;
    s.shadow.push_back({s.pos - n, std::move(sp.lp.tokens), m, accepted, bucket});
  }
  if (!sp.use_lookup && opts_.lookup_mode == 1) {
    s.spec.shadow_total_steps += 1;
    s.spec.shadow_dflash_all += accepted;
    if (s.shadow.empty() || s.shadow.back().pos != s.pos - n) s.spec.shadow_best += accepted;  // no proposal
  }
  score_shadow(s, false);
}

std::vector<int> Engine::speculate(int anchor, const SamplingParams& p, std::vector<float>* row_logits) {
  if (!can_speculate(p)) throw std::runtime_error("speculate: unsupported sampling parameters or no drafter");
  const int room = opts_.max_context - s_->pos;
  if (room < 1) throw std::runtime_error("context is longer than max_context");
  const int B = std::min(opts_.draft_block, room);
  if (B < 2) {  // no room to draft: one plain step
    step(anchor);
    if (row_logits) *row_logits = logits_;
    return {sample(logits_, p, s_->history)};
  }
  // P4b: the KV pages of every row the verify may write, before anything of the step runs (all or nothing: a short
  // pool throws KvPoolExhausted with nothing drawn or changed), so a replayed verify graph finds them mapped.
  kv_ensure(*s_, s_->pos, s_->pos + kv_step_rows(*s_, true), stream_);
  // Graphs have the block's shape and the drafter's chain; a short block at the context's end or a lookup
  // chain runs eagerly. Either way the same kernels run, so the result is the same.
  const bool graph = graphs_ && B == opts_.draft_block && !(opts_.lookup_mode == 2);
  auto t0 = std::chrono::steady_clock::now();
  StepPlan sp;
  sp.s = s_;
  sp.p = &p;
  sp.anchor = anchor;
  plan_begin(sp, B);
  if (graph) {  // the position every graph of this step reads
    pin_->pos = s_->pos;
    check(cudaMemcpyAsync(pos_dev_, &pin_->pos, sizeof(int), cudaMemcpyHostToDevice, stream_), "position");
    pin_->rdelta = s_->rope_delta;  // the rope delta after the prompt's images (VISION §20.3 (b))
    check(cudaMemcpyAsync(rdelta_dev_, &pin_->rdelta, sizeof(int), cudaMemcpyHostToDevice, stream_), "rope delta");
  }
  if (!sp.use_lookup) plan_draft(sp, B, graph);
  if (tree_step(p, sp.use_lookup, row_logits != nullptr, B)) {  // BACKLOG 26: the chain and its tree
    const int K16 = draft_->config().selector_top_k;
    TreeShape shape;
    shape.nodes = opts_.tree_nodes;
    shape.temp = opts_.tree_temp;
    shape.width = opts_.tree_width;
    shape.max_depth = opts_.tree_max_depth;
    shape.chain = opts_.tree_chain;
    sp.tr = build_tree(anchor, sp.cand.data(), pin_->scores, sp.E, K16,
                       std::vector<int>(sp.rows.begin() + 1, sp.rows.end()), sp.picks, shape);
    if (sp.tr.rows() == std::min(B, opts_.tree_chain + 1) + opts_.tree_nodes &&  // a full tree (the graph's shape),
        sp.tr.extra_mass >= opts_.tree_min_mass) {                              // worth its rows
      describe_tree(sp.tr, pin_->tree);
      sp.tree = true;
      sp.rows = sp.tr.token;
      sp.V = sp.tr.rows();
    }
  }
  const int V = sp.V, K = sp.K;  // rows verified, top K per row
  kv_ensure(*s_, s_->pos, s_->pos + V, stream_);  // P4b: mapped above (kv_step_rows bounds V); a no-op check
  auto t1 = std::chrono::steady_clock::now();
  s_->spec.draft_seconds += std::chrono::duration<double>(t1 - t0).count();

  // 2. Verify, with each row's top K (the argmax when greedy) on the GPU.
  std::copy(sp.rows.begin(), sp.rows.end(), pin_->rows);
  join_commit();  // the previous step's DeltaNet replay, beside this step's draft until here
  if (graph) {
    cudaGraphExec_t& g = (sp.tree ? s_->tree_verify_graphs : s_->verify_graphs)[K];
    run_graph(g, "verify", [&] { verify_launch(V, K, sp.tree); });
  } else {
    verify_launch(V, K, sp.tree);
  }
  check(cudaStreamSynchronize(stream_), "verify");
  plan_accept(sp, pin_->topk_vals, pin_->topk_ids);  // 3. accept
  if (sp.tree) {
    // The accepted path's rows to rows 0 .. n - 1 (KV slots, DeltaNet inputs, target features), so the commit is
    // the chain's; its tokens become the step's rows.
    const int n = static_cast<int>(sp.path.size());
    bool moved = false;
    for (int j = 0; j < n; ++j) moved = moved || sp.path[j] != j;
    if (moved) {
      std::copy(sp.path.begin(), sp.path.end(), pin_->gather_rows);
      check(cudaMemcpyAsync(gather_rows_dev_, pin_->gather_rows, n * sizeof(int), cudaMemcpyHostToDevice, stream_),
            "tree path");
      kernels::tree_gather(s_->gather, s_->ngather, gather_rows_dev_, n, kernels::DevPos(s_->pos), stream_);
      check(cudaStreamSynchronize(stream_), "tree gather");
    }
    std::vector<int> toks(n);
    for (int j = 0; j < n; ++j) toks[j] = sp.tr.token[sp.path[j]];
    sp.rows = toks;
  }
  if (std::FILE* f = dump_file(); f && !sp.use_lookup && !sp.tree) {
    const int K16 = draft_->config().selector_top_k;
    const int hdr[9] = {0x54524545, s_->pos, anchor, sp.E, K16, K, sp.accepted, sp.next, sp.greedy ? 1 : 0};
    std::fwrite(hdr, sizeof(int), 9, f);
    std::fwrite(sp.cand.data(), sizeof(int), size_t(sp.E) * K16, f);
    std::fwrite(pin_->scores, sizeof(float), size_t(sp.E) * K16 * K16, f);
    std::fwrite(sp.rows.data(), sizeof(int), size_t(V), f);
    std::fwrite(pin_->topk_vals, sizeof(float), size_t(V) * K, f);
    std::fwrite(pin_->topk_ids, sizeof(int), size_t(V) * K, f);
    std::fflush(f);
    if (std::FILE* h = hidden_dump_file(); h && !sp.hidden.empty()) {
      const auto& d = draft_->config();
      const int hh[5] = {0x48494444, s_->pos, sp.E, d.hidden, d.selector_rank};
      std::fwrite(hh, sizeof(int), 5, h);
      std::fwrite(sp.hidden.data(), sizeof(float), sp.hidden.size(), h);
      std::fwrite(sp.sel.data(), sizeof(float), sp.sel.size(), h);
      std::fflush(h);
    }
  }
  auto t2 = std::chrono::steady_clock::now();
  s_->spec.verify_seconds += std::chrono::duration<double>(t2 - t1).count();

  // 4. Commit the anchor and the accepted drafts.
  const int n = sp.accepted + 1;
  if (row_logits) {  // the rows the returned tokens were chosen from: row i chose rows[i + 1], row accepted chose next
    const size_t vocab = model_->config().vocab;
    row_logits->resize(size_t(n) * vocab);
    check(cudaMemcpyAsync(row_logits->data(), logits_dev_, size_t(n) * vocab * sizeof(float), cudaMemcpyDeviceToHost,
                          stream_),
          "row logits");
    check(cudaStreamSynchronize(stream_), "row logits");
  }
  if (commit_overlap_) commit_state_async(0, n, graph);
  if (graph) {
    if (s_->commit_graphs.size() < size_t(V) + 1) s_->commit_graphs.resize(size_t(V) + 1, nullptr);
    run_graph(s_->commit_graphs[n], "commit", [&] { commit_launch(0, n); });
  } else {
    commit_launch(0, n);
  }
  s_->history.insert(s_->history.end(), sp.rows.begin(), sp.rows.begin() + n);
  s_->pos += n;
  check(cudaStreamSynchronize(stream_), "commit");
  s_->spec.commit_seconds += seconds_since(t2);
  plan_count(sp);

  std::vector<int> out(sp.rows.begin() + 1, sp.rows.begin() + n);
  out.push_back(sp.next);
  return out;
}

bool Engine::can_batch(int session, const SamplingParams& p) const {
  const Session& s = sessions_.at(session);
  return can_speculate(p) && opts_.max_context - s.pos >= opts_.draft_block;
}

void Engine::verify_batch_launch(std::span<StepPlan> plans, int V) {
  const ModelConfig& c = model_->config();
  std::vector<Seg> segs;
  for (const StepPlan& sp : plans) segs.push_back({sp.s, sp.row0, sp.V});
  forward_rows(pin_->rows, V, Pass::Verify, -1, segs);
  // Each sequence's top K over its own rows (K differs between a greedy and a sampled one), at 64 values per row
  // from its first row on, so the regions never overlap.
  for (const StepPlan& sp : plans) {
    const size_t at = size_t(sp.row0) * 64, n = size_t(sp.V) * sp.K;
    kernels::topk_rows(logits_dev_ + size_t(sp.row0) * c.vocab, sp.V, c.vocab, sp.K, topk_scratch_, vals_dev_ + at,
                       ids_out_dev_ + at, stream_);
    check(cudaMemcpyAsync(pin_->topk_vals + at, vals_dev_ + at, n * sizeof(float), cudaMemcpyDeviceToHost, stream_),
          "topk");
    check(cudaMemcpyAsync(pin_->topk_ids + at, ids_out_dev_ + at, n * sizeof(int), cudaMemcpyDeviceToHost, stream_),
          "topk");
  }
}

void Engine::speculate_batch(std::span<SeqStep> seqs) {
  if (seqs.empty()) return;
  for (const SeqStep& q : seqs) {
    if (q.session < 0 || q.session >= sessions() || !q.params) throw std::runtime_error("speculate_batch: bad entry");
    if (!can_batch(q.session, *q.params)) throw std::runtime_error("speculate_batch: a sequence cannot join the batch");
    for (const SeqStep& o : seqs)
      if (&o != &q && o.session == q.session) throw std::runtime_error("speculate_batch: a session twice");
  }
  if (seqs.size() == 1) {  // a batch of one is the one-row step (graphs and all)
    SeqStep& q = seqs[0];
    select_session(q.session);
    try {
      q.out = speculate(q.anchor, *q.params, q.row_logits);
    } catch (const NonFiniteLogits&) {
      q.error = std::current_exception();
      reset_session();
    }
    return;
  }
  const int B = opts_.draft_block, n = static_cast<int>(seqs.size());
  if (n > kernels::kMaxSeqs) throw std::runtime_error("speculate_batch: more sequences than kMaxSeqs");
  if (kv_book_) {  // P4b: every sequence's verify pages first, all or nothing (see speculate)
    int need = 0;
    for (const SeqStep& q : seqs) need += kv_step_need(q.session, true);
    if (need > kv_book_->free_pages()) kv_retire(false);
    if (need > kv_book_->free_pages()) kv_retire(true);
    if (need > kv_book_->free_pages())
      throw KvPoolExhausted("KV page pool exhausted: the batch's step needs " + std::to_string(need) + " pages, " +
                            std::to_string(kv_book_->free_pages()) + " free");
    for (const SeqStep& q : seqs) {
      Session& s = sessions_[q.session];
      kv_ensure(s, s.pos, s.pos + kv_step_rows(s, true), stream_);
    }
  }
  const size_t vocab = model_->config().vocab;
  const int K16 = draft_->config().selector_top_k, E = B - 1;
  auto t0 = std::chrono::steady_clock::now();
  // 0. Per sequence: its seed and its lookup proposal.
  std::vector<StepPlan> plans(n);
  bool any_lookup = false;
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) {
    StepPlan& sp = plans[i];
    sp.s = &sessions_[seqs[i].session];
    sp.p = seqs[i].params;
    sp.anchor = seqs[i].anchor;
    plan_begin(sp, B);
    any_lookup = any_lookup || sp.use_lookup;
    order[i] = seqs[i].session;
  }
  // 1. The drafter over every sequence's block at once (stage 4), then each sequence's lattice walk; a batch with a
  // lookup chain (whose rows replace the drafter's) drafts per sequence, eagerly, as stage 3 did. With step graphs
  // the batch's draft, verify and commits replay as graphs keyed by the order of its sessions.
  const bool together = !any_lookup;
  const bool graph = graphs_ && together;
  BatchGraphs* bg = graph ? &batch_graphs_[order] : nullptr;
  if (graph) {  // the positions every graph of this step reads
    for (int i = 0; i < n; ++i) pin_->seq_pos[i] = plans[i].s->pos;
    check(cudaMemcpyAsync(pos_dev_, pin_->seq_pos, n * sizeof(int), cudaMemcpyHostToDevice, stream_), "positions");
    for (int i = 0; i < n; ++i) pin_->seq_rdelta[i] = plans[i].s->rope_delta;
    check(cudaMemcpyAsync(rdelta_dev_, pin_->seq_rdelta, n * sizeof(int), cudaMemcpyHostToDevice, stream_),
          "rope deltas");
  }
  if (together) {
    for (int i = 0; i < n; ++i) {
      pin_->draft_ids[i * B] = plans[i].anchor;
      for (int j = 1; j < B; ++j) pin_->draft_ids[i * B + j] = draft_->config().mask_token;
    }
    // BACKLOG 19: each sequence takes the head its session takes alone (bit i: the sparse one), and the graph is the
    // batch's for that combination.
    unsigned sparse = 0;
    for (int i = 0; i < n; ++i) sparse |= sparse_head_sync(*plans[i].s) ? 1u << i : 0u;
    if (graph) run_graph(sparse ? bg->draft_sparse[sparse] : bg->draft, "draft", [&] { draft_batch_launch(plans, B, sparse); });
    else draft_batch_launch(plans, B, sparse);
    for (int i = 0; i < n; ++i) {
      const bool si = (sparse >> i) & 1u;
      plans[i].s->sh_last_sparse = si;
      if (si) ++plans[i].s->spec.sparse_steps;
    }
    check(cudaStreamSynchronize(stream_), "draft");
    for (int i = 0; i < n; ++i)
      plan_walk(plans[i], B, pin_->cand + size_t(i) * E * K16, pin_->scores + size_t(i) * E * K16 * K16);
  } else {
    for (StepPlan& sp : plans) {
      if (sp.use_lookup) continue;
      s_ = sp.s;
      plan_draft(sp, B, false);
    }
  }
  int V = 0;
  for (StepPlan& sp : plans) {
    sp.row0 = V;
    V += sp.V;
  }
  if (V > kernels::kMaxStreamRows) throw std::runtime_error("speculate_batch: more rows than kMaxStreamRows");
  for (StepPlan& sp : plans) kv_ensure(*sp.s, sp.s->pos, sp.s->pos + sp.V, stream_);  // P4b: mapped above; a check
  auto t1 = std::chrono::steady_clock::now();

  // 2. One verify over every sequence's rows, then 3. each sequence's accept. A non-finite row fails only its own
  // sequence: the verify wrote no state of it, its commit is skipped and its session reset.
  for (const StepPlan& sp : plans) std::copy(sp.rows.begin(), sp.rows.end(), pin_->rows + sp.row0);
  join_commit();  // the previous step's DeltaNet replays, beside this step's draft until here
  if (graph) {
    std::vector<int> ks(n);
    for (int i = 0; i < n; ++i) ks[i] = plans[i].K;
    run_graph(bg->verify[ks], "verify", [&] { verify_batch_launch(plans, V); });
  } else {
    verify_batch_launch(plans, V);
  }
  check(cudaStreamSynchronize(stream_), "verify");
  for (size_t i = 0; i < plans.size(); ++i) {
    StepPlan& sp = plans[i];
    const size_t at = size_t(sp.row0) * 64;
    try {
      plan_accept(sp, pin_->topk_vals + at, pin_->topk_ids + at);
    } catch (const NonFiniteLogits&) {
      seqs[i].error = std::current_exception();
      sp.failed = true;
    }
  }
  auto t2 = std::chrono::steady_clock::now();

  // 4. Per sequence: its rows' logits when asked, then its commit from its own rows.
  for (size_t i = 0; i < plans.size(); ++i) {
    const StepPlan& sp = plans[i];
    if (sp.failed || !seqs[i].row_logits) continue;
    const int k = sp.accepted + 1;
    seqs[i].row_logits->resize(size_t(k) * vocab);
    check(cudaMemcpyAsync(seqs[i].row_logits->data(), logits_dev_ + size_t(sp.row0) * vocab,
                          size_t(k) * vocab * sizeof(float), cudaMemcpyDeviceToHost, stream_),
          "row logits");
  }
  for (int i = 0; i < n; ++i) {
    StepPlan& sp = plans[i];
    if (sp.failed) continue;
    const int k = sp.accepted + 1;
    s_ = sp.s;
    if (commit_overlap_) commit_state_async(sp.row0, k, graph);
    if (graph) {
      // Slot i's commit reads its position from pos_dev_[i] and its rows from row i * B; slot 0's is the one-row
      // step's own commit graph (the same rows, the same position slot).
      cudaGraphExec_t* g = nullptr;
      if (i == 0) {
        if (s_->commit_graphs.size() < size_t(B) + 1) s_->commit_graphs.resize(size_t(B) + 1, nullptr);
        g = &s_->commit_graphs[k];
      } else {
        g = &s_->batch_commit_graphs[{i, k}];
      }
      dpos_slot_ = i;
      try {
        run_graph(*g, "commit", [&] { commit_launch(sp.row0, k); });
      } catch (...) {
        dpos_slot_ = 0;
        throw;
      }
      dpos_slot_ = 0;
    } else {
      commit_launch(sp.row0, k);
    }
    sp.s->history.insert(sp.s->history.end(), sp.rows.begin(), sp.rows.begin() + k);
    sp.s->pos += k;
  }
  check(cudaStreamSynchronize(stream_), "commit");
  const double draft_s = std::chrono::duration<double>(t1 - t0).count();
  const double verify_s = std::chrono::duration<double>(t2 - t1).count(), commit_s = seconds_since(t2);
  for (size_t i = 0; i < plans.size(); ++i) {
    StepPlan& sp = plans[i];
    if (sp.failed) {
      s_ = sp.s;
      reset_session();
      continue;
    }
    // The batch's wall times, charged to each sequence in it (its step period).
    sp.s->spec.draft_seconds += draft_s;
    sp.s->spec.verify_seconds += verify_s;
    sp.s->spec.commit_seconds += commit_s;
    sp.s->spec.batched_steps += 1;
    plan_count(sp);
    seqs[i].out.assign(sp.rows.begin() + 1, sp.rows.begin() + sp.accepted + 1);
    seqs[i].out.push_back(sp.next);
  }
}

void Engine::score_shadow(Session& s, bool all) {
  // A record is scored once the tokens after its anchor are known (or, with `all`, as far as they are):
  // the proposal would have had accepted its leading tokens that equal what was generated.
  const int L = static_cast<int>(s.history.size());
  size_t keep = 0;
  for (size_t i = 0; i < s.shadow.size(); ++i) {
    ShadowRecord& r = s.shadow[i];
    const int need = r.pos + 1 + static_cast<int>(r.tokens.size());
    if (need > L && !all) {
      s.shadow[keep++] = std::move(r);
      continue;
    }
    int k = 0;
    while (k < static_cast<int>(r.tokens.size()) && r.pos + 1 + k < L && s.history[r.pos + 1 + k] == r.tokens[k]) ++k;
    s.spec.shadow_steps[r.bucket] += 1;
    s.spec.shadow_lookup[r.bucket] += k;
    s.spec.shadow_dflash[r.bucket] += r.dflash_accepted;
    s.spec.shadow_best += std::max(k, r.dflash_accepted);
  }
  s.shadow.resize(keep);
}

}  // namespace ling
