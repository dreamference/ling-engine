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
#include <stdexcept>

#include <nvtx3/nvToolsExt.h>

#include "core/engine.hpp"
#include "core/kernels.cuh"
#include "core/accept.hpp"
#include "core/lookup.hpp"

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
  std::vector<int> cand;                // the drafter's candidates [E][K16] (pin_ holds one sequence's at a time)
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
    for (auto& [k, g] : s.verify_graphs) cudaGraphExecDestroy(g);
    s.verify_graphs.clear();
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
  if (rows <= kernels::kMaxStreamRows)
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

void Engine::draft_launch(int B) {
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
    linear_bf16_drafter(L.attn_kproj, dh_, B, dcoef_);
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
    linear_bf16_drafter(L.mlp_kproj, dh_, B, dcoef_);
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
  linear_fp4(model_->lm_head(), dh_, E, dlogits_);
  kernels::topk_rows(dlogits_, E, c.vocab, K16, topk_scratch_, dunary_, dcand_, stream_);
  linear_bf16_drafter(draft_->selector_projection(), dh_, E, dhp_);
  kernels::selector_lattice(dhp_, dcand_, dunary_, draft_->predecessor_codebook(), draft_->successor_codebook(),
                            pin_->draft_ids[0], E, dscores_, stream_, ids_dev_);
  check(cudaMemcpyAsync(pin_->cand, dcand_, size_t(E) * K16 * sizeof(int), cudaMemcpyDeviceToHost, stream_),
        "draft candidates");
  check(cudaMemcpyAsync(pin_->scores, dscores_, size_t(E) * K16 * K16 * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_),
        "draft scores");
}

void Engine::draft_batch_launch(std::span<StepPlan> plans, int B) {
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
    linear_bf16_drafter(L.attn_kproj, dh_, R, dcoef_);
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
    linear_bf16_drafter(L.mlp_kproj, dh_, R, dcoef_);
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
  linear_fp4(model_->lm_head(), dh_, n * E, dlogits_);
  kernels::topk_rows(dlogits_, n * E, c.vocab, K16, topk_scratch_, dunary_, dcand_, stream_);
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
  if (graph) run_graph(s_->draft_graph, "draft", [&] { draft_launch(B); });
  else draft_launch(B);
  check(cudaStreamSynchronize(stream_), "draft");
}

void Engine::verify_launch(int V, int K) {
  const ModelConfig& c = model_->config();
  forward(pin_->rows, V, Pass::Verify);
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
  plan_walk(sp, B, pin_->cand, pin_->scores);
}

void Engine::plan_walk(StepPlan& sp, int B, const int* cand, const float* scores_host) {
  const SamplingParams& p = *sp.p;
  const int K16 = draft_->config().selector_top_k, E = sp.E;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  sp.cand.assign(cand, cand + size_t(E) * K16);
  sp.rows.resize(B);
  sp.q.assign(E, std::vector<double>(K16, 0.0));
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
    prev = pick;
  }
}

void Engine::plan_accept(StepPlan& sp, const float* vals, const int* ids) {
  // On the host, after the step (graph or not): no verified row may be NaN or inf (core/sampling.hpp).
  const int V = sp.V, K = sp.K, E = sp.E;
  require_finite_topk({vals, size_t(V) * K}, {ids, size_t(V) * K}, model_->config().vocab);
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
  s.spec.drafted += sp.E;
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
  }
  if (!sp.use_lookup) plan_draft(sp, B, graph);
  const int V = sp.V, K = sp.K;  // rows verified, top K per row
  auto t1 = std::chrono::steady_clock::now();
  s_->spec.draft_seconds += std::chrono::duration<double>(t1 - t0).count();

  // 2. Verify, with each row's top K (the argmax when greedy) on the GPU.
  std::copy(sp.rows.begin(), sp.rows.end(), pin_->rows);
  join_commit();  // the previous step's DeltaNet replay, beside this step's draft until here
  if (graph) {
    cudaGraphExec_t& g = s_->verify_graphs[K];
    run_graph(g, "verify", [&] { verify_launch(V, K); });
  } else {
    verify_launch(V, K);
  }
  check(cudaStreamSynchronize(stream_), "verify");
  plan_accept(sp, pin_->topk_vals, pin_->topk_ids);  // 3. accept
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
  }
  if (together) {
    for (int i = 0; i < n; ++i) {
      pin_->draft_ids[i * B] = plans[i].anchor;
      for (int j = 1; j < B; ++j) pin_->draft_ids[i * B + j] = draft_->config().mask_token;
    }
    if (graph) run_graph(bg->draft, "draft", [&] { draft_batch_launch(plans, B); });
    else draft_batch_launch(plans, B);
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
