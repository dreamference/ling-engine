// Speculative decoding with the DFlash2 drafter (SPEC.md §10, lever 2), as SGLang's DFLASH path runs it
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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "core/engine.hpp"
#include "core/kernels.cuh"
#include "core/accept.hpp"
#include "core/lookup.hpp"

namespace ling {
namespace {

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
  for (int l = 0; l < d.layers; ++l) {
    dkc_.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * dkv));
    dvc_.push_back(alloc<__nv_bfloat16>(size_t(opts_.max_context) * dkv));
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
  const int C = c.lin_conv_channels();
  for (size_t i = 0; i < gdn_state_.size(); ++i) {
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

void Engine::commit(int n) {
  const ModelConfig& c = model_->config();
  for (int li = 0; li < c.layers; ++li) {
    const LayerWeights& L = model_->layers()[li];
    if (L.full) continue;  // the verify already wrote the KV; rows past pos_ + n are overwritten later
    const int slot = layer_slot_[li];
    // The conv window advances over the accepted rows' raw inputs, the recurrence over their conv outputs.
    kernels::gdn_conv(v_pre_[slot], conv_state_[slot], L.conv, n, c.lin_conv_channels(), stream_);
    kernels::gdn_recurrent(v_post_[slot], v_g_[slot], v_beta_[slot], gdn_state_[slot], gdn_state_[slot], core_, n,
                           c.lin_k_heads, c.lin_v_heads, stream_);
  }
}

void Engine::draft_materialize(int rows, int pos0, int cap_row0) {
  if (rows <= 0) return;
  const DraftConfig& d = draft_->config();
  const int dkv = d.kv_heads * d.head_dim;
  // fc over the concatenated target features (already BF16), then the hidden norm. The context goes
  // straight to each layer's K and V: no input norm, no convolution (SGLang's prepare_context_hidden_for_kv).
  if (rows <= kernels::kMaxStreamRows)
    kernels::bf16_rows_bf16in(cap_ + size_t(cap_row0) * cap_stride_, rows, draft_->fc().w, dctx_, d.hidden, cap_stride_,
                              stream_);
  else
    kernels::gemm_bf16_cublas(cublas_, cap_ + size_t(cap_row0) * cap_stride_, rows, draft_->fc().w, dctx_, d.hidden,
                              cap_stride_);
  kernels::rmsnorm(dctx_, draft_->hidden_norm(), dctx_, rows, d.hidden, d.eps, false, stream_);
  for (int l = 0; l < d.layers; ++l) {
    const DraftLayer& L = draft_->layers()[l];
    linear_fp4_multi({{&L.k, dk_}, {&L.v, dv_}}, dctx_, rows);
    kernels::draft_qk_rope(dk_, rows, d.kv_heads, dkv, L.k_norm, pos0, d.rope_theta, d.eps, stream_);
    kernels::draft_store_kv(dk_, dv_, rows, dkv, pos0, dkc_[l], dvc_[l], stream_);
  }
}

void Engine::draft_propose(int anchor, int B) {
  const ModelConfig& c = model_->config();
  const DraftConfig& d = draft_->config();
  const int H = d.hidden, E = B - 1, dq = d.heads * d.head_dim, dkv = d.kv_heads * d.head_dim;
  const int K16 = d.selector_top_k;
  std::vector<int> ids(B, d.mask_token);
  ids[0] = anchor;
  check(cudaMemcpyAsync(ids_dev_, ids.data(), B * sizeof(int), cudaMemcpyHostToDevice, stream_), "draft ids");
  kernels::embed(model_->embed(), ids_dev_, B, H, dres_, stream_);
  for (int l = 0; l < d.layers; ++l) {
    const DraftLayer& L = draft_->layers()[l];
    // Pre-norm with the residual added in (SGLang's fused add + RMSNorm); the drafter's norms are plain.
    if (l > 0) kernels::add_inplace(dres_, dout_, B * H, stream_);
    kernels::rmsnorm(dres_, L.input_norm, dh_, B, H, d.eps, false, stream_);
    // Attention wrapped in the grouped conv: input side 0, output side 1, one kernel projection.
    linear_bf16_cublas(L.attn_kproj, dh_, B, dcoef_);
    kernels::grouped_conv(dh_, dcoef_, L.attn_base, 0, dh2_, B, H, d.conv_groups, B, stream_);
    linear_fp4_multi({{&L.q, dq_}, {&L.k, dk_}, {&L.v, dv_}}, dh2_, B);
    kernels::draft_qk_rope(dq_, B, d.heads, dq, L.q_norm, pos_, d.rope_theta, d.eps, stream_);
    kernels::draft_qk_rope(dk_, B, d.kv_heads, dkv, L.k_norm, pos_, d.rope_theta, d.eps, stream_);
    kernels::draft_attention(dq_, dkc_[l], dvc_[l], dk_, dv_, pos_, B, d.sliding_window - 1, d.heads, d.kv_heads,
                             dattn_, stream_);
    linear_fp4(L.o, dattn_, B, dh2_);
    kernels::grouped_conv(dh2_, dcoef_, L.attn_base, 1, dout_, B, H, d.conv_groups, B, stream_);
    kernels::add_inplace(dres_, dout_, B * H, stream_);
    kernels::rmsnorm(dres_, L.post_norm, dh_, B, H, d.eps, false, stream_);
    // The MLP, wrapped the same way.
    linear_bf16_cublas(L.mlp_kproj, dh_, B, dcoef_);
    kernels::grouped_conv(dh_, dcoef_, L.mlp_base, 0, dh2_, B, H, d.conv_groups, B, stream_);
    linear_fp4_multi({{&L.gate, dg_}, {&L.up, du_}}, dh2_, B);
    kernels::silu_mul(dg_, du_, dg_, B * d.intermediate, stream_);
    linear_fp4(L.down, dg_, B, dh2_);
    kernels::grouped_conv(dh2_, dcoef_, L.mlp_base, 1, dout_, B, H, d.conv_groups, B, stream_);
  }
  kernels::add_inplace(dres_, dout_, B * H, stream_);
  // Drafted positions 1 .. B - 1: the final norm, the target's LM head (no target final norm), the top 16,
  // the selector's projection and lattice.
  kernels::rmsnorm(dres_ + H, draft_->norm(), dh_, E, H, d.eps, false, stream_);
  linear_fp4(model_->lm_head(), dh_, E, dlogits_);
  kernels::topk_rows(dlogits_, E, c.vocab, K16, topk_scratch_, dunary_, dcand_, stream_);
  linear_bf16_cublas(draft_->selector_projection(), dh_, E, dhp_);
  kernels::selector_lattice(dhp_, dcand_, dunary_, draft_->predecessor_codebook(), draft_->successor_codebook(),
                            anchor, E, dscores_, stream_);
  cand_host_.resize(size_t(E) * K16);
  scores_host_.resize(size_t(E) * K16 * K16);
  check(cudaMemcpyAsync(cand_host_.data(), dcand_, cand_host_.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_),
        "draft candidates");
  check(cudaMemcpyAsync(scores_host_.data(), dscores_, scores_host_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_),
        "draft scores");
  check(cudaStreamSynchronize(stream_), "draft");
}

std::vector<int> Engine::speculate(int anchor, const SamplingParams& p) {
  if (!can_speculate(p)) throw std::runtime_error("speculate: unsupported sampling parameters or no drafter");
  const ModelConfig& c = model_->config();
  const int room = opts_.max_context - pos_;
  if (room < 1) throw std::runtime_error("context is longer than max_context");
  const int B = std::min(opts_.draft_block, room);
  const bool greedy = p.temperature <= 0.f;
  if (B < 2) {  // no room to draft: one plain step
    step(anchor);
    return {sample(logits_, p, history_)};
  }
  const int K16 = draft_->config().selector_top_k;
  int E = B - 1;
  if (p.seed != 0) rng_.seed(p.seed + history_.size());
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  auto t0 = std::chrono::steady_clock::now();

  // 0. The context lookup: verified instead of the drafter's chain on a long verbatim match (auto), or
  // only recorded and scored later (shadow).
  LookupProposal lp;
  if (opts_.lookup_mode > 0) {
    std::vector<int> seq(history_);
    seq.push_back(anchor);
    lp = lookup_propose(seq, E);
  }
  const bool use_lookup = opts_.lookup_mode == 2 && !lp.tokens.empty() && lp.match >= opts_.lookup_min_match;
  std::vector<int> rows(1, anchor);
  std::vector<std::vector<double>> q;
  if (use_lookup) {
    rows.insert(rows.end(), lp.tokens.begin(), lp.tokens.end());
    E = static_cast<int>(lp.tokens.size());
  }

  // 1. Draft, and walk the selector's lattice (SGLang's sample_path).
  if (!use_lookup) {
  draft_propose(anchor, B);
  rows.resize(B);
  q.assign(E, std::vector<double>(K16, 0.0));
  int prev = 0;
  for (int e = 0; e < E; ++e) {
    const float* s = scores_host_.data() + (size_t(e) * K16 + (e == 0 ? 0 : prev)) * K16;
    int pick = 0;
    if (greedy) {
      for (int k = 1; k < K16; ++k)
        if (s[k] > s[pick]) pick = k;  // ties to the lower index
      q[e][pick] = 1.0;
    } else {
      const double T = std::max(double(p.temperature), 1e-5);
      double mx = s[0];
      for (int k = 1; k < K16; ++k) mx = std::max(mx, double(s[k]));
      double sum = 0;
      for (int k = 0; k < K16; ++k) sum += q[e][k] = std::exp((s[k] - mx) / T);
      for (double& v : q[e]) v /= sum;
      const double u = uni(rng_);
      double cum = 0;
      pick = K16 - 1;
      for (int k = 0; k < K16; ++k) {
        cum += q[e][k];
        if (u < cum) {
          pick = k;
          break;
        }
      }
    }
    rows[e + 1] = cand_host_[size_t(e) * K16 + pick];
    prev = pick;
  }
  }
  const int V = E + 1;  // rows verified
  auto t1 = std::chrono::steady_clock::now();
  spec_stats_.draft_seconds += std::chrono::duration<double>(t1 - t0).count();

  // 2. Verify.
  forward(rows.data(), V, Pass::Verify);

  // 3. Accept.
  int accepted = 0, next = 0;
  if (greedy) {
    std::vector<int> am(V);
    kernels::topk_rows(logits_dev_, V, c.vocab, 1, topk_scratch_, vals_dev_, ids_out_dev_, stream_);
    check(cudaMemcpyAsync(am.data(), ids_out_dev_, V * sizeof(int), cudaMemcpyDeviceToHost, stream_), "argmax");
    check(cudaStreamSynchronize(stream_), "verify");
    while (accepted < E && rows[accepted + 1] == am[accepted]) ++accepted;
    next = am[accepted];
  } else {
    const int K = p.top_k;
    std::vector<float> vals(size_t(V) * K);
    std::vector<int> ids(size_t(V) * K);
    kernels::topk_rows(logits_dev_, V, c.vocab, K, topk_scratch_, vals_dev_, ids_out_dev_, stream_);
    check(cudaMemcpyAsync(vals.data(), vals_dev_, vals.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "topk");
    check(cudaMemcpyAsync(ids.data(), ids_out_dev_, ids.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_), "topk");
    check(cudaStreamSynchronize(stream_), "verify");
    auto q_of = [&](int e, int token) {  // the draft's probability of `token` at drafted position e
      if (use_lookup) return token == rows[e + 1] ? 1.0 : 0.0;  // a deterministic proposal
      double qt = 0;
      const int* cand = cand_host_.data() + size_t(e) * K16;
      for (int k = 0; k < K16; ++k)
        if (cand[k] == token) qt += q[e][k];
      return qt;
    };
    std::vector<Dist> P(V);
    for (int i = 0; i < V; ++i) P[i] = target_dist(vals.data() + size_t(i) * K, ids.data() + size_t(i) * K, K, p);
    const auto [acc, tok] = accept_sampled(P, std::vector<int>(rows.begin() + 1, rows.end()), q_of, rng_);
    accepted = acc;
    next = tok;
  }
  auto t2 = std::chrono::steady_clock::now();
  spec_stats_.verify_seconds += std::chrono::duration<double>(t2 - t1).count();

  // 4. Commit the anchor and the accepted drafts.
  const int n = accepted + 1;
  commit(n);
  draft_materialize(n, pos_, 0);
  history_.insert(history_.end(), rows.begin(), rows.begin() + n);
  pos_ += n;
  check(cudaStreamSynchronize(stream_), "commit");
  spec_stats_.commit_seconds += seconds_since(t2);
  spec_stats_.steps += 1;
  spec_stats_.drafted += E;
  spec_stats_.accepted += accepted;
  spec_stats_.accept_hist[accepted] += 1;
  if (use_lookup) {
    spec_stats_.lookup_steps += 1;
    spec_stats_.lookup_accepted += accepted;
  } else if (opts_.lookup_mode == 1 && !lp.tokens.empty()) {
    const int bucket = lp.match >= 32 ? 4 : lp.match >= 16 ? 3 : lp.match >= 8 ? 2 : lp.match >= 4 ? 1 : 0;
    shadow_.push_back({pos_ - n, std::move(lp.tokens), lp.match, accepted, bucket});
  }
  if (!use_lookup && opts_.lookup_mode == 1) {
    spec_stats_.shadow_total_steps += 1;
    spec_stats_.shadow_dflash_all += accepted;
    if (shadow_.empty() || shadow_.back().pos != pos_ - n) spec_stats_.shadow_best += accepted;  // no proposal
  }
  score_shadow(false);

  std::vector<int> out(rows.begin() + 1, rows.begin() + n);
  out.push_back(next);
  return out;
}

void Engine::score_shadow(bool all) {
  // A record is scored once the tokens after its anchor are known (or, with `all`, as far as they are):
  // the proposal would have had accepted its leading tokens that equal what was generated.
  const int L = static_cast<int>(history_.size());
  size_t keep = 0;
  for (size_t i = 0; i < shadow_.size(); ++i) {
    ShadowRecord& r = shadow_[i];
    const int need = r.pos + 1 + static_cast<int>(r.tokens.size());
    if (need > L && !all) {
      shadow_[keep++] = std::move(r);
      continue;
    }
    int k = 0;
    while (k < static_cast<int>(r.tokens.size()) && r.pos + 1 + k < L && history_[r.pos + 1 + k] == r.tokens[k]) ++k;
    spec_stats_.shadow_steps[r.bucket] += 1;
    spec_stats_.shadow_lookup[r.bucket] += k;
    spec_stats_.shadow_dflash[r.bucket] += r.dflash_accepted;
    spec_stats_.shadow_best += std::max(k, r.dflash_accepted);
  }
  shadow_.resize(keep);
}

}  // namespace ling
