// One sequence at a time, out of a pool of resident sessions (M4 stages 1 and 2: the server interleaves two
// sessions' prefill chunks and steps, switching the active session between them): the forward pass (chunked prefill, then decode or speculative verify), its
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
#include <functional>
#include <initializer_list>
#include <utility>
#include <chrono>
#include <exception>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/drafter.hpp"
#include "core/kernels.cuh"
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
  // M3: run each part of a speculative step (draft, verify, commit) as a captured CUDA graph instead of
  // ~1,400 separate launches (speculate.cpp). Off by default until measured; LING_STEP_GRAPHS=0/1 overrides.
  bool step_graphs = false;
  // M3: programmatic dependent launch for the rows path's kernels (launch.cuh): each kernel may start, and
  // issue its first weight loads, while the one before it finishes. Off by default; LING_PDL=0/1 overrides.
  // Process-wide (the kernels' launch setting), like the KV layout.
  bool pdl = false;
  // M3: the rows path's attention loads its KV tiles through the bulk-copy engine (kernels.cuh,
  // set_attention_bulk); bit for bit the same output. Off by default; LING_ATTN_BULK=0/1 overrides.
  bool attention_bulk = false;
  // M3: the rows path's attention prefetches its KV tiles into L2 this many tiles ahead (kernels.cuh,
  // set_attention_prefetch); bit for bit the same output. Measured at 24K: -12% per layer at 2. 0 turns it off;
  // LING_ATTN_PREFETCH overrides.
  int attention_prefetch = 2;
  // B34: the prefill's DeltaNet intermediates (the norm's copy for the a/b projections, the in_qkv and in_z
  // projections, the conv's output, the recurrence's output) in BF16 where production keeps BF16, instead of
  // FP32 (reports/M2-prefill.md §8 item 3). Not exact: a change of the prefill's numerics, so off by default
  // under MILESTONES §14's exactness policy until its KL and agreement checks pass (bench/prefill_bf16_check.sh).
  // The decode and verify rows path is untouched either way. LING_PREFILL_BF16=0/1 overrides.
  bool prefill_bf16 = false;
  // M4 §1: resident sessions, each a fixed slot with its own KV cache, DeltaNet and conv state, end-of-prompt
  // state, drafter context ring, history and RNG, all allocated here. Requests run one at a time in the
  // active session (select it with activate_session); the prefix checkpoints are shared by all of them.
  int sessions = 1;
  // M4's step savings (reports/M3.md §7 item 4): a speculative step's commit replays the DeltaNet and conv state
  // over the accepted rows on a second stream, beside the next step's draft (which reads none of that state), and
  // the next verify waits for it. The same kernels on the same inputs, so the same bits. LING_COMMIT_OVERLAP=0/1
  // overrides.
  bool commit_overlap = true;
  // BACKLOG §19.19: the drafter's context-aware sparse head. Empty: the full head (the default until its acceptance
  // gate passes). Otherwise a directory with model.safetensors holding `us` (the head's rank-256 left factor times
  // its singular values, BF16 [vocab][256]), `vt` (the right factor, BF16 [256][hidden]) and `fixed` (I32, the rows
  // always scored), written by bench/sparse_head/build.py. sparse_head_dynamic: the rows chosen per step besides.
  std::string sparse_head_dir;
  int sparse_head_dynamic = 16384;
};

struct EngineStats {
  double prefill_seconds = 0;  // wall time from begin_prefill to the last chunk (other sessions' work included)
  double compute_seconds = 0;  // the prefill's own chunks only
  int prefill_tokens = 0;
  int reused_tokens = 0;
  int chunks = 0;
};

// Counters of speculative steps since the last reset.
struct SpecStats {
  long steps = 0;
  long drafted = 0;   // drafted tokens offered to the verify
  long accepted = 0;  // drafted tokens accepted
  std::vector<long> accept_hist = std::vector<long>(33, 0);
  double draft_seconds = 0, verify_seconds = 0, commit_seconds = 0;
  long lookup_steps = 0, lookup_accepted = 0;  // steps that verified the lookup chain, and its accepted tokens
  // M4 §2: steps this session took inside a batch (speculate_batch with two or more sequences). A batched step's
  // draft, verify and commit times are the batch's wall times, charged to each sequence in it, so a session's
  // seconds per step read as its own step period.
  long batched_steps = 0;
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
  // The state comes from the active session's own kept states, the shared checkpoint store, or (when they
  // reach further) another session's, whose KV for the shared prefix is copied in first (M4 §1).
  // `from_scratch`: reuse nothing (the reference --prefix-check compares a resumed prefill with).
  const std::vector<float>& prefill(const std::vector<int>& prompt, EngineStats* stats = nullptr,
                                    bool from_scratch = false);
  // The same prefill one chunk at a time (M4 §3), so a server can run another session's step between chunks.
  // begin_prefill picks the state to resume from and readies the active session (prefill()'s first half);
  // each prefill_chunk call makes the cursor's session active and runs one pass of at most `max_tokens`
  // tokens (0: the full prefill_chunk; rounded down to a multiple of kResumeAlign, so every chunk ends
  // where a resumed prefill could start and the state is the same however the prompt was split), saving
  // the snap and checkpoints at the same positions prefill() does. It returns true after the last chunk,
  // when logits() holds the logits after the prompt's last token (until the next pass of any session).
  struct PrefillCursor {
    int session = -1;
    std::vector<int> prompt;
    int next = 0;          // the first position not yet processed
    int snap_at = 0;       // where the end-of-prompt state is kept
    int window = 0;        // the drafter's context starts here
    std::vector<int> bounds;  // chunk ends where checkpoints are saved
    uint64_t ckpt_epoch = 0;  // the checkpoint store's clock at begin_prefill: entries used since are this prompt's
    std::chrono::steady_clock::time_point t0;
    EngineStats stats;
    bool done() const { return next >= static_cast<int>(prompt.size()); }
  };
  PrefillCursor begin_prefill(const std::vector<int>& prompt, bool from_scratch = false);
  bool prefill_chunk(PrefillCursor& cur, int max_tokens = 0);
  const std::vector<float>& logits() const { return logits_; }
  int prefill_chunk_size() const { return opts_.prefill_chunk; }
  // Appends one token and returns the logits after it.
  const std::vector<float>& step(int token);
  // Picks the next token from logits (sample_logits). `history` is history(): a seed is offset by its
  // length, and presence and repetition penalties count only its tokens after the last prefill's prompt,
  // i.e. the request's output, as production does (never the prompt's own tokens). Throws
  // NonFiniteLogits if any logit is NaN or inf.
  int sample(const std::vector<float>& logits, const SamplingParams& p, const std::vector<int>& history);

  // Speculation. `anchor` is the next token, chosen but not yet processed (what step() would take).
  // One step drafts block - 1 tokens, verifies anchor + drafts in one pass, and keeps the anchor and the
  // accepted drafts. Returns the accepted drafts followed by the next token (chosen, not processed: the
  // next call's anchor). Greedy output equals plain decoding's token for token; sampled output follows
  // the same distribution (exact rejection sampling).
  bool has_drafter() const { return draft_ != nullptr; }
  bool can_speculate(const SamplingParams& p) const;
  // `row_logits`, when given, receives the raw logits each returned token was chosen from: one row of
  // config().vocab floats per returned token, in the same order (the verify's rows, which the
  // row-invariant rows path makes bit for bit what plain decoding gives). Null by default: nothing is
  // copied off the device beyond each row's top K.
  std::vector<int> speculate(int anchor, const SamplingParams& p, std::vector<float>* row_logits = nullptr);
  // A batch of speculative steps (M4 §2, stage 3): one step for each of several sessions, their verify rows
  // concatenated in one forward pass ([a1, d1 ..] of the first, then [a2, d2 ..] of the second); the drafter, the
  // state-bearing kernels (attention and its KV writes, the DeltaNet conv and recurrence), the lattice walk, the
  // accept rule and the commit run per sequence. The rows path is row-invariant, so each sequence's tokens,
  // state, steps and accepted drafts are bit for bit what speculate() gives it alone. Stage 4: the drafter runs
  // over every sequence's block at once and the state kernels in one launch each (kernels::SeqBatch), and with
  // step graphs on the draft, the verify and each sequence's commit replay as CUDA graphs. Each entry must be
  // can_batch(); their rows must fit kMaxStreamRows.
  struct SeqStep {
    int session = -1;
    int anchor = 0;                          // as speculate()'s
    const SamplingParams* params = nullptr;  // as speculate()'s
    std::vector<float>* row_logits = nullptr;
    std::vector<int> out;      // as speculate() returns: the accepted drafts, then the next token
    std::exception_ptr error;  // a non-finite row (NonFiniteLogits): nothing committed, the session reset
  };
  // Whether session i's next speculative step can join a batch: speculation possible for p and room for a full
  // draft block (a shorter block or a plain step runs alone, M4 §2).
  bool can_batch(int session, const SamplingParams& p) const;
  void speculate_batch(std::span<SeqStep> seqs);
  // The active session's speculation counters (each session counts its own steps, M4 §3), its lookup
  // shadow records scored as far as known.
  const SpecStats& spec_stats() {
    score_shadow(*s_, true);
    return s_->spec;
  }
  // Session i's counters, scored the same way.
  const SpecStats& session_spec_stats(int i);
  // Every session's counters added up (what /spec_stats reports), without scoring records still in flight.
  SpecStats spec_stats_total() const;
  void reset_spec_stats() {
    for (Session& s : sessions_) {
      s.spec = SpecStats{};
      s.shadow.clear();
    }
  }
  void set_lookup(int mode, int min_match) {
    opts_.lookup_mode = mode;
    opts_.lookup_min_match = min_match;
  }
  int draft_block() const { return opts_.draft_block; }
  void set_draft_block(int b);
  // Whether speculative steps run as CUDA graphs (false when asked for but unavailable: see step_graphs).
  bool step_graphs() const { return graphs_; }
  // Turns step graphs on or off between steps (ling-run --graph-check compares the two); `on` has no effect
  // where graphs are unavailable.
  void set_step_graphs(bool on);

  const std::vector<int>& history() const { return s_->history; }
  // The active session to the empty state, and the checkpoint store emptied (a cold engine).
  void reset();
  // The active session only: its history, kept states and drafter context forgotten; the store stays.
  void reset_session();

  // ---- The session pool (M4 §1). The binding policy lives here so ling-serve and ling-run share it. ----
  int sessions() const { return static_cast<int>(sessions_.size()); }
  int active_session() const { return static_cast<int>(s_ - sessions_.data()); }
  // The resident session a prompt continues, or -1 (a new session). A session is bound only when the
  // prompt extends it: the common prefix covers every complete message of the session's last prompt, so
  // what differs is at most its last turn's generation prompt and output (a next turn re-renders them). Among those the longest common prefix wins, the key's
  // session on a tie. When none is extended, the key's own session (the same thread, diverged: it
  // re-prefills from the divergence). The key is a hint: the token prefix decides what is reused.
  int match_session(const std::vector<int>& prompt, const std::string& key) const;
  // A slot for a new session: one never used, else the least recently used one that is neither in `pinned`
  // nor protected; -1 when there is none (the request waits). Protected: a session bound with a key (an
  // agent's thread) whose last request ran less than protect_seconds ago (M4 §7 question 6).
  int free_session(double protect_seconds, const std::vector<bool>& pinned) const;
  // Makes session i active. `fresh`: a new session takes the slot (an eviction when another held it).
  // Nothing is copied out or cleared: the slot's tokens stay usable as KV by a prompt that shares them.
  void activate_session(int i, const std::string& key, bool fresh);
  // Makes session i active for the next pass, with no other effect (the scheduler switches between the
  // sessions of its running requests, M4 §3).
  void select_session(int i);
  // Stamps the active session's (or session i's) last use (the LRU order and the protection clock).
  void touch_session();
  void touch_session(int i);
  // Every session emptied and the checkpoint store cleared (the exactness checks start from a cold engine).
  void reset_all();
  int sessions_resident() const;
  long session_evictions() const { return evictions_; }
  // FNV-1a over every DeltaNet and conv state, bit for bit (the exactness tests compare runs with it).
  uint64_t state_hash();

 private:
  struct Session;
  enum class Pass { Prefill, Decode, Verify };
  // Appends M tokens at position pos_ (a verify writes their KV but leaves pos_ and the recurrent state
  // to commit()). Prefill and Decode leave the logits after the last token in logits_; Verify leaves the
  // logits of every row in logits_dev_.
  // keep_at (prefill only, 0 < keep_at < M): also keep the DeltaNet and conv state after the first keep_at
  // tokens in snap_ (the end-of-prompt state at a multiple of 32, without a separate pass for the tail).
  void forward(const int* ids, int M, Pass pass, int keep_at = -1);
  // The same over the rows of several sessions (M4 §2): segment i is rows [row0, row0 + rows) of ids, in session
  // s at its own position. The row-wise kernels run once over all M rows; the state-bearing ones once per
  // segment. More than one segment only for Verify; forward() is one segment, the active session's.
  struct Seg {
    Session* s;
    int row0, rows;
  };
  void forward_rows(const int* ids, int M, Pass pass, int keep_at, std::span<const Seg> segs);
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
  bool prefill_bf16_ = false;             // opts_.prefill_bf16 or LING_PREFILL_BF16: BF16 DeltaNet intermediates
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
  // Keeps the state after history[0, pos) in the store. Full: the least recently used entry gives way, never one
  // on the current prompt's prefix (used at or after `epoch`), so a prompt longer than the store keeps its earliest
  // positions (the ones other sessions share); when every entry is the prompt's own, nothing is saved.
  void save_checkpoint(int pos, uint64_t epoch);
  void truncate_history(int n);  // forget the tokens from position n on, and every state past them

  // Speculation (speculate.cpp).
  // Advance the active session's recurrent state over a verify's rows row0 .. row0 + n - 1 (its anchor and
  // accepted drafts; row0 is the sequence's first row in a batch).
  void commit(int row0, int n, cudaStream_t st, float* out);
  // The commit beside the next draft (EngineOptions::commit_overlap): the DeltaNet replay runs on commit_stream_
  // and records commit_done_; join_commit() makes stream_ wait for it. Everything that reads or writes the
  // recurrent state, the verify's saved inputs or commit_core_ on stream_ (every forward pass, the state copies,
  // resets and hashes) joins first; the draft and the drafter's context KV (draft_materialize) do not need to.
  bool commit_overlap_ = false;
  bool commit_pending_ = false;
  cudaStream_t commit_stream_ = nullptr;
  cudaEvent_t commit_done_ = nullptr;
  float* commit_core_ = nullptr;  // the replay's recurrence output, which nothing reads (core_ belongs to stream_)
  void join_commit();
  // The commit's two halves: the DeltaNet replay (on commit_stream_ when overlapped) and the drafter's context KV.
  void commit_state_launch(int row0, int n, cudaStream_t st);
  void commit_state_async(int row0, int n, bool graph);  // the replay on commit_stream_, then commit_done_
  // Target features -> the drafter's context KV at positions pos0 .. pos0 + rows - 1.
  void draft_materialize(int rows, kernels::DevPos pos0, int cap_row0);
  void draft_propose(int anchor, int B, bool graph);  // the drafter's block, top-16 candidates and lattice
  void alloc_drafter();
  void alloc_sparse_head(const std::string& model_dir);
  // The drafter's head over its E rows of dh_: dunary_ and dcand_ (top K16 values and token ids per row), from the
  // full head or the sparse one (BACKLOG §19.19).
  void draft_head(int E);

  // The three parts of a speculative step as launch sequences with no host synchronization inside, so
  // each can be captured into a CUDA graph. Their inputs and outputs on the host side are pin_'s buffers.
  void draft_launch(int B);         // pin_->draft_ids -> pin_->cand, pin_->scores
  void verify_launch(int V, int K);  // pin_->rows -> each row's top K in pin_->topk_vals / topk_ids
  void commit_launch(int row0, int n);  // the anchor and n - 1 accepted drafts into the recurrent state and drafter KV
  // One sequence's part of a speculative step on the host (speculate.cpp): its proposal, the drafted rows and
  // the draft's distribution, then the accept.
  struct StepPlan;
  void plan_begin(StepPlan& sp, int B);                        // seed, lookup proposal
  void plan_draft(StepPlan& sp, int B, bool graph);            // the drafter's block and the lattice walk
  // The lattice walk over the drafter's candidates and scores for this sequence (E rows of each).
  void plan_walk(StepPlan& sp, int B, const int* cand, const float* scores);
  void plan_accept(StepPlan& sp, const float* vals, const int* ids);  // rows' top K -> accepted, next
  void plan_count(StepPlan& sp);                               // the session's counters and shadow records
  // The batch's verify: every plan's rows (pin_->rows) in one pass, then each sequence's top K at its rows.
  void verify_batch_launch(std::span<StepPlan> plans, int V);
  // The drafter over the blocks of every plan's session at once (M4 stage 4): pin_->draft_ids (B rows per
  // sequence) -> pin_->cand and pin_->scores (E = B - 1 rows per sequence, in order).
  void draft_batch_launch(std::span<StepPlan> plans, int B);
  // The batch path's graphs (M4 stage 4), per order of sessions in the batch: they bake the sessions' cache, state
  // and ring pointers and the row split. The draft graph has one shape (B rows per sequence); the verify graphs
  // are keyed by each sequence's top K; each sequence's commit graph is its session's, keyed by its slot in the
  // batch and its accepted count (slot 0 is the one-row step's own commit graph).
  struct BatchGraphs {
    cudaGraphExec_t draft = nullptr;
    std::map<std::vector<int>, cudaGraphExec_t> verify;
  };
  std::map<std::vector<int>, BatchGraphs> batch_graphs_;

  // CUDA graphs of the step (M3). The variable part of a step is the accepted count, which only the commit
  // depends on, so there is one commit graph per count; the draft and verify graphs have fixed shapes
  // (block B; the verify's attention launches the key ranges of the whole max_context, empty ones exiting at
  // once). Positions are read on the device from pos_dev_, written before every graph step.
  bool graphs_ = false;          // step graphs in use (opts_.step_graphs and nothing preventing them)
  bool graphs_allowed_ = false;  // nothing prevents them (a drafter, no LING_PROFILE or LING_KSPLIT)
  bool use_dev_pos_ = false;  // set while capturing: dpos() points the kernels at pos_dev_
  // pos_dev_[i]: the position of the batch's i-th sequence (the one-row step uses [0]); dpos() reads slot
  // dpos_slot_ (a batched sequence's commit: its slot in the batch).
  int* pos_dev_ = nullptr;
  int dpos_slot_ = 0;
  kernels::DevPos dpos(int off = 0) const {
    return use_dev_pos_ ? kernels::DevPos(s_->pos + off, pos_dev_ + dpos_slot_, off) : kernels::DevPos(s_->pos + off);
  }
  // The graphs themselves are per session (Session::draft_graph ...): they bake the session's cache and
  // state pointers into their kernels' arguments.
  // Runs `body` as the graph in `slot` (captured on first use), or eagerly when graphs are off. A failed
  // capture turns graphs off for the engine's lifetime, says why, and runs `body` eagerly.
  void run_graph(cudaGraphExec_t& slot, const char* what, const std::function<void()>& body,
                 cudaStream_t st = nullptr);  // st: the stream to capture and launch on (stream_ by default)
  cudaGraphExec_t capture(const std::function<void()>& body, cudaStream_t st);
  void clear_graphs();
  // Page-locked host buffers for every copy inside a step: a graph's memcpy nodes read and write fixed
  // host addresses when they run, so these must stay put and must not be pageable.
  struct StepPinned {
    int pos;
    int seq_pos[kernels::kMaxSeqs];  // a batched step's positions, per sequence
    int draft_ids[kernels::kMaxStreamRows];  // the drafter's block: the anchor, then mask tokens
    int rows[kernels::kMaxStreamRows];       // the verified rows: the anchor, then the drafted tokens
    int cand[kernels::kMaxStreamRows * 16];
    float scores[kernels::kMaxStreamRows * 16 * 16];
    float topk_vals[kernels::kMaxStreamRows * 64];
    int topk_ids[kernels::kMaxStreamRows * 64];
  };
  StepPinned* pin_ = nullptr;

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
  // --prefill-bf16: the BF16 views of xn_, t2_, z_, mixed_ and core_ (the same allocations, half the bytes used),
  // each live only between its producer and its consumer inside one DeltaNet layer of a prefill pass.
  __nv_bfloat16 *xn_bf16_ = nullptr, *t2_bf16_ = nullptr, *z_bf16_ = nullptr, *mixed_bf16_ = nullptr;
  __nv_bfloat16* core_bf16_ = nullptr;
  uint8_t *xq_ = nullptr, *xsf_ = nullptr;  // the prefill path's quantized activations and their block scales
  uint8_t *xq2_ = nullptr, *xsf2_ = nullptr;
  __half* xh_ = nullptr;     // the rows path's FP16 activations [32][max K]
  float* xinv_ = nullptr;    // and their per-row scales
  int* ids_dev_ = nullptr;

  std::vector<int> layer_slot_;  // layer -> index into a session's kcache/vcache or gdn/conv
  int n_linear_ = 0;             // DeltaNet layers
  // The checkpoint store (M4 §1), shared by the sessions and owning its identity: each entry is the state
  // after exactly `tokens` (a prompt prefix of at most checkpoint_limit ids), found by a hash of the prefix
  // and confirmed by comparing the ids. It outlives the history it was taken from; resuming from it also
  // needs the prefix's KV, which the target session holds or copies from another session.
  // Each entry's last use: its save, or a prefill whose prompt resumes from it or extends its prefix.
  struct Checkpoint {
    int pos, slot;
    uint64_t hash;
    std::vector<int> tokens;
    uint64_t used = 0;
  };
  uint64_t ckpt_clock_ = 0;
  std::vector<StateSlot> ckpt_slots_;
  std::vector<Checkpoint> ckpts_;  // sorted by position
  bool ckpt_match(const Checkpoint& k, const std::vector<int>& prompt, const std::vector<uint64_t>& prefix_hash) const;

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
  // The sparse head (BACKLOG §19.19): its factors, the fixed rows' mask, the target head in the checkpoint's layout
  // (gathered by row), and the step's selection.
  bool sparse_head_ = false;
  int sh_rank_ = 0, sh_cap_ = 0;
  __nv_bfloat16 *sh_us_ = nullptr, *sh_vt_ = nullptr;
  uint8_t *sh_fixed_ = nullptr, *sh_values_ = nullptr, *sh_scales_ = nullptr;
  float* sh_p_ = nullptr;
  void* sh_scratch_ = nullptr;
  int *sh_rows_ = nullptr, *sh_nrows_ = nullptr;
  float *topk_scratch_ = nullptr, *vals_dev_ = nullptr;
  int* ids_out_dev_ = nullptr;
  // The drafter attends to the last sliding_window positions only, so its context KV is a ring of
  // ring_ positions per session, position p at p % ring_ (M4 §1). The slack past the window keeps a resume
  // usable when the session's history ran up to that many tokens past the new prompt (a re-rendered last
  // output); the extra kMaxStreamRows cover a commit's rows.
  int ring_ = 0;
  struct ShadowRecord {
    int pos;                  // the anchor's position
    std::vector<int> tokens;  // the lookup's proposal for the positions after it
    int match, dflash_accepted, bucket;
  };
  struct Session;
  // Scores s's records whose continuation is known (all: the rest as far as known) into s.spec.
  void score_shadow(Session& s, bool all);

  // A resident session (M4 §1): everything that belongs to one sequence.
  struct Session {
    std::vector<__nv_bfloat16*> kcache, vcache;  // per full-attention layer, [Hkv][max_context][D]
    std::vector<float*> gdn, conv;                // DeltaNet and conv state, per linear layer
    std::vector<int> history;  // the tokens whose KV the caches hold (kept when another session takes the slot)
    size_t prompt_len = 0;     // the last prefill's prompt: history past it is the current request's output
    int turn_start = 0;        // where that prompt's last turn starts: after its last boundary_token (or its end)
    // The state after history[0, snap_pos) (the end of the last prompt), valid while those tokens stay.
    StateSlot snap;
    int snap_pos = -1;
    std::vector<__nv_bfloat16*> dkc, dvc;  // the drafter's context KV ring, per drafter layer
    // The ring holds positions [dkv_lo, history.size()), and of those only the last ring_ (dkv_from()).
    int dkv_lo = 0;
    int pos = 0;
    std::vector<ShadowRecord> shadow;
    SpecStats spec;  // this session's speculative steps
    std::mt19937_64 rng;
    cudaGraphExec_t draft_graph = nullptr;
    std::map<int, cudaGraphExec_t> verify_graphs;  // by top-K (1 when greedy)
    std::vector<cudaGraphExec_t> commit_graphs;    // by accepted count + 1
    std::map<std::pair<int, int>, cudaGraphExec_t> batch_commit_graphs;  // a batched step's: (slot, count), slot > 0
    // With commit_overlap, commit_graphs and batch_commit_graphs hold the drafter's context KV only, and these the
    // DeltaNet replay on commit_stream_, keyed by (first row, count).
    std::map<std::pair<int, int>, cudaGraphExec_t> state_commit_graphs;
    // The pool's bookkeeping.
    bool owned = false;  // a session has been bound to the slot
    std::string key;     // the prompt_cache_key it was bound with ("" without one)
    std::chrono::steady_clock::time_point last_used{};
  };
  std::vector<Session> sessions_;
  Session* s_ = nullptr;  // the active session
  long evictions_ = 0;
  // The first position whose drafter context KV session s still holds.
  int dkv_from(const Session& s) const {
    return std::max(s.dkv_lo, static_cast<int>(s.history.size()) - ring_);
  }
  // Copies session `from`'s KV for positions [0, n), its drafter ring and its history into the active
  // session, which then holds what `from` holds for its first n tokens.
  void adopt_prefix(const Session& from, int n);

  std::vector<float> logits_;
  bool profile_ = false;
  std::map<std::string, double> phase_seconds_;
};

}  // namespace ling
