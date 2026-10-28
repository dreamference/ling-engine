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
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/draft_guard.hpp"
#include "core/drafter.hpp"
#include "core/kernels.cuh"
#include "core/kv_pool.hpp"
#include "core/model.hpp"
#include "core/prefix_kv.hpp"
#include "core/sampling.hpp"
#include "core/tree.hpp"
#include "core/vision.hpp"

namespace ling {

// Images (specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.3, §20.7): a prompt holds an image as `tokens()` copies of the
// image's id, an id at or above kImageTokenBase that the server's registry gives each distinct image, so every prefix
// match tells two images apart. The engine never looks such an id up in the vocabulary.
constexpr int kImageTokenBase = 1000000;
struct PromptImage {
  int token = 0;              // the image's id in the prompt
  int grid_h = 0, grid_w = 0; // the patch grid (both even)
  int tokens() const { return grid_h * grid_w / 4; }
  std::shared_ptr<const std::vector<uint16_t>> patches;  // BF16 bits [grid_h * grid_w][1536], merge-window order
};

struct EngineOptions {
  int max_context = 65536;
  int prefill_chunk = 2048;  // measured: 647 tokens/s against 582 at 1024 on a 7.7K-token prompt
  std::string draft_dir;     // the DFlash2 drafter's checkpoint; empty: no speculation
  int draft_block = 16;      // rows per verify: the anchor and block - 1 drafted tokens (2 .. 32)
  // BACKLOG 33: the drafter's fc and conv kernel projections as NVFP4 copies rounded at load, read on the rows path
  // instead of BF16 (the conv projections' BF16 originals are then not kept; the fc's stay for the prompt's chunks).
  // Changes only the drafter's proposals, never what the verify accepts. LING_DRAFT_NVFP4=0/1 overrides.
  bool draft_nvfp4 = true;
  // The context-lookup source: 0 off; 1 shadow (proposed and scored afterwards against what was really
  // generated, never verified); 2 auto (verified instead of the drafter's chain when the match is at
  // least lookup_min_match tokens long, and the drafter's pass is skipped).
  int lookup_mode = 1;
  int lookup_min_match = 8;
  // BACKLOG 10: the drafter-off guard (core/draft_guard.hpp, which justifies the numbers). Per session, when the running
  // mean of tokens per speculative step (accepted + 1) falls below draft_off_below, the session's steps are plain
  // one-row decodes (Engine::step, no drafter pass) with one speculative probe every 16 of them, until the mean is
  // back at draft_off_below + draft_on_margin. 0: the guard off. A negative margin is for ling-run's check only
  // (set_draft_guard). Greedy output is token for token the same either
  // way; sampled output keeps its distribution but not its tokens (as speculative against plain decoding already
  // does), so the guard is kept to greedy requests by default (the standing rule: a change merged without the owner
  // leaves every output token as it was); draft_off_sampled = true extends it to sampled ones. LING_DRAFT_OFF_BELOW,
  // LING_DRAFT_ON_MARGIN and LING_DRAFT_OFF_SAMPLED=0/1 override these.
  double draft_off_below = 1.5;
  double draft_on_margin = 0.25;
  bool draft_off_sampled = false;
  // Prefix checkpoints: DeltaNet and conv states kept at chosen prompt positions (the end of the first two
  // messages, i.e. after `boundary_token`, and multiples of prefill_chunk up to checkpoint_limit), so a later
  // prompt that shares the prefix resumes there. Each costs ~157 MB; 0 turns them off.
  int prefix_checkpoints = 8;
  int boundary_token = -1;  // <|im_end|>
  int checkpoint_limit = 32768;
  // BACKLOG 34: a checkpoint also where the prompt leaves the longest prefix it shares with another conversation (a
  // resident session's history or a stored entry), so the next prompt of either branch resumes there. Exact (a kept
  // state at a multiple of kResumeAlign, as every other). With inpass_checkpoints its state is captured inside a pass
  // (alone it cost a short extra pass, BACKLOG 34). LING_BRANCH_CKPT=0/1 overrides.
  bool branch_checkpoints = true;
  // BACKLOG 35: the message ends' and the branch point's states captured inside the pass that crosses them (as the
  // end-of-prompt snap is), so they cut no chunk short; a chunk holding the snap or a second such point still ends
  // there. Exact: the captured state is the one a chunk ending there would leave. LING_CKPT_INPASS=0/1 overrides.
  bool inpass_checkpoints = true;
  // BACKLOG 35: the prefix KV store (core/prefix_kv.hpp), the KV of the prompts' shared beginnings (up to the
  // furthest of the first two message ends and the branch point) kept beyond the resident sessions in a radix tree,
  // with states at those positions, in at most this many MB, allocated once at start from what the engine's own
  // memory leaves above a reserve (LING_PREFIX_KV_RESERVE_MB, 16 GB): it shrinks or stays off rather than grow the
  // footprint while serving. A prompt resumes from it as from another session's KV. Exact: the copies are the KV and
  // states its own prefill computes. 0: off. LING_PREFIX_KV_MB overrides.
  int prefix_kv_mb = 4096;
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
  // P4 (BATCHING §5.3): the full-attention KV in 64-position pages of one pool, each session reading and writing its
  // positions through a page table, instead of a flat slot per session. Only addresses change, so every result is the
  // flat layout's bit for bit. P4b: a session maps pages lazily, as its passes reach them, and gives back the ones past
  // a new prompt's end (Engine's KV page pool, below). LING_KV_PAGES=0/1 overrides.
  bool kv_pages = false;
  // P4b: the pool's size in pages (each kKvPageTokens positions of every full-attention layer's K and V, 4 MiB for
  // Qwen3.8-27B), besides the reserved page 0. 0, or more than can ever be mapped: sessions x max_context positions.
  // ling-serve sets it from its memory plan (serve/admission.hpp) or --kv-pool-mb.
  int kv_pool_pages = 0;
  // M4 §1: resident sessions, each a fixed slot with its own KV cache, DeltaNet and conv state, end-of-prompt
  // state, drafter context ring, history and RNG, all allocated here. Requests run one at a time in the
  // active session (select it with activate_session); the prefix checkpoints are shared by all of them.
  int sessions = 1;
  // M4's step savings (reports/M3.md §7 item 4): a speculative step's commit replays the DeltaNet and conv state
  // over the accepted rows on a second stream, beside the next step's draft (which reads none of that state), and
  // the next verify waits for it. The same kernels on the same inputs, so the same bits. LING_COMMIT_OVERLAP=0/1
  // overrides.
  bool commit_overlap = true;
  // The prefill beside a decode (drafts/M4-prefill-beside-decode-2026-10-17.md, option B): a prefill chunk runs on a
  // second stream with its own scratch while other sessions decode on the main stream (prefill_chunk_launch /
  // prefill_chunk_finish). > 0: the second stream belongs to a green context of that many SMs, so the chunk's
  // kernels stay off the other SMs and the decode keeps them; -1: a plain second stream (no partition); 0: off.
  // The same kernels on the same inputs, so the same bits (--interleave-check with --side-prefill).
  int side_prefill = 0;
  // Tree verification (BACKLOG 26): this many drafter candidates beyond the chain verified in the same pass, as a
  // tree (core/tree.hpp), on one-stream steps; 0: the chain alone. On by default since 2026-10-19 (the owner's
  // decision) at the measured shape: the chain's first 8 drafts plus 7 nodes at temperature 1.5, 16 rows.
  // LING_TREE (or ling-serve's --tree; 0 turns it off), LING_TREE_T, LING_TREE_W, LING_TREE_DEPTH, LING_TREE_CHAIN
  // and LING_TREE_MIN override these.
  int tree_nodes = 7;
  double tree_temp = 1.5;
  int tree_width = 16;
  int tree_max_depth = 1000;
  int tree_chain = 8;        // the chain's length within the block (the block's later positions hold tree nodes)
  double tree_min_mass = 0;  // a step whose extra nodes' summed path probability is below this verifies the chain alone
  // BACKLOG 19: the drafter's context-aware sparse head. "full": the target's whole LM head. "sparse" (the default):
  // per step the drafter scores the fixed list, the session's context vocabulary (the token ids of its prompt and
  // output) and the `sparse_head_dynamic` rows a rank-256 copy of the head rates best, and takes its top 16 among
  // them (the target verifies every token, so the output is the full head's; only the proposals change); a session
  // whose recent tokens are too often outside the fixed list and its context falls back to the full head.
  // sparse_head_dir: model.safetensors with `us` (BF16 [vocab][256], the head's left factor times its singular
  // values), `vt` (BF16 [256][hidden]) and `fixed` (I32, the rows always scored), bench/sparse_head/build.py's;
  // empty: <draft_dir>/sparse-head; without a usable file there the drafter keeps the full head (one line on stderr).
  // LING_DRAFT_HEAD (full|sparse) and LING_SPARSE_HEAD_DIR override these, and
  // LING_SH_DYNAMIC, LING_SH_CONTEXT, LING_SH_WINDOW, LING_SH_HI, LING_SH_LO and LING_SH_ACCEPT_FLOOR the rest.
  std::string draft_head = "sparse";
  std::string sparse_head_dir;
  int sparse_head_dynamic = 16384;
  int sparse_head_context = 32768;  // the most context rows beyond the fixed list a session adds (0: no union)
  // The fallback: over the last `window` tokens of the session (its prompt's tail first), the share that were outside
  // the fixed list and the context before them. Above `hi` the session drafts with the full head, below `lo` it
  // returns to the sparse one. accept_floor > 0: also the full head (for a window of tokens) when the sparse head's
  // running mean of tokens per step falls below it.
  int sparse_head_window = 64;
  double sparse_head_hi = 0.15, sparse_head_lo = 0.08, sparse_head_accept_floor = 0;
  // Images (VISION §20.3): the checkpoint's vision tower, loaded with scratch for images of up to vision_max_pixels
  // (the processor's maximum by default), and a cache of encoded images (their embeddings, by image id) of at most
  // vision_cache_mb, reserved at start. Off: an image id in a prompt is an error.
  bool vision = false;
  long vision_max_pixels = 16777216;
  int vision_cache_mb = 512;
};

struct EngineStats {
  int image_tokens = 0;         // image tokens in the prefilled part of the prompt
  int images_encoded = 0;       // images the tower ran on (the rest came from the embedding cache)
  double vision_seconds = 0;    // the tower's GPU time
  double prefill_seconds = 0;  // wall time from begin_prefill to the last chunk (other sessions' work included)
  double compute_seconds = 0;  // the prefill's own chunks only
  int prefill_tokens = 0;
  int reused_tokens = 0;
  int store_tokens = 0;  // of those, restored from the prefix KV store (BACKLOG 35)
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
  // BACKLOG 26: tree verifies, those whose accepted path left the chain, and the off-chain tokens they accepted.
  long tree_steps = 0, tree_branch_steps = 0, tree_off_chain = 0;
  // BACKLOG 19: drafts with the sparse head, and the session's switches to the full head (its fallback).
  long sparse_steps = 0, sparse_fallbacks = 0;
  // BACKLOG 10: plain one-row steps the drafter-off guard took instead of a speculative step (not counted in `steps`),
  // and its switches (off, or back on).
  long drafter_off_steps = 0, drafter_switches = 0;
  // Shadow scoring, per bucket of match length (3, 4-7, 8-15, 16-31, 32+): steps with a proposal, the
  // tokens the proposal would have had accepted (its agreement with what was generated next), and the
  // drafter's accepted tokens on the same steps. `shadow_best` adds up the better of the two per step.
  long shadow_steps[5] = {}, shadow_lookup[5] = {}, shadow_dflash[5] = {};
  long shadow_best = 0, shadow_total_steps = 0, shadow_dflash_all = 0;
};

// P4b: the KV page pool has no free page for a pass (Engine's KV page pool, below). Thrown before anything of the
// pass is launched or changed; ling-serve's scheduler admits and steps requests so that it is never reached.
class KvPoolExhausted : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
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
                                    bool from_scratch = false, const std::vector<PromptImage>* images = nullptr);
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
    std::vector<int> soft;    // positions whose states are captured inside a pass and saved (BACKLOG 35)
    int kv_q = 0;              // the prefix KV store gets prompt[0, kv_q) at the end (BACKLOG 35)
    std::vector<int> kv_cuts;  // with states at these positions
    uint64_t kv_epoch = 0;
    uint64_t ckpt_epoch = 0;  // the checkpoint store's clock at begin_prefill: entries used since are this prompt's
    std::chrono::steady_clock::time_point t0;
    EngineStats stats;
    bool done() const { return next >= static_cast<int>(prompt.size()); }
    // Images (VISION §20.3 (b)): the prompt's images, and each run of an image's ids: where it starts, how many
    // tokens, the image (an index into images), the merged grid (rows, cols) and the RoPE position of its first row.
    std::vector<PromptImage> images;
    struct Span {
      int start = 0, len = 0, image = 0, rows = 0, cols = 0, base = 0;
    };
    std::vector<Span> spans;
    bool has_images() const { return !spans.empty(); }
  };
  // `images`: the images whose ids the prompt holds (VISION §20.3); an image id without its image is an error.
  PrefillCursor begin_prefill(const std::vector<int>& prompt, bool from_scratch = false,
                              const std::vector<PromptImage>* images = nullptr);
  bool has_vision() const { return vision_ != nullptr; }
  VisionTower* vision() { return vision_.get(); }
  // The RoPE positions (temporal, height, width) of prompt rows [from, to) into out (3 per row), and the delta of the
  // positions after the prompt (VISION §20.1). Public for the checks.
  static void rope_rows(const PrefillCursor& cur, int from, int to, std::vector<int>& out);
  static int rope_delta_after(const PrefillCursor& cur);
  bool prefill_chunk(PrefillCursor& cur, int max_tokens = 0);
  // The same chunk beside other sessions' work (EngineOptions::side_prefill): prefill_chunk_launch queues the
  // chunk's pass on the side stream, with the side scratch, and returns at once; the caller goes on with other
  // sessions' steps on the main stream; side_ready() says whether the pass has finished on the GPU, and
  // prefill_chunk_finish() does the rest of prefill_chunk (it waits for the pass if needed) and returns what
  // prefill_chunk returns, the cursor's session active. One chunk in flight at a time; nothing else may run a
  // prefill, bind or prefill a session meanwhile (activate_session and begin_prefill wait for it).
  bool side_enabled() const { return opts_.side_prefill != 0; }
  bool side_busy() const { return side_cur_ != nullptr; }
  void prefill_chunk_launch(PrefillCursor& cur, int max_tokens = 0);
  bool side_ready();
  bool prefill_chunk_finish();
  void side_join();     // waits until the side stream is idle (its chunk's pass and the work its finish queued)
  void side_abandon();  // the in-flight chunk's request is gone: waits for the pass and forgets it
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
  // Whether session i's next step for p is a plain one-row step because its drafter-off guard is off and no probe is
  // due (BACKLOG 10); speculate() then takes that step.
  bool drafter_off(int session, const SamplingParams& p) const;
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
  // The drafter-off guard's threshold and margin (EngineOptions::draft_off_below, draft_on_margin), every session's
  // guard reset; ling-run --spec-check compares runs with and without it. A margin of -off_below or less makes every
  // probe switch the drafter back on and its next step switch it off again: the check's way through both switches.
  void set_draft_guard(double off_below, double on_margin) {
    opts_.draft_off_below = off_below;
    opts_.draft_on_margin = on_margin;
    guard_cfg_.off_below = off_below;
    guard_cfg_.on_at = off_below + on_margin;
    for (Session& s : sessions_) s.guard.reset();
  }
  double draft_off_below() const { return opts_.draft_off_below; }
  double draft_on_margin() const { return opts_.draft_on_margin; }
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
  // ---- The KV page pool (P4b, BATCHING §5.3; with kv_pages only, else every count is 0). ----
  // A session maps a page for every 64 positions it writes, when a pass first writes them (lazily); begin_prefill
  // takes the prompt's pages and kKvGrowthPages more at once, and gives back the session's pages past those; a
  // session's pages go only with the session (kv_evict, reset). A released page becomes free at a later step boundary,
  // once every stream that could still read it has passed the point of its release (deferred frees).
  static constexpr int kKvGrowthPages = 2;
  bool kv_paged() const { return kv_pages_; }
  int kv_pages_total() const { return kv_book_ ? kv_book_->capacity() : 0; }
  // Free pages, after retiring the released ones whose readers have all finished.
  int kv_pages_free();
  int kv_pages_pending() const { return kv_book_ ? kv_book_->pending_pages() : 0; }
  // The pages evicting session i frees (P4c: those it shares with another session or the prefix KV store stay).
  int kv_session_pages(int session) const;
  // P4c: the pages freed once the listed sessions and, with `store`, the prefix KV store let go of theirs (a page
  // every one of whose references is among them).
  int kv_reclaimable(const std::vector<int>& sessions, bool store) const;
  // P4c: the pages evicting the whole prefix KV store frees, and the eviction of its least recently used leaf (the
  // pool's first reclaim, before idle sessions); false when the store is empty.
  int kv_store_pages() const;
  bool kv_store_evict();
  // What begin_prefill of a prompt of `prompt_len` tokens in session i draws from the pool, net: every page index below
  // the prompt's end plus kKvGrowthPages pages (kv_admit_pages) that the session does not map yet, less the pages it
  // maps past that point, which begin_prefill gives back first (0 when those are more). Exact: begin_prefill rewrites
  // the session's own pages below that point whatever it resumes from (P4b copies a donor's or the store's KV into
  // them), so a scheduler that admits only when this many pages are free never sees begin_prefill throw.
  int kv_admit_need(int session, size_t prompt_len) const;
  // The pages a prompt of `prompt_len` tokens maps in all (its end plus kKvGrowthPages pages): more than
  // kv_pages_total() and it can never run.
  int kv_admit_pages(size_t prompt_len) const { return KvPool::pages_for(kv_target(prompt_len)); }
  // The pages the next decode step of session i maps (a speculative one: every row its verify may write, the tree's
  // included; else one row). speculate() and speculate_batch() map them before anything else of the step.
  int kv_step_need(int session, bool spec) const;
  // Session i's pages released and the session emptied (an idle conversation evicted for the pool's sake: its next
  // turn prefills from scratch or from the prefix KV store). The slot becomes free for a new session.
  void kv_evict(int session);
  // The least recently used session that maps pages and is not in `busy` (running a request), or -1: what kv_evict
  // reclaims first. Protection (free_session's protect_seconds) does not count here: the pool's need overrides it.
  int kv_lru_idle(const std::vector<bool>& busy) const;
  long kv_evictions() const { return kv_evictions_; }
  // The step boundary: released pages whose readers have finished become free (wait: synchronize with those readers
  // first). Returns whether nothing is left pending.
  bool kv_retire(bool wait);

  // FNV-1a over every DeltaNet and conv state, bit for bit (the exactness tests compare runs with it).
  uint64_t state_hash();
  // FNV-1a over the drafter's context KV of the active session's positions [dkv_from, history), bit for bit; 0
  // without a drafter.
  uint64_t draft_hash();

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
  bool kv_pages_ = false;                 // opts_.kv_pages or LING_KV_PAGES (P4)
  __nv_bfloat16* kv_pool_ = nullptr;      // P4: the pool, [page][layer][K, V][kv head][kKvPageTokens][D]; page 0 unused
  size_t kv_page_elems_ = 0;              // P4: elements per page (all full-attention layers' K and V)
  // P4c: the active session maps session `from`'s pages for positions [0, n) (shared, no copy).
  void kv_pages_adopt(const Session& from, int n);
  // P4b: the pool's bookkeeping (core/kv_pool.hpp), with kv_pages.
  std::unique_ptr<KvPool> kv_book_;
  long kv_evictions_ = 0;
  // Deferred frees: an event recorded on each stream that may read a page (the main stream, the side prefill's stream,
  // the commit stream) at every release; retire() runs once all three have passed them. A later record on a stream
  // covers the earlier releases too, so one event per stream suffices.
  cudaEvent_t kv_release_ev_[3] = {};
  bool kv_release_rec_[3] = {};
  cudaStream_t main_stream() const { return side_active_ ? side_.stream_ : stream_; }
  cudaStream_t side_stream() const { return side_active_ ? stream_ : side_.stream_; }
  int kv_target(size_t prompt_len) const;  // a prompt's end plus kKvGrowthPages pages, within max_context
  // Rows the next step of s may write: 1, or a speculative step's verify rows (the tree's included).
  int kv_step_rows(const Session& s, bool spec) const;
  // Before a pass on stream st writes positions [from, to) of s: maps the pages (KvPool::ensure), makes the copies it
  // asks for (copy-on-write; none in P4b, where no page is shared) and writes the changed table entries, all on st.
  // Throws KvPoolExhausted, with nothing changed, when the pool is short even after waiting for the deferred frees.
  // Never inside a graph capture (a captured table update would replay stale entries): there it only checks.
  void kv_ensure(Session& s, int from, int to, cudaStream_t st);
  // s's pages wholly at or past position n released (their table entries back to page 0 on the main stream), and
  // the release recorded on every stream.
  void kv_release(Session& s, int n);
  void kv_table_sync(Session& s, cudaStream_t st);  // the device table to s.kv_ids, the changed entries only
  void kv_record_release();
  // Runs a prefix KV store call that may drop page references, recording the release when it did.
  void kv_store_dropped(const std::function<void()>& op);
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
  void linear_bf16_drafter(const Bf16Weight& w, const float* x, int M, float* y);  // the drafter's BF16 matrices
  // A conv kernel projection: its NVFP4 copy on the rows path when it has one (BACKLOG 33), else the BF16 matrix.
  void linear_draft_kproj(const Bf16Weight& w, const Fp4Weight& w4, const float* x, int M, float* y);
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
  // from: a slot, not the state; soft: a message end or branch point (BACKLOG 35), kept before chunk multiples
  void save_checkpoint(int pos, uint64_t epoch, const StateSlot* from = nullptr, bool soft = false);
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
  // The overlap probe (BACKLOG 28; LING_GDN_FORK, off by default): in a single sequence's verify, a DeltaNet layer's
  // latency-bound chain (the a/b projections, gating, conv, recurrence) runs on fork_stream_ beside the in_z
  // projection's weight stream on stream_; the same kernels on the same inputs, so the same bits. 1: a plain stream;
  // 2: the highest stream priority, and the step graphs instantiated with node priorities.
  int fork_ = 0;
  // The megakernel's stage 1 in the step (LING_MK_FFN, on by default since 2026-10-19; LING_MK_FFN=0 turns it off):
  // the rows path's FFN as one persistent kernel (kernels::ffn_persistent), with its own FP16 buffer for the down
  // projection's input and one counter set per layer.
  int mk_ffn_ = 1;
  unsigned* mk_counters_ = nullptr;
  // Stage 2 (LING_MK_GDN, on by default since 2026-10-19; LING_MK_GDN=0 turns it off): a verify's DeltaNet
  // mixer as one persistent kernel; a tree verify's too since 2026-10-20 (LING_MK_GDN_TREE=0: its graph path).
  int mk_gdn_ = 1;
  unsigned* mk_gdn_counters_ = nullptr;
  int mk_attn_ = 0;  // stage 3 (LING_MK_ATTN=1): a verify's attention layer as one persistent kernel
  unsigned* mk_attn_counters_ = nullptr;
  unsigned long long* mk_attn_times_ = nullptr;  // LING_MK_TIMES: [layer][kernels::kMkAttnEvents]
  int mk_span_ = 0;  // stage 4 (LING_MK_SPAN=1): a verify's DeltaNet mixer and FFN in one persistent launch
  __half* xh3_ = nullptr;  // its post-attention norm's FP16 copy and the FFN's down input (not the mixer's buffers)
  float* xinv3_ = nullptr;
  __half* xh4_ = nullptr;
  float* xinv4_ = nullptr;
  int mk_layers_ = 0;  // stage 4 (LING_MK_LAYERS=1): the residual adds and the next norms inside the persistent kernels
  // LING_MK_TIMES=1: the persistent kernels' timings store ([layer][kMkTimesBlocks][kMkTimesPerBlock] globaltimer
  // stamps of each layer's last launch), summarized on stderr when the engine is destroyed.
  static constexpr int kMkTimesBlocks = 128;
  unsigned long long* mk_ffn_times_ = nullptr;
  unsigned long long* mk_gdn_times_ = nullptr;
  void mk_times_report();
  __half* xh2_ = nullptr;
  float* xinv2_ = nullptr;
  cudaStream_t fork_stream_ = nullptr;
  cudaEvent_t fork_xn_ = nullptr, fork_qkv_ = nullptr, fork_done_ = nullptr;
  // The commit's two halves: the DeltaNet replay (on commit_stream_ when overlapped) and the drafter's context KV.
  void commit_state_launch(int row0, int n, cudaStream_t st);
  void commit_state_async(int row0, int n, bool graph);  // the replay on commit_stream_, then commit_done_
  // Target features -> the drafter's context KV at positions pos0 .. pos0 + rows - 1.
  void draft_materialize(int rows, kernels::DevPos pos0, int cap_row0);
  void draft_propose(int anchor, int B, bool graph);  // the drafter's block, top-16 candidates and lattice
  void alloc_drafter();
  void alloc_sparse_head();
  // The drafter's head over n sequences' E rows each of dh_: dunary_ and dcand_ (the top K16 values and token ids per
  // row); sequence i's from the sparse head over masks[i] (its session's fixed list and context, BACKLOG 19), or the
  // full head when masks[i] (or masks) is null, each bit for bit what the sequence gets drafted alone.
  void draft_head(int n, int E, const uint8_t* const* masks);
  // Brings session s's context vocabulary up to its history and decides its head for the next draft (BACKLOG 19).
  // True: the sparse head.
  bool sparse_head_sync(Session& s);

  // The three parts of a speculative step as launch sequences with no host synchronization inside, so
  // each can be captured into a CUDA graph. Their inputs and outputs on the host side are pin_'s buffers.
  void draft_launch(int B, bool sparse = false);  // pin_->draft_ids -> pin_->cand, pin_->scores
  void verify_launch(int V, int K, bool tree = false);  // pin_->rows -> each row's top K in pin_->topk_vals / topk_ids
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
  void draft_batch_launch(std::span<StepPlan> plans, int B, unsigned sparse = 0);  // bit i: sequence i's sparse head
  // The batch path's graphs (M4 stage 4), per order of sessions in the batch: they bake the sessions' cache, state
  // and ring pointers and the row split. The draft graph has one shape (B rows per sequence); the verify graphs
  // are keyed by each sequence's top K; each sequence's commit graph is its session's, keyed by its slot in the
  // batch and its accepted count (slot 0 is the one-row step's own commit graph).
  struct BatchGraphs {
    cudaGraphExec_t draft = nullptr;
    std::map<unsigned, cudaGraphExec_t> draft_sparse;  // with the sparse head, by which sequences take it (BACKLOG 19)
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
  int* rdelta_dev_ = nullptr;  // rdelta_dev_[i]: the rope delta of the batch's i-th sequence, written beside pos_dev_
  int dpos_slot_ = 0;
  kernels::DevPos dpos(int off = 0) const {
    return use_dev_pos_ ? kernels::DevPos(s_->pos + off, pos_dev_ + dpos_slot_, off).with_rope(0, rdelta_dev_ + dpos_slot_)
                        : kernels::DevPos(s_->pos + off).with_rope(s_->rope_delta, nullptr);
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
    int rdelta;                       // the one-sequence step's rope delta
    int seq_rdelta[kernels::kMaxSeqs];  // a batched step's rope deltas
    int draft_ids[kernels::kMaxStreamRows];  // the drafter's block: the anchor, then mask tokens
    int rows[kernels::kMaxStreamRows];       // the verified rows: the anchor, then the drafted tokens
    int cand[kernels::kMaxStreamRows * 16];
    float scores[kernels::kMaxStreamRows * 16 * 16];
    float topk_vals[kernels::kMaxStreamRows * 64];
    int topk_ids[kernels::kMaxStreamRows * 64];
    kernels::TreeDesc tree;                     // a tree verify's description (BACKLOG 26)
    int gather_rows[kernels::kMaxStreamRows];  // its accepted path's rows
  };
  StepPinned* pin_ = nullptr;

  EngineOptions opts_;
  std::unique_ptr<Model> model_;
  // ---- Images (VISION §20.3). ----
  std::unique_ptr<VisionTower> vision_;
  struct EmbEntry {
    __nv_bfloat16* p = nullptr;
    size_t bytes = 0;
    uint64_t used = 0;
  };
  std::map<int, EmbEntry> emb_;  // the embedding cache, by image id
  size_t emb_bytes_ = 0, emb_cap_ = 0;
  uint64_t emb_clock_ = 0;
  cudaMemPool_t emb_pool_ = nullptr;
  // The embedding of `im`, from the cache or the tower (run on stream_); entries of the ids in `keep` are not evicted.
  const __nv_bfloat16* image_embedding(const PromptImage& im, const std::vector<int>& keep, EngineStats& stats);
  // The current prefill chunk's images (rows of h_ replaced after the embedding) and RoPE table (null: text only).
  struct ImgRows {
    int row, count;
    const __nv_bfloat16* src;
  };
  std::vector<ImgRows> chunk_img_;
  int* mrope_dev_ = nullptr;  // [prefill_chunk][3]
  bool chunk_mrope_ = false;
  std::unique_ptr<DraftModel> draft_;
  cudaStream_t stream_ = nullptr;
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
    bool soft = false;  // a message end or a branch point
  };
  uint64_t ckpt_clock_ = 0;
  std::vector<StateSlot> ckpt_slots_;
  std::vector<Checkpoint> ckpts_;  // sorted by position
  bool branch_checkpoints_ = true;  // EngineOptions::branch_checkpoints
  bool inpass_ = false;             // EngineOptions::inpass_checkpoints (with a checkpoint store)
  StateSlot soft_slot_;             // a soft point's state, captured inside a pass, until chunk_end stores it
  const StateSlot* keep_slot_ = nullptr;  // where forward()'s keep_at state goes: the snap, unless set
  const StateSlot& keep_target() const { return keep_slot_ ? *keep_slot_ : s_->snap; }
  std::unique_ptr<PrefixKvTree> kv_tree_;  // EngineOptions::prefix_kv_mb
  std::unique_ptr<ArenaAlloc> kv_arena_;   // its buffers, carved from one allocation made at start
  void* kv_arena_base_ = nullptr;
  uint64_t kv_epoch_ = 0;
  StateSlot state_view(const void* p) const;  // a store state buffer as a slot
  // The active session's KV for positions [from, to) into (to_buf) or out of a store buffer.
  void kv_copy(void* buf, int buf_start, int buf_len, int from, int to, bool to_buf);
  void kv_restore(const PrefixKvTree::Match& m, int n);
  void kv_insert(const PrefillCursor& cur);  // the prompt's shared beginning into the store, at its prefill's end
  void copy_slot(const StateSlot& dst, const StateSlot& src);
  bool ckpt_match(const Checkpoint& k, const std::vector<int>& prompt, const std::vector<uint64_t>& prefix_hash) const;

  // A verify's per-layer DeltaNet inputs, kept for commit(): the in-projection before and after the conv,
  // the decay and the write strength, for up to 32 rows.
  std::vector<float*> v_pre_, v_post_, v_g_, v_beta_;
  float* conv_tmp_ = nullptr;
  // Tree verification (BACKLOG 26): the device copy of pin_->tree, set as tree_ while a tree verify's pass runs;
  // tree_grid_ gdn paths launched (fixed per configuration, the unused ones exit at once).
  kernels::TreeDesc* tree_dev_ = nullptr;
  const kernels::TreeDesc* tree_ = nullptr;
  int tree_grid_ = 1;
  int* gather_rows_dev_ = nullptr;
  bool tree_step(const SamplingParams& p, bool use_lookup, bool want_logits, int B) const;
  // The drafter-off guard (BACKLOG 10): its configuration, and whether it applies to requests with p.
  DraftGuardConfig guard_cfg_;
  bool guarded(const SamplingParams& p) const {
    return guard_cfg_.off_below > 0 && (p.temperature <= 0.f || opts_.draft_off_sampled);
  }

  // Drafter: target features [rows][layers * hidden] (BF16), its context KV per layer [max_context][kv],
  // and the block's buffers.
  std::vector<int> feature_layer_;  // target layer -> index of its feature slot, or -1
  __nv_bfloat16* cap_ = nullptr;
  __half* xh_fc_ = nullptr;  // BACKLOG 33: the FP16 rows of the fc's input for its NVFP4 copy
  int cap_stride_ = 0;
  std::vector<__nv_bfloat16*> dkc_, dvc_;
  float *dctx_ = nullptr, *dk_ = nullptr, *dv_ = nullptr, *dres_ = nullptr, *dh_ = nullptr, *dh2_ = nullptr;
  float *dout_ = nullptr, *dcoef_ = nullptr, *dq_ = nullptr, *dg_ = nullptr, *du_ = nullptr, *dattn_ = nullptr;
  float *dlogits_ = nullptr, *dunary_ = nullptr, *dhp_ = nullptr, *dscores_ = nullptr;
  int* dcand_ = nullptr;
  // The sparse head (BACKLOG 19): its factors, the fixed list (host mask), the step's selection, and the pinned and
  // device buffers a session's new context ids are marked through.
  bool sparse_head_ = false;
  int sh_rank_ = 0, sh_cap_ = 0, sh_nfixed_ = 0;
  __nv_bfloat16 *sh_us_ = nullptr, *sh_vt_ = nullptr;
  std::vector<uint8_t> sh_fixed_;
  float* sh_p_ = nullptr;
  void* sh_scratch_ = nullptr;
  int *sh_rows_ = nullptr, *sh_nrows_ = nullptr, *sh_ids_dev_ = nullptr;
  float* sh_xtmp_ = nullptr;  // a batch's full-head rows copied together (draft_head_rows)
  static constexpr int kShMarkMax = 4096;  // more new ids at once: the whole mask is uploaded
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
    // per full-attention layer, [Hkv][max_context][D]; P4 (--kv-pages): the layer's K and V bases in the pool
    std::vector<__nv_bfloat16*> kcache, vcache;
    uint64_t* pages = nullptr;           // P4: the device page table (element offsets of its pages), or nullptr: flat
    std::vector<uint64_t> page_host;     // P4: what the device table holds (written only through kv_table_set)
    std::vector<int> kv_ids;             // P4b: its pages' ids, KvPool's table (0: unmapped)
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
    // The RoPE position of the slot after the last prompt's images, less the slot (VISION §20.1): 0 for text-only
    // prompts; every pass after the prompt's prefill turns its rows at slot + rope_delta.
    int rope_delta = 0;
    std::vector<ShadowRecord> shadow;
    SpecStats spec;  // this session's speculative steps
    DraftGuard guard;  // BACKLOG 10: whether its next step drafts (reset with the session)
    std::mt19937_64 rng;
    cudaGraphExec_t draft_graph = nullptr;
    cudaGraphExec_t draft_graph_sparse = nullptr;  // the draft with the sparse head (BACKLOG 19)
    // The sparse head's per-session state (BACKLOG 19): the device mask of the rows always scored (the fixed list
    // joined with the context vocabulary), its host copy, how much of history it holds, and the fallback's window.
    uint8_t* sh_mask = nullptr;
    std::vector<uint8_t> sh_seen;
    size_t sh_synced = 0;
    bool sh_dirty = true;  // history was replaced: rebuild the mask from it
    bool sh_rewin = false;  // history was cut (a new turn): the window starts again from the tokens that follow
    int sh_ctx = 0;        // context rows beyond the fixed list
    std::vector<uint8_t> sh_win;  // the last window tokens' novelty, a ring
    int sh_win_pos = 0, sh_win_n = 0, sh_novel = 0;
    bool sh_full = true;  // the fallback: the full head
    size_t sh_cool_until = 0;  // the acceptance floor's fallback lasts until history reaches this length
    long sh_last_steps = -1, sh_last_acc = 0;
    bool sh_last_sparse = false;
    double sh_acc_ema = -1;
    std::map<int, cudaGraphExec_t> verify_graphs;  // by top-K (1 when greedy)
    std::map<int, cudaGraphExec_t> tree_verify_graphs;  // a tree verify's, by top-K
    kernels::GatherBuf* gather = nullptr;  // the buffers a tree's accepted path is gathered in (device)
    int ngather = 0;
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

  // The prefill beside a decode (side_prefill). The members a pass writes, swapped as a set: while side_active_ the
  // names below are the side pass's (its stream, its scratch, its active session, its quantized-input cache and a
  // pinned buffer for its logits), and side_ holds the main ones; otherwise the other way round. Everything else
  // (the model, the sessions, the checkpoint store, the decode's graphs and pinned step buffers) is shared and is
  // touched by one pass at a time: a session by its own pass only, the store from the host between passes.
  float* logits_host_ = nullptr;  // where a pass copies its last row's logits: logits_'s data, or the side's pinned buffer
  struct ChunkRun {
    int i = 0, end = 0, keep_at = -1, n = 0, pos0 = 0;
    int soft_at = -1;  // the soft point whose state the pass captured into soft_slot_ (-1: none, or the snap)
    std::chrono::steady_clock::time_point t;
  };
  ChunkRun chunk_begin(PrefillCursor& cur, int max_tokens);  // prefill_chunk's pass, launched
  bool chunk_end(PrefillCursor& cur, const ChunkRun& run);   // and everything after it
  void alloc_scratch();
#define LING_SIDE_SET(X)                                                                                     \
  X(h_) X(xn_) X(t1_) X(t2_) X(mixed_) X(z_) X(a_) X(b_) X(g_) X(beta_) X(core_) X(normed_) X(q_gate_) X(k_)     \
  X(v_) X(q_) X(gate_) X(attn_) X(attn_scratch_) X(logits_dev_) X(x_bf16_) X(xn_bf16_) X(t2_bf16_) X(z_bf16_)    \
  X(mixed_bf16_) X(core_bf16_) X(xq_) X(xsf_) X(xq2_) X(xsf2_) X(xh_) X(xinv_) X(ids_dev_) X(conv_tmp_) X(cap_)  \
  X(xh_fc_) X(dctx_) X(dk_) X(dv_) X(stream_) X(s_) X(quantized_) X(xq_scale_) X(xq_nvfp4_) X(logits_host_)
#define LING_SIDE_DECL(m) decltype(Engine::m) m{};
  struct SideSet {
    LING_SIDE_SET(LING_SIDE_DECL)
  };
#undef LING_SIDE_DECL
  SideSet side_;
  bool side_active_ = false;
  void swap_side();
  cudaEvent_t side_done_ = nullptr;      // recorded after the in-flight chunk's pass
  cudaEvent_t side_after_main_ = nullptr;  // the main stream's work so far (the side chunk starts after it)
  void* side_green_ = nullptr;           // the side stream's green context (CUgreenCtx), when side_prefill > 0
  PrefillCursor* side_cur_ = nullptr;    // the in-flight chunk's cursor
  ChunkRun side_run_;
};

}  // namespace ling
