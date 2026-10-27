# C2 in ling-serve: the fused pass (design, 2026-10-28)

BATCHING §5.5 and §6 (C2) put one weight read under a decode verify and a prefill chunk. The kernels exist and are
bit-exact (branches `batching/c2-fused-gemm` and `batching/c2-fp8-prefill`, [C2-fused-gemm](./C2-fused-gemm-2026-10-27.md)).
This note is how ling-serve runs the pass: the engine's fused forward, what stays where, the scheduler's choice, the
exactness gate, memory, and the numbers to expect. Built on branch `batching/c2-integrate` (main with C1 merged in),
behind `--fused-pass`, off by default until measured.

## 1. The shape of a fused step

A fused step is one speculative decode step of one or two sessions (a plain `speculate` or a `speculate_batch` pair)
with one prefill chunk of a third session *attached* to its verify. Only the verify's forward pass changes:

| Stage | Without a chunk (today) | With a chunk attached |
| --- | --- | --- |
| draft (drafter's block, lattice walk) | step graph | the same step graph |
| verify forward | step graph; megakernels (`attn_persistent`, `gdn_persistent`, `ffn_persistent`, span) | eager `forward_fused`: per projection one fused launch for the verify rows and the chunk rows |
| accept | host | host |
| commit (DeltaNet replay, the verify rows' drafter context) | step graph | the same step graph |
| the chunk's end (snap, checkpoints, its drafter context, prefix store) | `chunk_end` on the side stream | the same `chunk_end`, on the side stream |

So everything outside the verify forward runs exactly the code it runs today, and the chunk's bookkeeping is the side
chunk's (`prefill_chunk_finish`). A step with no chunk attached is unchanged: graphs, megakernels and all.

## 2. Per layer

The two row groups keep their own activations: the verify rows in the main set (`h_`, `xn_`, `xh_`/`xinv_`, ...), the
chunk rows in the side set (`side_.h_`, `side_.xq_`/`side_.xsf_`, ...) that the side prefill already allocates. Both are
addressed explicitly in one function; nothing is swapped mid-pass. The chunk's session is passed in (its KV, conv and
recurrent state, position); the verify's sessions come in `segs` as in `forward_rows`.

Streams: every fused GEMM runs on the main stream (the whole GPU). The verify rows' row-wise and state kernels run on
the main stream; the chunk rows' on the side stream (the green context of `--side-prefill` SMs). Around each
projection, an event pair: the main stream waits for the side stream's input (the chunk's quantized activations), then
the side stream waits for the GEMM's output. Between two projections the two groups' state kernels run concurrently.

| Layer | Verify rows (main stream, `forward_rows`' verify branches without megakernels) | Fused launch | Chunk rows (side stream, `forward_rows`' prefill branches) |
| --- | --- | --- | --- |
| attention | `rmsnorm_half` | q, k, v: `fused_gemm_fp8` (3 launches; the rows path's one multi-matrix launch gives the same bits per matrix) | `rmsnorm_quant` FP8 (FP32 copy when k or v quantize differently) |
| | q/k norm, RoPE, KV write into each verify session's cache, `attention_rows` (multi, single or tree) | | the same for the chunk's rows into the chunk session's cache, 256-query slices |
| | `sigmoid_mul`, `to_half_rows` | o: `fused_gemm_fp8`, chunk rows `+=` into `side_.h_` | `sigmoid_mul_fp8` |
| | `add_inplace` | | |
| DeltaNet | `rmsnorm_half` | in_qkv, in_z: `fused_gemm_fp8` (verify rows into `v_pre_`, `z_`; chunk rows into `side_.t2_`, `side_.z_`) | `rmsnorm_quant` FP8 with the FP32 copy |
| | a/b `bf16_rows`; conv on a copy of each session's window (or the tree conv); gating; recurrence from each session's state, left unwritten (the verify's inputs kept for commit) | | a/b `narrow_bf16`; `gdn_conv_prefill`, gating, `gdn_recurrent_prefill` into the chunk session's state, the snap or soft state captured at `keep_at` |
| | `gated_rmsnorm`, `to_half_rows` | out: `fused_gemm_fp8`, chunk `+=` | `gated_rmsnorm_fp8` |
| | `add_inplace` | | |
| FFN | `rmsnorm_half` | gate\|up: `fused_swiglu_nvfp4` (verify: gate and up FP32; chunk: SwiGLU straight to NVFP4) | `rmsnorm_quant` NVFP4 |
| | `silu_mul`, `to_half_rows` | down: `fused_gemm_nvfp4`, chunk `+=` into `side_.h_` | |
| | `add_inplace` | | |
| drafter features | `copy_rows_bf16` into `cap_` | | `copy_rows_bf16` into `side_.cap_` |
| lm head | final norm, every row | `fused_gemm_nvfp4`: the verify rows and the chunk's last row (one read instead of two) | final norm of the last row, quantized as the prefill quantizes it |

After the lm head the main stream does the verify's top K as today; the side stream copies the chunk's logits to its
pinned buffer and records `side_done_`, so `prefill_chunk_finish` takes over unchanged.

**What attaches.** A chunk attaches when `Engine::can_attach(cur)`: `--fused-pass` on, the side stream idle (no side
chunk in flight), the prompt has no images (the M-RoPE table path stays on the side stream), `--prefill-bf16` off, and
the prompt not done. Chunks of any length up to `--fused-chunk` (default 256) attach, including the one that captures
the end-of-prompt snap or a soft checkpoint inside the pass (`keep_at`): the chunk planning is `chunk_begin`'s, split
from its forward pass. Everything else takes the side stream as today; both paths give the same bits, so a prompt may
mix them.

**The side stream** stays: as the stream the chunk rows' own kernels run on inside a fused pass, and as the fallback
whenever no decode can carry a chunk (non-speculative sampling, a short block at the context's end), a chunk does not
attach, or the scheduler holds the fused pass back.

**The megakernels.** The verify rows of a fused pass take the rows path's kernels, not the persistent ones (they
contain their own stream GEMMs). The code holds that both give the same bits (the batch path never takes
`attn_persistent`/`gdn_persistent` and `--batch-check` passes against the one-sequence step that does); the
`--fused-check` (§5) compares against a plain step with the megakernels on and with `--graphs`, i.e. production's
step, so it proves it for this path too.

## 3. Graphs

A fused verify runs eagerly; the draft and commit around it replay their graphs as today. Capturing the fused verify
would freeze the chunk's session pointers, its length, `keep_at`, and launch the chunk's attention over full-context
key ranges (the dev-position mode). The bounded set the task asks for, graphs per batch size (1, 2) x per-sequence K x
chain/tree x with/without a 256-row chunk, needs each slot's session read from a device descriptor table: BATCHING
§5.2's refactor, which P2 does for the batch path anyway. Until then:

- The cost of eager: a verify forward is ~1,300 launches. Their host cost (~5 us each, ~7 ms) overlaps the GPU's
  ~170 ms pass; what remains is the inter-kernel gaps a graph removes, estimated at 1-3% of the pass. Measured in the
  first window as the fused pass against the same plain step eager and with graphs (`--fused-check` prints both).
- The memory upside: an eager fused verify captures nothing, so no graph memory appears after the start for it
  (§5.2's rule).

## 4. The drafter

The drafter's own pass (its block over the decode sessions, then the lattice walk) runs before the fused verify as
today, on the main stream, from its graph. The verify rows' drafter context (`draft_materialize` in `commit_launch`)
runs after the accept, as today. The chunk's drafter context (its features in `side_.cap_`, materialized by
`chunk_end` through the drafter's BF16 GEMM and K/V projections) runs on the side stream after the fused pass, beside
the decode's accept, commit and next draft. Fusing it with the commit's `draft_materialize` would save ~2 x 5 drafter
K/V reads of a few MB: not worth a kernel.

## 5. Scheduler

`--fused-pass` (off by default), `--fused-chunk N` (default 256, a multiple of 32), `--fused-share N` (default 0).

- **Carrier.** When a decode turn comes (`unit`'s speculative step, or `batch_step`'s pair from `batch_with`), every
  member batchable (`can_batch`), and a prefilling request's next chunk can attach, the step runs fused with that
  chunk. Afterwards the decode tokens are emitted as `batch_step` emits them; the chunk is finished
  (`prefill_chunk_finish`), and if it was the prompt's last, that request starts decoding (`begin_decode`, first token
  emitted) in the same turn.
- **Rider choice: C1's rule.** Of the prefilling requests whose next chunk can attach, the one with the fewest tokens
  left goes first, unless a longer one has yielded for 5 s (`yield_prefill`). A warm follow-up's append therefore
  rides the next decode pass: its TTFT is about one fused step plus its own first step.
- **A prefilling request's own turn** does nothing while some other request can carry its chunk; with no carrier
  (nobody decodes speculatively) it prefills as today: a full chunk alone, or on the side stream beside a
  non-speculative decode.
- **Hold.** `--fused-share N`: while two or more other requests decode, N decode passes run without a chunk after each
  attached one (C1's `--side-share` for the fused pass). Default 0: every decode pass carries a chunk while one waits.
  `--side-share` keeps governing side-stream chunks. Measured both ways before a default is set.
- **Batch of two.** `kMaxStreamRows / draft_block` = 2 sequences per verify, as today; the pair's 24 rows are the
  verify rows of the fused launch (the kernel takes up to 32: a second block per panel past 16). Four warm streams
  then run as two pairs, each pair's steps carrying the waiting chunks in turn.

## 6. Exactness gate

1. `ling-fused-tests`: the existing 214 cases plus the shapes the pass adds: attention q|k|v and o FP8, in_qkv|z FP8
   with FP32 `=` output, the NVFP4 lm head with one chunk row; decode 0-32 rows x chunk 0, 1, 32, 256 rows.
2. `ling-run --fused-check` (`--draft`, `--sessions 2`, prompts in pairs (A, B), greedy and sampled, both orders):
   each prompt first alone from a cold engine (prefill, then speculative decode with production's settings:
   megakernels on, tree on, `--graphs` as given); then A decodes while B's chunks attach to A's steps
   (`speculate_fused`), then the two take steps in turn. Compared bit for bit: B's state hash, drafter-context hash
   and logits after its prefill against the prefill alone; each sequence's tokens, state hash, drafter-context hash,
   steps and accepted drafts against its run alone. The number of chunks that attached and that fell back is printed
   (a pass with none attached is a failure). With `--sessions 3` and triples (A, B, C), A and C decode as a batch of
   two (chain, tree off, as `--batch-check`) while B's chunks attach.
3. `ling-run --interleave-check` and `--batch-check` unchanged (the paths a step without a chunk takes).
4. agentlong greedy 16/16 against one stream at 2 and 4 streams, warm and cold, with `--fused-pass`.

## 7. Memory

The fused pass allocates nothing new: the chunk rows live in the side set that `--side-prefill` allocates at start,
which `admission.hpp`'s `kScratchBytes` already counts ("their second set for the side prefill"); `--fused-pass`
therefore requires `--side-prefill` != 0. New at start: two CUDA events. The fused verify is not captured, so it adds
no graph memory after the start (BATCHING §5.2's rule; it removes nothing either). The fused kernel takes no
workspace (dynamic shared memory only).

## 8. Expected

From the prototype's timings (C2-fused-gemm §4, L2 flushed, 256-row chunk):

| | 12-16 verify rows | 24 rows (a pair) |
| --- | --- | --- |
| fused projections over the model | 129-134 ms | ~151 ms |
| + state kernels, the chunk's beside the verify's (max of ~24 and ~25-30) | ~+30 ms | ~+35 ms |
| + lm head (one read for both), eager gaps | ~+5-8 ms | ~+5-8 ms |
| fused verify pass | ~165-172 ms | ~190-195 ms |
| + draft and commit (graphs, unchanged) | ~+12 ms | ~+20 ms |
| a decode step with a chunk attached | **~180 ms** | **~210 ms** |

Against today: 205-313 ms for a step beside a side-stream chunk (§2.2), 222-229 ms in the probe. The chunk advances 256
tokens per carried step, ~1.2-1.4K tokens/s while something decodes (alone: ~1.8K at 2,048-token chunks), so a 39K cold
prime takes ~30 s of carried steps; a warm append of a few hundred tokens rides one or two steps.

Per-stream decode at ~6.2 tokens a step: ~34 tokens/s for a conversation stepping beside a prime (today ~24, A in the
probe 60.0 overall), ~30 each for a pair. At four warm streams the slowest conversation should rise from 27.3 tokens/s
toward ~32-35, pooled from 35.6 toward ~40; one stream unchanged (no chunk attaches without another request).

## 9. Not in this phase

Graphs for the fused verify (§3, with P2's descriptors); fusing the megakernels' FFN and DeltaNet mixer with a chunk
(the decode rows would regain the megakernels' ~5-8 ms); image prompts in a fused pass; three or four decode sessions
in one fused verify (P2's row budget).
