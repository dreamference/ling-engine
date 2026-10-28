# ling-engine: Batching several requests

Part of the ling-engine specification; the index is [README.md](./README.md). Like the CLI and API documents it has no number from the single-file `SPEC.md`; it extends [§11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md) ("Batches") and [BACKLOG §19](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) item 31 (batches of three and four), and it builds on M4's sessions and batched verify ([drafts/M4](./drafts/M4-sessions-and-batches-2026-10-10.md)) and the prefill beside decode ([drafts/M4-prefill-beside-decode](./drafts/M4-prefill-beside-decode-2026-10-17.md)).

Written 2026-10-27 from the owner's question of 2026-10-26, "batching several requests well: how are we doing on that?", and his decision of 2026-10-27 that the engine becomes Mightling's default (1.8.0) only once it serves four or more concurrent conversations well. First where `ling-serve` main (`c5ec4ec`) stands against SGLang at one to eight concurrent agent streams, measured on node-b; then where a step's time and a session's memory go; then the target, the design and the phases, each phase exact. Phase 0 (admission control on host memory) is merged; P1 is built on a branch and parked (§6); the rest of §5 is design. After the measurements the owner set the order (2026-10-27): concurrency first, each conversation moving while the others prefill (C1, C2), then the wider batch.

## 1. The question and the scope

Mightling runs several conversations at once: SWE-bench at two to four instances, Night Shift's best-of-3, refine's study and fix, Ask and Work side by side, subagents. Every one of them is an *agent* stream: a long prompt (28-62K tokens on the agentlong set), then turns of tool output (a few hundred new tokens each) and short answers, many of them tool calls. So "batching well" is not only the decode step at N sequences; it is what each conversation sees when the others are prefilling, decoding and waiting, and whether the machine survives N of them.

In scope: `ling-serve` with N = 1-8 concurrent agent streams, cold (every turn a new session) and warm (follow-ups on a resident session, `prompt_cache_key`), greedy and seeded sampling; the batched step's cost; memory per session; prefill/decode interference; safety. Out of scope: two-node tensor parallel ([drafts/two-node-tp](./drafts/two-node-tp-2026-10-12.md)), FP8 KV as a default (non-exact, §6 P5), more than one model.

## 2. Measured

All on node-b (GB10, 121 GB), `ling-serve` `c5ec4ec` with the ScaleSearch drafter, `--draft-block 12 --graphs --pdl`, 65,536-token context, m0-prod8000 stopped, every server beside `mem_guard.sh` from 2026-10-27 09:00. SGLang is v0.5.21 (production's phase-merge image) at `--mem-fraction-static 0.5`, 12 draft tokens. The tools are in `bench/agentlong/`: `n_streams.py` (N concurrent agentlong streams, `--compare A B` = greedy identity turn for turn), `streams_server.py RUN.jsonl SERVERLOG` (server-side timing: decode intervals, steps batched, steps beside a prefill, TTFT), `streams_window.sh` (a GPU window: the lock, m0 stopped and restarted, the guard, at most four sessions), `batch_profile.sh` (the step in place and under nsys).

The client's own first-delta timing is not usable: a tool-call answer streams nothing until its end, so it reads as a long TTFT and a short, fast decode (that artefact also gave the "two streams at 28.6 tokens/s" that started this question). Every number below is server-side.

### 2.1 Cold agent turns (16 turns, 520-555K tokens of new prefill, 5,275 output tokens greedy)

| Run | Wall s | End-to-end tok/s | Per-stream decode tok/s | Aggregate decode tok/s (any decoding) | ms/pass | Steps batched | Steps beside a prefill | TTFT median/max s |
|---|---|---|---|---|---|---|---|---|
| engine, 1 stream | 378.1 | 13.95 | 64.0 | 64.0 | 105.3 | 0 | 0 | 16.7 / 31.6 |
| engine, 2 streams | 379.6 | 13.90 | 30.4 | 32.1 | 217 | 18% | 70% (253 ms) | 31.9 / 55.6 |
| engine, 3 streams, 3 sessions | 377.2 | 13.98 | 26.5 | 35.0 | 241 | 48% | 69% (271 ms) | 46.5 / 75.5 |
| engine, 4 streams, 2 sessions (queued) | 375.2 | 14.06 | 30.7 | 33.9 | 212 | 31% | 63% | 35.1 / 54.7 |
| engine, 4 streams, 4 sessions | 380.5 | 13.86 | 24.5 | 30.5 | 261 | 31% | 73% (288 ms) | 62.5 / 122.7 |
| SGLang, 1 stream | 442.0 | 11.57 | 51.4 | 51.4 | 110.4 | | | 20.4 / 36.6 |
| SGLang, 2 streams | 432.8 | 11.92 | 20.4, 11.9 | 16.5 | 401 | | | 23.2 / 74.7 |
| SGLang, 4 streams | 420.8 | 9.60 | 4.1-27.4 | 13.2 | 600 | | | 26.8 / 61.2 |
| SGLang, 8 streams | 416.1 | 11.76 | 0.9-52.1 | 12.5 | 1,527 | | | 27.4 / 138.0 |

(SGLang W8, the same bodies byte for byte; its output was 5,113 / 5,159 / 4,039 / 4,895 tokens, since its greedy output changes with the concurrency, §2.3.) On cold turns the engine takes 375-381 s where SGLang takes 416-442, and a stream decoding beside another's prefill keeps 24-31 tokens/s on the engine against 4-27 on SGLang, whose decode steps wait behind its prefill chunks (401-1,527 ms a step).

Cold agent turns are prefill-bound: about 290 s of the 378 s is prefill at 1.8-2.2K tokens/s, so more streams add no throughput and only spread the waiting. What they cost each stream is decode speed: 63-73% of the decode steps at two to four streams run beside another request's prefill on the side stream (a green context of 40 of 48 SMs) and take 250-290 ms instead of ~105. That is the real-world loss, not the batched step (§3.1).

Seeded sampling (seed 1013), end to end: engine 2/3/4 streams 8.35 / 9.0 / 8.88 tokens/s (4 streams on 2 sessions 8.23; the one-stream run was invalid, started while another server was still releasing memory, and its rerun was dropped for the concurrency work); SGLang 1/2/4/8 streams 8.56 / 11.72 / 12.02 / 6.46 at walls of 429 / 423 / 399 / 403 s. Sampled lengths differ between runs (SGLang's output was 3,676 / 4,953 / 4,792 / 2,605 tokens), so these compare walls, not decode.

### 2.2 Warm follow-ups

`n_streams.py --warm` with `prompt_cache_key` (W11, 2026-10-27): in rounds, every stream first sends its turn's prompt without the last tool call and its output (max 1 output token, not counted), all wait, then all send their full turns at once. The same 16 turns and 5,275 output tokens; the measure is the rounds' wall (the warm-up requests excluded). The appended part is not small on this set: 112-125K new tokens over 16 turns, ~7K a turn (up to ~30K), so even warm turns carry real prefill.

| Run | Rounds s | Output tok/s over the rounds | Per-stream decode tok/s | Aggregate while all N decode (s of it) | Aggregate while any decodes | ms/pass | Tokens/pass | Steps batched | Steps beside a prefill (ms) | Prefill s (sum) | TTFT median/max s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| engine, 1 stream | 147.7 | 35.7 | 66.1 | 66.1 | 66.1 | 101.8 | 6.73 | 0 | 0 | 67.1 | 0.42 / 18.2 |
| engine, 2 streams | 150.4 | 35.1 | 40.5, 51.1 | 106.5 (5.3 s) | 46.3 | 149.8 | 6.63 | 11% | 28% (256) | 96.2 | 0.77 / 29.3 |
| engine, 3 streams | 146.7 | 36.0 | 40.9, 26.7, 53.9 | 117.3 (5.0 s) | 51.1 | 159.1 | 6.48 | 38% | 29% (230) | 122.7 | 1.0 / 27.9 |
| engine, 4 streams | 141.0 | 37.4 | 23.4, 30.9, 20.3, 43.0 | 66.8 (2.3 s) | 41.2 | 234.4 | 6.22 | 62% | 52% (313) | 196.0 | 3.85 / 55.5 |
| SGLang, 1 stream | 441.8 | 12.0 | 54.3 | 54.3 | 54.3 | 110.5 | 6.01 | | | | 20.5 / 36.7 |
| SGLang, 2 streams | 422.8 | 12.7 | 45.1, 16.6 | 78.3 (31.0 s) | 29.2 | 235 | 5.86 | | | | 28.4 / 69.6 |
| SGLang, 4 streams | 394.2 | 11.7 | 23.3, 11.2, 6.5, 26.0 | 138.9 (6.8 s) | 24.1 | 430 | 5.95 | | | | 31.3 / 130.5 |
| SGLang, 8 streams | 387.9 | 12.9 | 4.1-53.4 | (never all eight) | 18.5 | 720 | 5.93 | | | | 43.1 / 197.9 |

(W7, the same bodies; SGLang's request log gives no prefill/decode overlap, so the beside-a-prefill columns are empty.) SGLang does not keep the conversations warm: it prefilled 531-545K new tokens in every run, as many as the cold turns (its cache served 108-142K tokens in all), so its "warm" rounds are cold rounds and take 388-442 s against the engine's 141-150 s. On the workload Mightling actually sends, the engine's resident sessions are worth about 3x in wall time at every concurrency, before any batching. While all four of SGLang's streams decode it reaches 139 tokens/s (M0's ~135 again), but for 6.8 s of 394.

So concurrency buys the engine nothing on warm agent turns either: the rounds take the same ~145 s at one to four streams. While two or three sequences decode together the batch does what it should (106-117 tokens/s, 1.6-1.8x one stream), but that is 5 s of a 150 s run; the rest is decode beside a prefill (256-313 ms a step, 2.5-3x) and prefill slowed by the decodes beside it (the summed prefill time 67 s at one stream, 196 s at four). At four even a step with no prefill beside it takes 144 ms, because only two sequences fit a pass and four take turns in pairs.

**C1 (scheduling, merged 2026-10-27).** The shortest prefill goes first: of the requests waiting to prefill, the one with the fewest tokens left takes the next chunk. A long prefill on the side stream no longer relaunches its next chunk while a shorter one waits, and a request that has yielded for 5 s takes its chunk anyway. Before this, a warm follow-up that arrived during a cold prompt waited out the whole prompt.

- *Probe* (`probe.py`, `--sessions 4 --batch 4 --session-protect-seconds 0`, 3 repeats). A decodes 1,500 tokens while B sends a cold 39K prime and C, 5 s later, a warm append:
  - C's TTFT: 20.5 s on main, 1.43 s with C1.
  - B's TTFT: 37.0 s on main, 35.2 s with C1.
  - A's decode: 61.7 tok/s on main, 60.0 with C1 (alone 89.4 / 89.3 ms a step, beside the prime 222 / 229 ms).
- *Warm agentlong, 4 streams*:
  - Rounds: 145.0 s on main, 139.3 s with C1.
  - TTFT median / max: 1.43 / 51.7 s on main, 1.02 / 53.8 s with C1.
  - Summed prefill: 186.5 s on main, 139.0 s with C1.
  - Per-stream decode: 46.6 / 28.1 / 21.0 / 33.7 tok/s on main (pooled 32.2), 22.4 / 23.2 / 21.8 / 49.6 with C1 (pooled 25.7). The prefill now finishes sooner but occupies more of the decode steps: 62% beside a prefill against 43%.
- *Warm agentlong, 2 streams*: 151.4 s with C1, the same as main.
- *Exactness*: greedy 16/16 at 2 and 4 against one stream.

C1 is what makes a follow-up's first token fast. It does not make the decoding conversations faster; that is C2. With the default `--sessions 2 --batch 2`, C waits ~20 s on main and C1 alike, for admission rather than for chunks (§3.5).

**C1's other two knobs (2026-10-27, warm agentlong at four streams, greedy 16/16 against one stream in every run).**

*The side stream's SM count.* Fewer SMs lose on both axes: the prefill takes longer, so more decode steps run beside it.

| SMs | Rounds | Pooled decode | A in the probe |
|---|---|---|---|
| 24 | 168.0 s | 22.0 tok/s | 50.4 tok/s |
| 32 | 150.9 s | 25.5 tok/s | 53.7 tok/s |
| 40 (default, kept) | 139.3 s | 25.7 tok/s | 60.0 tok/s |

*The prefill's share (`--side-share N`, merged with default 16).* While two or more other requests decode, the side stream waits after each chunk (any request's) until N decode passes have run on the main stream alone. A first version kept one mark per request; another request's chunk then took the side stream at once and nothing changed (N = 2 and 4: pooled 26.9 and 27.1).

| N | Rounds s | Pooled decode tok/s | Per-stream tok/s | TTFT median / max s |
|---|---|---|---|---|
| 0 | 139.3 | 25.7 | 22.4 / 23.2 / 21.8 / 49.6 | 1.02 / 53.8 |
| 4 | 142.6 | 28.9 | 23.4 / 26.7 / 28.7 / 54.0 | 1.03 / 55.3 |
| 8 (two runs) | 144.4, 142.7 | 32.3, 31.8 | 23.9 / 36.5 / 34.6 / 58.8 | 1.08 / 53.2 |
| 12 | 139.4 | 34.9 | 28.2 / 39.3 / 33.8 / 53.8 | 1.04 / 52.7 |
| **16** | 140.9 | 35.6 | 27.3 / 44.2 / 36.9 / 53.9 | 1.17 / 52.4 |
| 24 | 140.5 | 34.8 | 27.1 / 38.0 / 38.5 / 52.3 | 1.82 / 55.4 |
| 32 | 141.1 | 37.1 | 28.9 / 40.3 / 42.4 / 53.8 | 2.12 / 53.5 |

16 is the knee: decode up 39% at the same round time, before the follow-ups' median TTFT starts to rise. The probe with two conversations decoding while B primes (`probe.py --decoders 2`) measures the case the hold is for:
- A and A2 decode at 70.1 / 77.5 tok/s with N = 16, against 42.8 / 49.5 with N = 0.
- B's cold prime is unchanged (38.7 s against 38.3): the decodes finish sooner and free the GPU.
- C's TTFT is 1.60 s against 1.28.
With one conversation decoding (the plain probe) the hold is off and nothing moves: C 1.36 s, B 35.5 s, A 61.4 tok/s.

At four conversations the 1.8.0 bar's per-conversation floor (34.6 tok/s) is now met by the pooled rate but not yet by the slowest conversation (27-29 tok/s). That is C2's job.

### 2.3 Exactness

Engine greedy: 16/16 turns identical to one stream at 2, 3 and 4 streams cold and at 2 and 4 warm and, with `prompt_cache_key`, at 2, 3 and 4 warm (W11; its one-stream run is also 16/16 identical to the cold one-stream run). Seeded sampling: 12-15/16 identical. That is expected and exact: a batched step verifies the 12-row chain while a one-row step verifies the 16-row tree, so the accepted lengths and with them the random draws differ; each draw is still from the exact distribution (SPECULATION §10). SGLang sampled 0-1/16 (its speculative draws are unseeded: not a correctness signal); SGLang greedy is not batch-invariant: cold, 3/16, 2/16 and 2/16 turns identical to its one-stream run at 2, 4 and 8 streams, warm 2/16, 5/16 and 5/16 (and 0/16 identical to the engine's: different kernels, a different model of exactness).

### 2.4 The batched step in place

`ling-run` on a raw-text pair of ~24K and ~31K tokens, 1,000 tokens greedy (W5):

| Step | ms | Tokens per step |
|---|---|---|
| one stream, chain (12 rows) | 99.1-99.5 | 3.53 |
| one stream, tree (16 rows) | 100.4-100.6 | 4.31 |
| batch of two, chain (2 x 12 rows) | 121.7-121.8 per batched step | 3.53 each |

The second sequence costs +22.5 ms at ~27K: the pair's aggregate is 1.63x one stream's chain and 1.35x its tree. Under nsys (150 batched and 100 single steps, `runs/w5-profile/prof/*.breakdown.txt`), the batched step minus the single one: attention +9.8 ms (10.8 -> 20.6); the DeltaNet path +8.6 (a batched verify leaves the gdn megakernel: its FP8 GEMMs 32.5 ms and recurrence and conv 13.0 against the megakernel's 36.9); the FFN at 24 rows +4.6 (`ffn_persistent<4>` 46.1 against `<2>` 41.5); the drafter and the NVFP4 head +2. The long in-place runs (124K) produced no output and are not used.

### 2.5 Memory

MemAvailable before and after the start (prefix store off): one session at 64K 31.5 GB, two 35.7, four 44.9; two at 128K 44.7, two at 256K 61.3; two at 64K with a 4 GB prefix store 40.0. So a session is ~4.2-4.6 GB at 64K, ~8.9 at 128K, ~17.3 at 256K: a fixed ~0.48 GB (the DeltaNet state 151 MB, conv, the end-of-prompt snapshot, the drafter's ring 168 MB) plus 64 KB per token of BF16 KV, allocated flat at `--max-context` whether used or not. SGLang at 0.5: an FP8 KV pool of 253,201 tokens (7.7 GB, 32 KB a token), a mamba cache of 96 slots (BF16 SSM state 6.8 GB plus intermediates 7.6 GB), the drafter's BF16 KV 4.8 GB, `max_running_requests` 8. Host memory drifts ~1 GB during an engine run, ~3 GB during SGLang's.

After the start the engine grows little: sampled every second during W11 (`~/batching/memtrace`), MemAvailable fell 0.7 GB below its value at "serving" at one session, 4.0 GB at two and 3.7 GB at four (warm agent turns, 64K) (the host as a whole, so the client and page cache are in it too). It does not grow with the number of sessions up to four. Every large device buffer (KV, state, checkpoints, the snapshot slots, the drafter's rings, activations) is allocated at start; only the stream GEMMs' split buffer (at most ~0.25 GB) and the CUDA graphs come later.

### 2.6 The freeze of 2026-10-27

A W6 item, `ling-serve --sessions 8 --batch 8` at 65,536 tokens with a 4 GB prefix store, reported itself serving at 08:41:33 with 41.7 GB of MemAvailable left; eight cold primes (28-62K tokens) arrived together; no prime completed, the journal stops at 08:42:11 with no OOM line, and the host rebooted at about 08:44. The start was within budget: eight 64K sessions plus weights and scratch plan to ~61 GB against ~95 GB free, and admission control's start check (§5.6) passes it. So whatever took the last ~40 GB in under 40 seconds was allocated after the start: memory the start-time plan does not see. The data from that run cannot say what; what was measured since (§2.5) does not explain it: up to four sessions nothing grows by more than 4 GB after the start, and the only allocations made after it are graphs and the split buffer. The suspects left are the graphs of 56 ordered session pairs at eight sessions (captured on first use; untested above two orders' worth) and a fault in the eight-session path itself (`kMaxSeqs` is 4; `--batch 8` is capped to pairs by the row budget, but nothing at eight sessions had run before). A reproduction needs eight sessions under `mem_guard.sh` and the per-second trace, after admission control is merged and with the owner's say. The consequences are already in force: `mem_guard.sh` beside every bench server (Mightling's PSI rule plus a MemAvailable floor), at most four sessions in any run until admission control is merged and the owner raises the cap, and §5.2's rule that every allocation a batch needs is made, and counted, at the start.

A second suspect was found after the freeze: the tokenizer's BPE cache (an `unordered_map`) was shared by the HTTP threads without a lock. `ling-tokenizer-threads-tests` crashed in 2 of 3 runs before the fix and passes after it, and four streams crashed ling-serve with SIGSEGV in one C1 window. Eight large prompts tokenized at once could corrupt the map and allocate without bound on the host. The fix is merged (`92495ef`). The reproduction on a second engine machine (node-c, an otherwise empty box) runs at the commits before and after it, with the ling-serve process's resident memory in the per-0.5 s trace.

## 3. Analysis

### 3.1 Where a batched step's time goes

A step model that fits W5 and the nsys breakdown: step(N, L) ~= 99 + (N - 1) x c(L) ms, where c(L) ~= 0.37 ms per 1K tokens of the sequence's context (attention) + ~9 (the DeltaNet path outside the megakernel) + 3-5 (more rows through the stream GEMMs and the FFN); c(27K) ~= 22.5 measured. At N = 4 and 32K that is ~170 ms; at the chain's agentlong ~6.2 tokens per step that is ~145 tokens/s aggregate greedy, against SGLang's ~135 while four decode (M0, 148 ms step) and OVERVIEW §7's target of >= 175. Two changes take it there: the DeltaNet path of a batch in one persistent launch (the single stream's megakernel generalised to N, about -6 ms per added sequence) and the tree inside the batch (+8-9% tokens per step on agentlong, BACKLOG 26; the W5 raw-text pair shows +22%, an upper end), ~185 tokens/s.

### 3.2 Why a batch loses the tree

`fit = kMaxStreamRows / draft_block` = 32 / 12 = 2, so a batch of two verifies two 12-row chains and a one-row step verifies the tree (16 rows at `c5ec4ec`, 17-24 on main since tree/rows). Three or four sequences do not fit one pass at all today; they take turns in pairs. The stream GEMMs hold 230-240 GB/s only up to about 32 rows (item 31), so going wider needs a 33-64-row tier that is row-invariant (each row's result independent of the others in the tile) to keep the outputs bit for bit.

### 3.3 Prefill interference dominates the real workload

Cold, 63-73% of decode steps run beside a prefill at 2.5-2.9x their alone time; warm with keys 28-52% at 2.3-3x (§2.2), and the batch proper (all N decoding, no prefill) is 2-5 s of a 150 s run. Batching more sequences into a decode step helps only the steps that are not beside a prefill. The arithmetic of a warm round says where the time is: at one stream it is ~53 s of prefill (112K new tokens at ~2.1K tokens/s, compute-bound) and ~80 s of decode (5,275 tokens at 66 tokens/s, bound by the weight read) in series, ~145 s. The two use different resources, so a round could approach the larger of the two, not their sum; today, at two to four streams, running them together on two streams costs each about what it saves the other, and the round stays at ~145 s. The prefill-beside-decode work of 2026-10-17 ([drafts](./drafts/M4-prefill-beside-decode-2026-10-17.md) §4) already swept the knobs of that arrangement (contended chunks of 256-2,048 tokens, a side stream on 24-48 SMs): each trades the other stream's step (137-997 ms) against the prefill's time (18-42 s for 35K) along one curve. Moving along the curve is not the lever; reading each weight tile once for both (§5.5) is.

### 3.4 Memory

Flat slots at `--max-context` make resident sessions expensive: Mightling 1.7.2's recipe (no `--sessions`, so the default two, at 262,144 tokens) holds 2 x 17.3 GB of KV whether the conversations are 30K or 250K long, ~61 GB with the weights. Eight resident sessions at 64K fit (~61 GB); eight at the model's full context do not (~160 GB). Paged KV (§5.3) makes a session cost what it holds.

### 3.5 Four sessions at what context: the decision for the owner

**Decided 2026-10-27:** the full 262,144-token context stays, paid for by P4 (paged KV, about two weeks; §5.3); the recipe is unchanged until P4 lands, then passes `--sessions 4 --batch 4` or more. What follows is the analysis the decision was made on.

ling-serve ran `--sessions 2 --batch 2` unless told otherwise (`src/serve/main.cpp`; since 2026-10-28 it sizes itself, §5.6), and Mightling's recipe passes neither, so a third conversation waits for a whole request to finish (the probe at the default: C's first token after ~20 s, all of it spent waiting to be admitted). Four conversations that each move need `--sessions 4 --batch 4` in the recipe. With flat slots, that costs memory in proportion to `--max-context`. There are three ways to pay for it: a shorter context, paged KV (P4), or both.

**(a) Memory at four sessions (flat slots).** The figures are ling-serve's own start-time plan (§5.6; GB = 10^9 bytes): weights 23.5, scratch 4.3, and per session ~0.5 GB fixed (DeltaNet and conv state and their end-of-prompt copy, the drafter's ring 168 MB, the token history) plus 64 KB per token of BF16 KV at `--max-context`. Admission control keeps 16,384 MiB (17.2 GB) free on top of the plan. On node-b, with nothing else on the GPU (the production server stopped), MemAvailable at the start was 109,423 MiB = 114.7 GB.

| `--max-context` | Per session | Four sessions' slots | Plan (weights + scratch + 4 sessions) | Plan + reserve | Start on node-b (114.7 GB) |
|---|---|---|---|---|---|
| 65,536 | 4.8 GB | 19.2 GB | 47.0 GB (logged: 46.9) | 64.2 GB | passes, 50 GB spare |
| 131,072 | 9.1 GB | 36.4 GB | 64.2 GB | 81.4 GB | passes, 33 GB spare |
| 196,608 | 13.4 GB | 53.5 GB | 81.4 GB | 98.6 GB | passes, 16 GB spare |
| 262,144 | 17.7 GB | 70.7 GB | 98.5 GB | 115.7 GB | **refused** (1 GB short) |
| 262,144 x 2 (1.7.2 today) | 17.7 GB | 35.4 GB (two) | 63.2 GB (60 measured) | 80.4 GB | passes |

What else comes out of the same 114.7 GB on a user's node is not in this table: the sidecars `server start` brings up (search, image search, voice, `ling web`), the desktop, and the prefix KV store (it sizes itself from what is left, 4 GB by default). Mightling's pre-load check (`start_server`) reserves the recipe's 0.50 arena, ~61 GB. That covers four sessions only at 64K. At 128K and above, the recipe's arena figure has to rise to the plan, or the check guards less than the engine takes. On the product side, `--max-context` is what the launcher reads from `/v1/models` as the model's window, so Codex compacts a conversation near it. 64K means compaction on long agent sessions; agentlong's prompts already reach 62.3K. 128K is a middle that fits with 33 GB spare.

A middle option that has not been built: slot classes, e.g. two sessions at 262,144 and two at 65,536 (plan 72.8 GB). A conversation that outgrows a small slot moves to a free large one by a device copy, which is exact and takes ~20 ms for 64K tokens, or waits for one. That is less work than P4 but still a fixed partition.

**(b) P4, paged KV (§5.3): effort, risk, what it buys.**
- *Effort.* About two weeks of focused work, the size of M4's batching stages. The KV is addressed in five places: the attention kernels (`attention_body.cuh`: prefill, and verify/decode split along the context), the KV writes (`kernels.cu`, and the stream path's fused write in `stream_gemm.cu`), the engine's allocation, checkpoints and prefix restores (`engine.cpp`, `speculate.cpp`, `prefix_kv.cpp`), and admission's plan (one pool instead of N slots). New code: a page pool and allocator, a device page table per session, copy-on-write prefix pages merged with the prefix KV store's radix arena, and a policy for a full pool (answer 503, move an idle session's pages to the prefix store, or preempt). The attention kernels are the only GPU code whose addressing changes. The drafter's ring and the DeltaNet state stay as fixed slots.
- *Risk to exactness: low and checkable.* With 64-token pages, a multiple of the attention kernels' KV tile, each tile loads the same values and reduces them in the same order as today, and split-K partitions by logical position, so the bits do not change. The gates are a memcmp of paged against flat per kernel plus greedy 16/16 at 1, 2, 4 and 8 streams. The places it could slip, each tested on purpose: a vectorised load crossing a page boundary, the partly filled last page under copy-on-write, and a prefix restore that copies a slot rather than mapping pages. Speed risk: one page-table read per 64-token tile in decode attention, which is ~10.8 ms of a ~100 ms step at 27K, so under 1%.
- *What it buys.* A session costs ~0.5 GB plus 64 KB for each token it actually holds, so memory grows with the conversations' real lengths, as on SGLang, and `--max-context` stays 262,144 for everyone. On node-b the pool would be ~114.7 - 27.8 (weights, scratch) - 17.2 (reserve) - 2 (four fixed parts) = ~67.7 GB, ~1.03M BF16 tokens. That holds four conversations near full context (4 x 262K = 1.05M, just at the edge), or eight averaging ~128K, against SGLang's 253K-token pool. With P5's FP8 KV (a non-exact switch) it would be ~2M tokens. A prefix-store hit then maps pages instead of copying them. P4 does not remove §2.6's suspects (graphs captured after the start), which §5.2 handles.

**(c) What SGLang's recipe gives at four conversations.** Production's SGLang (v0.5.21, memory fraction 0.50, `max_running_requests` 8) holds one shared FP8 KV pool of 253,201 tokens (7.7 GB, 32 KB a token) and 96 mamba slots. A request takes only the tokens it holds, so four conversations are resident together only while their contexts sum to ≤ 253K: four of ~63K, or two of ~126K. A single 262K conversation does not fit at all. Beyond that SGLang evicts its radix cache and re-prefills, or retracts a running request. On agentlong at four streams it kept nothing warm anyway. It re-prefilled 531-545K new tokens per run, and its warm rounds took 394 s against the engine's 141-145 s (§2.2). Per-stream decode was 23.3 / 11.2 / 6.5 / 26.0 tok/s, with 139 tok/s while all four decoded (6.8 s of 394), and TTFT median 31.3 s, max 130.5 s. Cold (§2.1): 420.8 s, 4.1-27.4 tok/s per stream. Its greedy output changes with concurrency (§2.3). So the engine at `--sessions 4 --batch 4 --max-context 131072` would hold more context per conversation than SGLang can hold for four at once, in BF16, and keep it warm.

## 4. The target: four or more conversations, well

The engine is the 1.8.0 default when, on node-b, with the agentlong set:

1. **Every conversation moves (the owner's bar of 2026-10-27).** Warm agentlong (`n_streams.py --warm --cache-key`) at one, two and four streams: the round's aggregate output tokens/s rises with the streams (today 35.7 / 35.1 / 37.4: flat), and every conversation's decode stays at or above production's two-stream 34.6 tokens/s (item 31's floor; today 20-43 at four). Reported at every phase as rounds s, aggregate tokens/s and per-stream decode tokens/s, against W11 and SGLang's W7 (whose rounds take 388-442 s on the same set).
2. **Time to first token while another stream prefills.** A probe: stream A decodes while B sends a cold 30-60K prime and C a warm append of 0.3-2K tokens; reported as A's decode tokens/s during B's prefill, B's TTFT and C's TTFT (target: C's within 2x its value alone, A's decode >= 34.6).
3. **Throughput when all decode.** Aggregate decode while four decode >= SGLang's at four (139 tokens/s, W7) and, as the longer-term target, >= 175 (OVERVIEW §7).
4. **Exactness.** Greedy output per sequence identical to that sequence alone, turn for turn, at 2, 3, 4 (and 8 once P4 lands) streams; seeded sampling exact in distribution, identical whenever the batch shape is equal.
5. **Safety.** Eight resident sessions start and run with no host-memory growth after the start beyond the plan's margin; a request the host cannot hold is refused (503), never allowed to take the host down.

## 5. Design

### 5.1 The row budget

`kMaxStreamRows` 32 -> 64, with the per-row arrays it sizes (the tree's `depth`/`var`/`path`, `gdn_step`, `kMkFfnWords`, `kMkGdnWords`) and a `stream_gemm` tier for 33-64 rows that is row-invariant; `kMaxSeqs` stays 4 (8 only with P4). The scheduler's `fit` becomes a per-sequence row share by batch size: two sequences x up to 24 tree rows (48), three or four x 16 rows (48-64: the tree at its 16-row shape, or the chain at 12 when a sequence's acceptance says the tree does not pay). The share is a fixed function of the batch size, not of the other sequences' acceptance, so a seeded sequence's draws are identical whenever the batch shape is.

### 5.2 N sequences in one pass

The batch path today (M4 stage 4) runs the drafter over every block at once and the state kernels in one launch each (`kernels::SeqBatch`), and captures its graphs per *ordered tuple of sessions* (`batch_graphs_`): eight sessions are 56 ordered pairs, each with draft, verify and per-count commit graphs, captured lazily on first use. That becomes graphs per *batch size*, reading each slot's session (its KV and state pointers, positions, row offsets) from a device descriptor table written before the replay, so the set of graphs is bounded (sizes 1-4) and **captured at start, inside the memory plan**: no allocation after the start depends on which sessions meet. The DeltaNet path of a batch moves into the persistent gdn kernel generalised to N sequences (each sequence's rows its own recurrence), recovering most of §2.4's +8.6 ms per sequence.

### 5.3 Paged KV (P4)

The owner's decision of 2026-10-27 (§3.5): four or more conversations keep the model's full 262,144-token context, paid for by paged KV, not by a shorter context. Mightling's recipe stays as it is (no `--sessions`, no `--batch`) until P4 is on by default; then it passes `--sessions 4 --batch 4` or more.

**What changes and what does not.** Only where the full-attention layers' KV lives. Today each session owns a flat slot of `--max-context` positions per layer, head-major `[Hkv][max_context][D]` (`Engine` constructor, `kernels::set_kv_layout`), allocated at start whether it is used or not. Under P4 the KV lives in one pool of 64-position pages and each session maps the positions it holds through a page table. The values written and read, the tiles, the key ranges of the split (`attention_chunk()`, 4,096 keys at fixed positions), the order of every reduction and the combine are unchanged, so every output bit is unchanged. These stay one fixed slot per session, outside the pool: the DeltaNet and conv state and their end-of-prompt copy (`snap`), the checkpoint store's states, the drafter's context ring (168 MB, a ring of 8,224 positions) and the token history. A session then costs ~0.5 GB plus 64 KB for each position it actually holds.

**The page.** 64 positions of every full-attention layer's K and V: 16 layers x {K, V} x 4 KV heads x 64 positions x 256 values x BF16 = **4 MiB**, one contiguous block laid out `[layer][K, V][kv head][64][D]`. The pool is one allocation made at start, `[page][layer][K, V][kv head][64][D]`. Why this shape:
- *64 positions*: a multiple of the attention tile (`kAttnTk` = 32 keys; the megakernel's `kMkAttnTk` the same). Tiles start at multiples of 32, because every key range starts at a multiple of 4,096 and walks in 32-key steps, so **no tile crosses a page**. A tile's 32 keys of one head are one contiguous 16 KB run, as in the head-major flat layout, so the L2 prefetch of a whole tile (`attn_prefetch_tile`, one bulk prefetch for K and one for V) and the 512-byte bulk copies (`attn_bulk_tile`) work unchanged. 64 rather than 32 halves the table and the page-boundary cases; 64 rather than 128 or more keeps a copy-on-write copy small (4 MiB) and wastes at most 63 positions in a session's last, partly filled page.
- *All 16 layers in one page*: one page id serves every layer, so a session has **one** table, not sixteen, and allocation, sharing and copy-on-write are one operation per 64 positions (a copy-on-write is one 4 MiB device copy, ~20 µs).

A table entry is the page's element offset into the pool (64-bit), so a kernel needs one more pointer and no stride. A layer's K base is `pool + layer x 2 x 4 x 64 x 256` elements and its V base 4 x 64 x 256 further; position `p` of KV head `h` is at `base + table[p >> 6] + h x (64 x 256) + (p & 63) x 256`. Inside a page the head stride is 64 x 256 and the position stride 256, which `set_kv_layout` already parameterises; the flat layout is the same expression with no table.

**The table.** One device array per session of `max_context / 64` entries (4,096 at 262,144: 32 KB), allocated at start, with a host mirror. Its pointer is fixed for the engine's life, so a captured graph that bakes it stays valid while the pages behind it change. The host writes the entries it changed into the device table, asynchronously on the stream of the work that will read them, before that work is launched. Entries past a session's last page point at a reserved page 0 that is never handed out, so an address formed for a masked key (`attn_load_tile` forms one even for an invalid key with a zero-byte copy; a captured graph launches the ranges of the whole context) is always inside the pool. With §5.2's descriptor table, a batch slot's table pointer becomes one more field read at replay, so graphs per batch size need nothing more from P4.

**Every place that addresses the KV** (each takes the table; a null table is the flat layout, so one build serves both):
1. the tile loaders in `attention_body.cuh`, shared by `attention_rows`, `attention_rows_seqs` and the megakernel's attention layer (`mk_attn_range`, LING_MK_ATTN): `attn_load_tile` and `attn_bulk_tile` (one table read per tile, since a tile is inside one page), `attn_prefetch_tile` (unchanged run, base from the table), and `attn_load_tile_tree`, whose keys from `pos0` on come from `pos0 + path[j]` and may lie in another page (one table read per key there);
2. the KV writes: `attn_prepare_body` (decode, verify, prefill, tree slots `pos0 + m`), `attn_prepare_seqs`, and the megakernel's prepare items;
3. `tree_gather`, which moves an accepted tree path's KV from slot `pos + row` to `pos + depth`: today a `GatherBuf` per layer and head with a flat base and a row stride, baked at start; under P4 a KV `GatherBuf` carries the table and computes each row's address from it (source and destination can be in different pages);
4. the copies: `Engine::adopt_prefix` (a session's prefix from another session, `cudaMemcpy2DAsync` with the flat pitch), `Engine::kv_copy` (to and from the prefix KV store's buffers) and the store's `Ops`;
5. the memory plan (`serve/admission.hpp` `session_bytes`, `plan_memory`) and the engine's allocation.
The fused pass (C2, `forward_fused`) launches the same kernels and inherits the table from the session it serves; P4 is rebased onto C2 when C2 merges.

**Allocation and release.** A host class `KvPool` (pages, a free list, a reference count per page) and a `PageTable` per session, host bookkeeping only and tested on the host as `PrefixKvTree` is (`tests/kv_pool_tests.cpp`). Before any forward that writes positions `[from, to)` of a session (a prefill chunk, a verify's rows, a fused pass's chunk), `ensure(session, from, to)` maps a page for every index up to `(to - 1) >> 6` and makes every page it will write private (copy-on-write, below). When a session's history is truncated to `n` (a new request that diverges from it, a rejected tail), the pages wholly past `n` are released. A released page is not handed out again until the work that could still read it has completed: frees are deferred to the next step boundary (after `side_join` and `join_commit`), so the side prefill's stream and the commit's stream never read a page that another session already rewrites.

**Sharing and copy-on-write.** A full page holds KV that is a function of the token prefix only, so two holders of the same prefix can map the same physical page. (1) *Sessions*: `adopt_prefix(from, n)` maps `from`'s pages for `[0, n)` instead of copying them, so a prompt that starts like another session's history (the system prompt and AGENTS.md of every conversation; Night Shift's best-of-3; refine's study and fix) holds that prefix once. (2) *The prefix KV store* (BACKLOG 35): a node holds references to the pages that overlap its run instead of a copy in its arena; a restore maps, for each page index below `n`, the page held by the deepest node on the matched path that covers it (that node's page was written by a prompt that matched the path through its end, so it is correct for every position below `n` that lies in it). Its DeltaNet and conv states stay in the store's arena, whose budget (`--prefix-kv-mb`) then holds states only, and its pages are the pool's first reclaimable memory. The rule that keeps both exact: **any write to a page whose reference count is above one first copies the page** (one 4 MiB device copy on the writer's stream, before the write is launched) and repoints the writer's table entry. In practice that is the partly filled page at a shared prefix's end, once per adoption or store insert.

**The pool's size.** From the admission plan (§5.6), at start, before anything else is allocated: pool = MemAvailable - (weights + scratch + sessions x the fixed part + the store's state arena) - `--host-reserve-mb`, capped at `sessions x max_context` positions (more could never be used), or `--kv-pool-mb` when given (refused at start if the plan with it does not fit). The start line logs it in GB, pages and positions. On node-b (§3.5's figures: 114.7 GB available, weights and scratch 27.8, four fixed parts 2.0, reserve 17.2; the store's states 1 GB) that is ~66.7 GB, **~15,900 pages, ~1.02M positions**: four conversations averaging 254K, or eight averaging 127K, all at `--max-context 262,144`, in BF16. Four conversations at the full 262,144 at once (1,048,576 positions) are ~3% over it, so the pool can fill; the next paragraph is what happens then.

**A full pool.** In order:
1. *Admission.* A request's prompt pages (what it does not already map from its session or the store) are taken whole when it is admitted, plus two pages of growth, so a prefill never stops half way for memory. A request that does not fit waits in the queue (C1's shortest-first order can admit a smaller one meanwhile); it is not refused. HTTP 503 stays what §5.6 says, the host's MemAvailable below `--admit-reserve-mb`.
2. *Reclaim, cheapest first.* Pages only the prefix store holds (least recently used leaves first), then idle sessions' pages (a session no request is running on; least recently used first, the `--session-protect-seconds` order, which the pool's need overrides). An evicted idle conversation's next turn prefills from scratch, or from the store: its output is that of a cold turn, which the agentlong gates already require to be identical to a warm one (§2.2: 16/16 against the cold one-stream run).
3. *Decode growth.* A running request needs one more page every 64 positions. If none is free after reclaiming, its step waits while the others run (and may free pages).
4. *The last resort.* If every running request is waiting for a page and nothing is reclaimable, the request admitted last ends as if it had reached the context limit (`finish_reason` `length`), its pages are freed and `ling:kv_pool_exhausted_total` counts it. No output token of any request changes; on node-b this needs four conversations all within ~3% of 262,144 at once.

**Exactness and its tests.** By construction every kernel loads the same values into the same tiles and reduces them in the same order; the tests make sure of it where it could slip (a load that crosses a page, the partly filled last page, a tree path that crosses a page, a masked key's address, a copy-on-write racing a reader, a deferred free):
- `ling-paged-tests` (new): the same random KV laid out flat and in pages behind a shuffled table, compared with memcmp: `attention_rows` at 1, 12, 16, 24 and 32 rows and contexts 1, 63, 64, 65, 4,095, 4,096, 4,097, 27,000 and 200,001 positions; cp.async and bulk tiles; L2 prefetch 0 and 2; eager and graph (`fixed_ctx`); the tree verify with paths crossing a page; `attention_rows_seqs` with 2-4 sequences, each with its own table; the megakernel's attention layer; `attn_prepare`'s writes and `tree_gather`, read back position by position;
- `tests/kv_pool_tests.cpp` (host): allocation, references, copy-on-write, deferred frees, reclaim order, the last resort, the plan's arithmetic;
- the engine's checks with `--kv-pages`: `ling-stream-tests`, `--batch-check`, `--interleave-check`, `--prefix-check`, `--spec-check`, `--graph-check`, and C2's `--fused-check` once merged;
- agentlong greedy 16/16 per turn, paged against flat, at 1, 2 and 4 streams, warm and cold, at `--sessions 4 --max-context 262144`, and once more with a small pool (`--kv-pool-mb 8192`) that forces idle sessions out and copies on write;
- speed: one stream's agentlong tokens/s within noise of flat (expected under 1%: one 8-byte table read per tile per thread, against decode attention's ~10.8 ms of a ~100 ms step at 27K).

**Steps, each behind `--kv-pages` (off; `LING_KV_PAGES`), each merged when its gate passes (exact changes merge under the standing rule; nothing ships before a night shows no loss):**
- **P4a, the addressing.** Every site above takes the table; each session's table maps a fixed run of the pool (the same memory as flat), so only the addressing changes. Gate: `ling-paged-tests`, the engine checks, agentlong 16/16 at 1/2/4, one stream within noise. (~4 days)
- **P4b, the pool.** `KvPool` and lazy mapping, release on truncation with deferred frees, the pool's size from the plan, admission, reclaim of idle sessions, the last resort, `--kv-pool-mb`, gauges `ling:kv_pages_total`, `ling:kv_pages_free`. Gate: host tests; `--sessions 4 --max-context 262144` starts on node-b and grows less than 1 GB after the start; agentlong 16/16 at 1/2/4; the small-pool run 16/16. (~4 days)
- **P4c, sharing.** `adopt_prefix` maps pages; the prefix KV store holds pages, its arena states only; copy-on-write. Gate: `ling-prefix-kv-tests` extended, `--prefix-check`, agentlong 16/16 with prefill tokens no more than today's. (~3 days)
- **P4d, the default.** `--kv-pages` on; then in Mightling the recipe passes `--sessions 4 --batch 4` at 262,144, and `start_server`'s pre-load check counts the engine's plan, not the recipe's 0.50 arena (~61 GB), or it guards less than the engine takes (§3.5). Ships after a night with no loss. Eight sessions only after node-c's eight-session reproduction (§2.6) and the owner's say. (~1 day and a night)

### 5.4 Steps with different acceptance

Sequences in a batch accept different numbers of tokens; there is no lockstep and no padding: each commits its own count (as M4 already does). Cross-sequence allocation of rows by expected gain (more rows to the sequence that accepts more) is greedy-identical and distribution-exact but breaks seeded identity across batch shapes, so it ships only behind a switch, measured against the fixed share.

### 5.5 Prefill and decode

**The fused pass (C2).** A decode step is bound by the weight read (the stream GEMMs read 17.56 GB in ~76 ms of a ~100 ms step); a prefill chunk is bound by compute (~0.45-0.5 ms a token). Per projection, one launch reads each weight tile once and computes both: the decode rows with the stream GEMM's arithmetic, the chunk's rows with the prefill GEMM's, each path's device code reused unchanged, so each output is bit for bit what its own kernel gives (the prefill is already chunk-size-invariant: contended 256-token and full 2,048-token chunks give the same bits). The state-bearing kernels (attention and its KV writes, the DeltaNet conv and recurrence) stay per sequence. The pass then costs about the larger of the read and the chunk's compute plus the state kernels: with a 256-token chunk ~130-150 ms, so each decoding conversation keeps ~45-50 tokens/s (today ~24 beside a prefill) while the prefill runs at ~1.7-2K tokens/s. The chunk size becomes the scheduler's dial between the two. A mixed pass through one GEMM (prefill rows and decode rows in the same tiles) stays rejected: the decode rows' bits would depend on their tile-mates.

**Scheduling (C1, output-identical by construction).** A warm append (<= 2K new tokens) runs ahead of any long cold prime, as its own chunk: it is a follow-up's whole TTFT; queued cold primes go shortest first; while several decode, the prime's share of the time is capped so no conversation stalls. These are policy only and change no output; the side stream's SM split and the contended chunk size stay as measured on 2026-10-17 until C2 replaces them.

### 5.6 Memory budget and admission control (phase 0, merged)

`ling-serve` plans its own footprint before allocating anything (weights, the sessions' slots at `--max-context`, the drafter's rings, 4 GB of scratch) and refuses to start (exit 3, saying by how much) unless `--host-reserve-mb` (default 16,384) of MemAvailable is left; while serving, a generation request arriving with MemAvailable below `--admit-reserve-mb` (default 8,192) is answered HTTP 503 `host_memory_low` with `Retry-After` before anything is queued, and `ling:memory_refusals_total` counts it. Neither changes a number the engine computes. What it does not cover is memory allocated *after* the start (§2.6): §5.2's graphs-at-start and §5.3's pool close that, and until they land the 503 check is the only runtime guard besides Mightling's PSI watchdog.

**Auto-size (merged after phase 0; the owner's decision of 2026-10-28).** The same plan also sizes the server: what the operator does not give of `--sessions` and `--max-context` is chosen at start from MemAvailable (`choose_size` in `src/serve/admission.hpp`, which tests candidates only with `plan_memory` and `refuse_start`, so there is one estimate). The context first: the model's full context (`config.json`'s `max_position_embeddings`, 262,144), halved in multiples of 2,048 down to 32,768 only when one session (or the given sessions) does not fit; then as many sessions as fit at it, at most 4 (`kAutoSessionsCap`, the cap on node-b until the owner approves more), at least 1; `--batch`, not given, follows the chosen sessions. Given values are kept, both given is the check as before, and `--auto-size 0` (`LING_AUTO_SIZE=0`), `--host-reserve-mb 0` or an unreadable MemAvailable give the old defaults (2 sessions, batch 2, 65,536). One start line (`size: …`) names the values, whether each was chosen, given or a default, and the plan's numbers. With §3.5's figures (node-b, 114.7 GB available, nothing else on the GPU) it chooses 3 sessions at 262,144: 80.8 GB planned with images off, 82.4 GB with them (their scratch and cache), plus the 17.2 GB reserve; four are refused. Mightling's recipe passes `--max-context 262144` (its `max_model_len`) and neither `--sessions` nor `--batch`, so after this the default server sizes its sessions itself: three on an otherwise empty 128 GB node, fewer beside the sidecars, the desktop and the prefix store. Placement only: for a given (sessions, batch, context) nothing the engine computes changes, so a server sized automatically answers bit for bit as one started with the same three flags. Until P4 the sessions are still flat slots, so three at 262,144 hold ~53 GB of KV whatever the conversations' lengths; Mightling's pre-load check (the recipe's 0.50 arena, ~61 GB) then guards less than the engine plans and has to follow the plan.

**Requirement on the Mightling side (the engine's plan is the authority).** Before the Mightling engine pin moves past the auto-size merge, `start_server`'s pre-load `check_host_safety()` must size its reservation from the engine's own plan, not from the recipe's 0.50 arena (~61 GB): with no flags the engine plans ~82 GB (3 sessions at 262,144, vision on) and the check would guard 20 GB less than the engine takes. The default technical fix is that `start_server` asks the engine for its plan before the load (a dry run of the start check that prints the `size:` line's numbers and exits, or the same numbers from `choose_size`), and refuses the load when the plan plus the reserve exceeds MemAvailable; the alternative is the recipe passing `--sessions 2`, which makes the plan ~61 GB again but gives up the third session. The PSI watchdog during the load stays as it is.

### 5.7 Exactness checks

- `--batch-check` generalised to N: every batched step re-run alone and compared bit for bit (tokens, state, accepted drafts), as two-sequence batches are checked today;
- agentlong greedy at N streams against one stream, `n_streams.py --compare`, 16/16 turns at every N a phase enables;
- `ling-stream-tests`: memcmp per batched kernel (the 33-64-row GEMM tier, the N-sequence gdn kernel, paged attention against flat).

## 6. Phases

Each phase merges into main once its exactness gate passes and nothing regresses (one stream's agentlong tokens/s within noise of main, every earlier gate still green). Expected gains are from §3.1's model and are measured, not assumed, at each gate.

| Phase | What | Gate | Expected |
|---|---|---|---|
| P0 | Admission control on host memory (§5.6) | the start and 503 refusals; agentlong greedy 16/16 unchanged | safety only (**merged 2026-10-27, `e1eac84`**) |
| C1 | Scheduling: warm appends first, shortest prime first, the prime's share capped (§5.5) | greedy 16/16 at 2 and 4 warm; §4 items 1-2 measured | **merged 2026-10-27**: shortest prefill first with a 5 s yield cap, and `--side-share 16`. C's TTFT 20.5 -> 1.43 s; four warm conversations' pooled decode 25.7 -> 35.6 tok/s at the same round time; the side stream's SM count stays 40 (§2.2) |
| C2 | The fused pass: one weight read for a decode verify and a prefill chunk, each path's arithmetic unchanged (§5.5) | memcmp of each fused projection against its two kernels; `--interleave-check`; greedy 16/16 at 2 and 4 warm and cold | decode beside a prime ~130-150 ms instead of 205-313; warm rounds from ~145 s towards ~80 s at four. Prototype (branches `batching/c2-fused-gemm`, `batching/c2-fp8-prefill`, not merged): every fused projection bit-exact (214 + 336 memcmp cases). An L2 prefetch of each FP8 weight tile two chunks ahead takes the FP8 prefill GEMM at 256 rows from 98 to 160 GB/s, exact, with agentlong 16/16 at one stream. Pass estimate 180-188 ms beside 12-16 decode rows; integration into ling-serve next |
| P2 | Three and four sequences in one pass: the row share by batch size, graphs per batch size captured at start (§5.2), the batched DeltaNet path in the persistent kernel | `--batch-check` at 3 and 4; greedy 16/16; no memory growth after start beyond 1 GB at 4 sessions | four decoding at ~145 -> ~185 tokens/s instead of pairs in turn (144 ms a step each) |
| P1 | The tree inside a batch of two: 2 x up to 24 rows, `kMaxStreamRows` 64 and the 33-64-row tier | `--batch-check` with trees; greedy 2-vs-1 16/16; one stream unchanged | built (branch `batching/p1-tree-batch`): +3% at two (129.0 ms and 135 steps against 119.6 and 150 for 1,000 tokens), less than the +8-9% hoped for, because the FFN megakernel does not take more than 32 rows; parked behind C1, C2 and P2 |
| P4 | Paged KV and shared prefix pages (§5.3), in steps P4a-P4d behind `--kv-pages` | each step's gate in §5.3: memcmp paged against flat (`ling-paged-tests`); the engine's checks with `--kv-pages`; agentlong greedy 16/16 at 1, 2 and 4 streams at `--sessions 4 --max-context 262144`, and with a small pool; one stream within noise | four or more conversations at the full 262,144 context (the owner's decision of 2026-10-27): a pool of ~1.02M positions on node-b instead of four 262,144-position slots that do not fit; then Mightling's recipe passes `--sessions 4 --batch 4`. Design 2026-10-27, P4a started |
| P5 | FP8 KV (32 KB a token) behind the quality gate | VALIDATION §13's quality gate (not exact: a switch, off by default) | 2x the sessions per GB |

In that order: C1 and C2 are what make four conversations each move (§3.3), P2 what makes them fast once they all decode; the owner's bar for 1.8.0 (§4) needs P0, C1, C2 and P2, and P4 to hold four conversations at the full context (§3.5). P4 is what lets a node hold Night Shift's and the user's conversations together at full context. Single-stream speed work goes on beside this on a second engine machine.

**After C2 (2026-10-28).** C2 measured exact everywhere but slower (warm four pooled 25.6 against 34.1 tok/s; `--fused-pass` off, branch not merged; [C2-integration §10](./drafts/C2-integration-2026-10-28.md) on `batching/c2-integrate`). The slowest conversation at four is designed for in [drafts/C3-per-conversation-speed](./drafts/C3-per-conversation-speed-2026-10-28.md): a fair, floor-driven scheduler (A), three and four sequences in one pass with rows by batch size (B, P2's first step), and a kill test for overlap (C); design only.
