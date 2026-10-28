# C3: every conversation's speed at four (design, 2026-10-28)

BATCHING §4 item 1 is the 1.8.0 bar: at four warm agent conversations every conversation decodes at or above 34.6 tokens/s. After C1 (shortest prefill first, `--side-share 16`) the pooled rate passes it (35.6) but the slowest conversation does not (27.3 in C1's run `g16w`; 26-29 across runs). C2, the fused pass, was exact and slower and stays off ([C2-integration §10](./C2-integration-2026-10-28.md) on `batching/c2-integrate`). This note is the design for 1.8.x: where the slowest conversation's time goes, three designs with their expected gains, the exactness argument, the measurements, and the GPU windows. **Design only: nothing here is built.**

## 1. Where the slowest conversation's time goes

`g16w` (node-b, `~/batching/c1/w8/g16w.serve.log`, main with C1 and `--side-share 16`, `--sessions 4 --batch 4`, warm agentlong, four rounds). In every round three conversations send a small append (200-1,000 new tokens) and one a large one (21-34K). Per request, from the server's own `request:` lines:

| Round | Request | Out tokens | Passes (batched) | Step period | tok/s | What it decoded beside |
|---|---|---|---|---|---|---|
| 1 | s0 | 211 | 42 (42) | 184 ms | 26.9 | s3's 21K prefill, paired |
| 1 | s1 | 522 | 91 (70) | 173 ms | 32.6 | s3's prefill, paired, then alone |
| 3 | **s3** | 598 | 91 (**0**) | **266 ms** | **24.4** | s0's 33.6K prefill, **the only decoder** |
| 4 | **s2** | 629 | 130 (80) | **224 ms** | **21.4** | s1's 26.8K prefill, paired, then the only decoder |
| 4 | s0 | 662 | 101 (89) | 168 ms | 38.3 | s1's prefill, paired |
| 3 | s0 (the large append) | 531 | 59 (0) | 103 ms | 85.7 | nothing: its prefill done, the others finished |

Three causes, in order of weight:

1. **One decoder beside a prime has no hold.** `hold_side` (`src/serve/main.cpp:579-582`) holds the side stream only while `others_decoding(r) >= 2`. When the other small appends have finished, the last decoder runs every step beside a side chunk (223-266 ms against ~100 alone) for the rest of the prime: s3 and s2 above, the two slowest requests of the run. The hold was scoped to two or more decoders to keep the probe's prime unchanged with one decoder (BATCHING §2.2).
2. **The pairing favours the oldest request.** `run()` gives the turn round-robin (`main.cpp:463`), but `batch_with` (`main.cpp:508-516`) pairs the request whose turn it is with the *first* batchable request in `running_` order. With A, B, C decoding and D prefilling, the turns A, B, C, D form (A,B), (B,A), (C,A) and D's no-op: per three passes A steps 3 times, B twice, C once. With four decoding: A 4, B 2, C 1, D 1 per four passes. The pooled rate is the same as a fair pairing; the slowest conversation's step rate is a third to a half of it. This fits the one fast stream in every four-stream run (W11 `m4w` 27.0 / 33.9 / 37.0 / 60.9; `g16w` 27.3 / 36.9 / 44.2 / 53.9).
3. **All four decoding is a pair in turn.** `fit = kMaxStreamRows / draft_block` = 32 / 12 = 2. Four decoders cannot exceed half a pass each: 6.37 tokens per step (the run's mean) at 121.7 ms a pair is 26.2 tokens/s each, below the floor however they are scheduled. The phase is short today (1.6-2.3 s of a run, BATCHING §2.2), but its share grows as A shortens the prefill-beside phases.

Note on the counters: "beside a prefill" in the request line (`main.cpp:656`) means another request is *mid-prefill*, not that a chunk is in flight, so a held step counts as beside. The trace in §5 separates the two.

## 2. Design A: a fair, floor-driven decode scheduler (scheduler only)

**A1. Fair pairing.** The batch for a turn is the request whose turn it is plus the batchable requests with the fewest decode passes in a trailing window (16 passes), ties by `running_` order. Three decoders then step (A,B), (C,A), (B,C): two of three passes each.

**A2. A decode-rate floor in place of the pass count.** While any decoding request's trailing rate (tokens over its last 16 passes' wall time) is below `--decode-floor F` (default 36 tokens/s: the bar plus a margin), the side stream does not launch a prefill's next chunk. This applies at **one decoder or more**. Two bounds keep C1's results:
- *A short prefill is exempt.* A request with at most 2,048 tokens left (a warm append) is never held: its whole prefill is the follow-up's first token (C1's 20.5 -> 1.43 s).
- *A prime still moves.* A held prefill takes its next chunk after `--prefill-max-hold S` seconds (default 3) whatever the rates, the same rule as C1's 5 s yield cap. So the prefill keeps a minimum share and a floor that cannot be met (four decoders in pairs, §1 item 3) cannot starve it.
`--side-share` stays as an override for comparison; with the floor on, its default becomes 0.

**A3. Chunk size as the second dial.** The prefill is chunk-size invariant (contended 256 and full 2,048 chunks give the same bits, BATCHING §5.5), so the side chunk may be cut to 1,024 while the floor binds: a decoder's step beside it is shorter and the hold finer, at some prefill efficiency (the 2026-10-17 sweep: 1,024-token chunks prefill 35K in ~20.6 s against 18.4 s at 2,048). Swept, not assumed.

**Expected.** One decoder beside a prime: with alone steps of ~100-105 ms (tree, ~6.7 tokens) and beside steps of ~250 ms, a rate of 36 needs at most ~50% of its steps beside a chunk; s3 and s2 move from 21-24 to ~36 for as long as the cap allows (the cap releases a chunk every 3 s at worst, so the floor holds while a chunk takes under ~1.5 s). Three decoders paired fairly: (2/3) x 6.37 / 0.122 = ~35 tokens/s while none is beside a chunk (today's C: ~17). Round time: the side-share sweep moved N from 0 to 32 with rounds flat at 139-144 s and the probe's prime unchanged under the hold, because held prefill time is decode time done alone at full speed; the gate allows +5%. Cost: a cold prime beside one decoder takes longer to its first token (bounded by the cap; measured as B in the probe). **Slowest of four warm: 27.3 -> ~33-37, short of the floor only in the all-four phase.**

**Exactness.** Order of units only, as C1. Every unit runs in its own session's state; a batched step is bit for bit each sequence's step alone (`--batch-check`); a greedy verify accepts the target's argmax whatever was proposed. So greedy output is identical per conversation (gate: 16/16 against one stream). What changes is *whether* a step is paired (12-row chain) or alone (16-row tree), so accepted lengths and with them seeded draws differ from today's runs: sampling stays exact in distribution, identical only where the batch shapes match, as BATCHING §2.3 already states for C1.

**Effort.** 2-3 days: the window bookkeeping per request, the hold rule, the exemption and cap, two flags, the per-step trace (§5), host tests of the pairing and hold decisions with a fake engine.

## 3. Design B: three and four in one pass, costed per conversation

P2 (BATCHING §5.1-5.2, §6) puts three or four sequences in one verify. This design adds the per-conversation arithmetic and a cheaper first step.

Step costs from W5 and §3.1 at ~27K (attention grows ~0.37 ms per 1K tokens per sequence): one 12-row sequence 99 ms; each added sequence ~18.4 ms (attention 9.8, DeltaNet off the megakernel 8.6); rows past 12 to 24 +4.6 (FFN); rows past 24 to 32 +11 (the four-group tile). Tokens per step by rows from `g16w`'s accept histogram (852 steps, `E[min(accepted, d)] + 1`): 12 rows 6.37, 10 rows 5.96, 8 rows 5.40, 6 rows 4.61.

| Decoding | Today (oldest favoured) | A1 (fair pairs) | B1: one pass, rows by batch size, <= 32 | B2: P2 proper (48-64 rows, N-sequence DeltaNet kernel) |
|---|---|---|---|---|
| 3 | slowest ~17 tok/s | ~35 | 3 x 8 = 24 rows, ~140 ms: **~38.5** | 3 x 16, ~140 ms with the -6 ms a sequence: ~45 |
| 4 | slowest ~13 | ~26 | 4 x 8 = 32 rows, ~170 ms: **~32** | 4 x 12-16, ~150-170 ms: ~36-40 |

**B1** (the cheaper first step): `speculate_batch` over three or four sequences inside today's 32-row tiers, each verifying the first `d` rows of its drafter block with `d` a fixed function of the batch size (12 at two, 8 at three and four). No 33-64-row tier is needed; it needs `kMaxSeqs` 4 (it is) and graphs per batch size from §5.2's descriptor table, or an eager verify first (its cost, 1-3% of the pass, is measured in the same window). **B2** is P2 as specified; it ships when the four-decoding phase is long enough to matter after A, which the trace measures.

**Exactness.** Greedy output does not depend on how many rows are proposed, only on each row's logits being what the row gives alone. B1 adds the 25-32-row tile at three and four sequences, so its gate is a memcmp of every row against the same row in a 12-row launch (`ling-stream-tests`), `--batch-check` generalised to three and four, and 16/16 at 3 and 4 streams. The row depth is a function of the batch size only, never of acceptance, so seeded draws are identical whenever the batch shape is (BATCHING §5.1).

**Speculation depth by acceptance per stream (considered, not proposed).** Rows are nearly free alone (a 16-row verify costs a one-token step, 84.8 against 84.9 ms), so a shallower tree alone only loses tokens (12 -> 8 rows: 6.37 -> 5.40). Depth earns something only as above, to fit more sequences into a pass, and as a function of the batch size. Moving rows between sequences by acceptance (more rows to the stream that accepts more) raises the pooled rate and lowers the slowest stream's, the opposite of the bar, and breaks seeded identity across shapes (§5.4). Not for 1.8.x.

## 4. Design C: decode and prefill overlapped (C2 done right; a kill test first)

The moonshot's M6 (Mightling `specs/DREAMFERENCE_MOONSHOT_BEAT_HUSKY.md`) is the 2.0 home of this; C3 only defines the test that decides whether any of it belongs in 1.8.x. What is known: C1's side stream already overlaps (the 2026-10-17 sweep: in 26.6 s the decoder takes ~130 steps beside a 35K prefill against ~100 for the same prefill time-sliced, about 1.2x the work of either alone); C2's fused pass lost that and more: its carrying step costs about decode alone plus chunk alone (247-268 ms), because each projection is a handoff that puts the chunk's state kernels (~80-90 ms) on the critical path, the decode rows cost ~43 ms inside the fused GEMM, and the chunk's attention runs 2x slower beside the GEMMs (C2 note §10).

**The kill test** (node-c, no lock, the C2 branch and `fused_breakdown.py`): the chunk one layer behind the decode, with its projections unfused (so no shared weight read, but no handoff per projection), against today's side chunk at the same chunk size. **Bar:** a decode step beside a 256-row chunk at or under ~150 ms with the prefill at or above 1.2K tokens/s, i.e. better than C1's side path (warm-4 pass 189 ms, steps beside a chunk 205-313 ms). Below the bar, C stays in 2.0 and A plus B are the 1.8.x plan. **Expected if it passes:** the steps beside a prime, the largest phase in §1, from ~250 to ~150 ms; with A's floor the prime then loses less time to the hold.

**Exactness** as C2: each path's device code unchanged, the state-bearing kernels per sequence, `--fused-check` and memcmp per projection.

## 5. Measurements

- **A per-step trace** (prerequisite, part of A's build): `LING_STEP_TRACE=FILE`, one line per unit: time, kind (decode pass, side launch, side finish, main chunk, hold), the sessions in the pass and their rows, whether a chunk was *in flight*, its duration. From it: each conversation's time in four phases (alone, paired, beside an in-flight chunk, held), the share of passes each conversation got while it decoded, and the length of the all-four phase.
- **`streams_server.py`**: per request decode tok/s min and p10 per stream (the bar is about the slowest), and passes that ran unbatched beside an in-flight chunk.
- **Per run** (n_streams `--warm --cache-key`, `--sessions 4 --batch 4 --max-context 65536` as W11): rounds s, pooled and per-stream decode tok/s, slowest stream, TTFT median/max; `probe.py` plain and `--decoders 2` (A's decode, B's and C's TTFT); greedy 16/16 against one stream (`m1w`).
- **Baselines** already measured: W11 `m4w` (main before C1), C1's `g16w` (main), C2's warm-4 run.

Gates: slowest conversation >= 34.6 tok/s at four warm (A, then A + B1); rounds within +5% of `g16w` (140.9 s); C's TTFT median <= 1.5 s; B's TTFT in the probe within +25%; one stream unchanged (no hold, no pairing with one request); greedy 16/16 at 2 and 4.

## 6. GPU windows

All on node-b under `streams_window.sh` (the lock, the production server stopped and restarted, `mem_guard.sh`, at most four sessions) unless marked node-c.

| Window | What | Length | Decides |
|---|---|---|---|
| W-A0 | main with the trace only: warm 4, probe plain and `--decoders 2` | 30 min | §1's decomposition with in-flight chunks separated; the size of the all-four phase |
| W-A1 | A1 alone (fair pairing): warm 4, probe; 16/16 | 30 min | the pairing's share of the gap |
| W-A2 | A2 + A3: floor 30 / 36 / 42, cap 3 / 5 s, side chunk 1,024 / 2,048 (six arms, warm 4 each; probe at the chosen arm); 16/16 | 2 x 45 min | the defaults; merge if the gates pass |
| W-B0 | `ling-run` in place: one pass of N = 1-4 sequences x 8 / 10 / 12 rows at 27K and 45K, eager and graphs; `--batch-check` at 3 and 4 | 20 min (node-c) | B1's step table against §3's model |
| W-B1 | B1 in ling-serve: warm 4, probe; 16/16 at 3 and 4 | 30 min | B1's merge; whether B2 is needed |
| W-C | the kill test, Nsight on node-c | 1 h (node-c) | whether C enters 1.8.x |

Order: W-A0 and W-A1 in one slot (no new kernels; the trace and A1 are a day's work), then W-A2, then W-B0 and W-C on node-c beside them, then W-B1. Each merge follows the standing rule: exact, gates green, one stream within noise of main.
