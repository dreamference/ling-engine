# Design note: one model across both GB10s (tensor parallelism)

*Draft, 2026-10-12. **After M5, not scheduled** ([BACKLOG §19.21](../DREAMFERENCE_LING_ENGINE_BACKLOG.md); the owner's decision of the night of 2026-10-12). Nothing here is built or measured. Numbers measured on our machines cite their report; the link's latency and bandwidth are the advertised line rate or typical figures for this class of NIC, marked as estimates; the 1.5–1.8× is an estimate to be measured, not a result. This note revisits a non-goal of [OVERVIEW](../DREAMFERENCE_LING_ENGINE_OVERVIEW.md) ("multi-node inference across two Sparks over ConnectX-7") for after the first release.*

## 1. Why it could pay

Decode is weight-read bound. Each verify streams the target's 17.6 GB ([MODEL §5.2](../DREAMFERENCE_LING_ENGINE_MODEL.md)), ~82 ms of the ~108 ms single-stream step at ~215 GB/s ([reports/M1.md](../../reports/M1.md) §5, [reports/M3.md](../../reports/M3.md) §1; M4 stages 1 and 2: 107–108 ms on the replay). A second GB10 has its own 273 GB/s bus. Splitting every layer's weights across the two machines halves the bytes each reads per step, and the per-head work (attention, the DeltaNet recurrence, the KV cache and the recurrent state) halves with it. The price is an exchange of partial results over the DGX Spark's ConnectX-7 link after every row-split projection. A vendor claims "up to 1.7×" for two-node tensor parallelism on two 64 GB units ([reports/hn-survey-2026-10-09.md](../../reports/hn-survey-2026-10-09.md) §4); the estimate below is our own arithmetic for this model and this engine.

Pipeline parallelism (layers 0–31 on one machine, 32–63 on the other) was considered and is not the proposal: a step's layers run one after another, so at one stream each machine idles half the step and the step is no shorter. It only adds aggregate throughput with a second stream in flight, which two separate engines already give today.

## 2. The split

Standard tensor parallelism over two ranks, one all-reduce after each row-split projection:

| Part | Shape (in × out) | Split | Per machine |
| --- | --- | --- | --- |
| DeltaNet in-projections (q, k, v; z) | 5120 × 10,240; 5120 × 6,144 | by output (heads) | 8 of 16 key heads, 24 of 48 value heads |
| DeltaNet β/α, conv1d, recurrence, gated norm | per head | by head | local; the gated norm is over one head's 128 values ([MODEL §5.4](../DREAMFERENCE_LING_ENGINE_MODEL.md); the kernel is checked before building) |
| DeltaNet out-projection | 6144 × 5120 | by input (K) | **all-reduce 1** |
| Attention q (with gate), k, v | 5120 × 12,288; 5120 × 1024 × 2 | by output (heads) | 12 of 24 query heads, 2 of 4 KV heads; q/k norms and RoPE per head, local |
| Attention out-projection | 6144 × 5120 | by input (K) | **all-reduce 1** |
| FFN gate + up (fused) | 5120 × 34,816 | by output, gate and up halves paired | 8,704 of 17,408; SwiGLU local |
| FFN down | 17,408 × 5120 | by input (K) | **all-reduce 2** |
| Residual, RMSNorm | 5120 per row | replicated | both machines hold the full residual after each all-reduce |
| LM head | 5120 × 248,320 | by output (vocabulary) | 124,160 rows each; each logit computed whole on one machine; per-row top-K candidates exchanged and merged |

All the split dimensions fall on the tiled layout's boundaries: the halves of K are 3,072 (out-projections) and 8,704 (FFN down) values, multiples of the 128-value chunk; the halves of N are multiples of the 16-row tile. The NVFP4 scale blocks (16 values along K) never straddle a split.

**Two all-reduces per layer, 128 per step**, the DeltaNet layers included (their out-projection has the attention out-projection's shape). The drafter is a choice:

- **Form A, the drafter replicated.** Each machine runs the whole drafter (1.55 GB of layers and its pass over the 0.72 GB head, so each keeps a full copy of the head for it) on the target hidden states of layers 5, 19, 33, 47 and 61, which are full residuals and identical on both machines after the all-reduces. Both compute the same proposals; no exchange; no saving on its ~10 ms.
- **Form B, the drafter split too.** The same pattern over its 5 layers: 10 more all-reduces and a top-16 merge for the selector, about half its GEMM time saved.

Form A is the first build (fewer moving parts); Form B is measured after it.

## 3. The exchange, sized

`stream_gemm` accumulates in FP32 and writes FP32 ([reports/M1.md](../../reports/M1.md) §1), so a partial sum is 5120 × 4 B = 20 KB per row. A verify has as many rows as the draft block: 12 at `ling-serve`'s default `--draft-block 12`, 16 at 16; two streams in one batch (M4 stage 3 on) double it ([drafts/M4 §2](./M4-sessions-and-batches-2026-10-10.md)).

| Rows | Partial per all-reduce (FP32) | Wire time each way at 25 GB/s | Per step (128 all-reduces), latency L = 5 / 15 / 30 µs |
| --- | --- | --- | --- |
| 12 (one stream, block 12) | 240 KB | 9.8 µs | 1.9 / 3.2 / 5.1 ms |
| 16 (one stream, block 16) | 320 KB | 13.1 µs | 2.3 / 3.6 / 5.5 ms |
| 24 (two streams, block 12) | 480 KB | 19.7 µs | 3.2 / 4.4 / 6.4 ms |
| 32 (two streams, block 16) | 640 KB | 26.2 µs | 4.0 / 5.3 / 7.2 ms |

How the columns are made:

- **A two-rank all-reduce is one exchange:** each machine sends its partial to the other at the same time (the link is full duplex) and adds the one it receives. Time per all-reduce ≈ L + size ÷ bandwidth.
- **Bandwidth:** 200 Gb/s is the ConnectX-7's advertised rate, 25 GB/s each way at line rate. What the GB10's attachment of the NIC sustains is the first measurement (section 6).
- **Latency, three cases, all estimates:**
  - 5 µs: GPU-initiated RDMA writes polled from the GPU, back to back on one cable.
  - 15 µs: a host-proxied RDMA path.
  - 30 µs: a general collective library's small-message all-reduce across two hosts.
- **The bytes are not a bus cost:** 128 × 240 KB in and out is ~63 MB per step, ~0.4% of the 17.6 GB the step reads today.
- **Form B** adds 10 all-reduces at the same sizes: +0.15 to +0.4 ms.
- **BF16 partials** would halve the wire time, but they change the sum's rounding (section 5); the table keeps FP32.

At one stream the exchange is 2–5.5 ms per step whatever the block, so **latency dominates**, not bandwidth: at 12 rows the wire is 9.8 µs per all-reduce against 5–30 µs of latency.

## 4. The estimate: 1.5–1.8× on decode, to be measured

Per machine, from M3's trace of the single-stream step ([reports/M3.md](../../reports/M3.md) §1, 34.8K-token prompt, ~114 ms; the replay's ~108 ms has less attention at its shorter mean context):

| Part | One machine (ms) | Per machine, Form A | Per machine, Form B | Why |
| --- | --- | --- | --- | --- |
| Target weight GEMMs (17.6 GB at 215 GB/s) | 81.7 | 40.9 (44.0 at 200 GB/s) | 40.9 (44.0) | 8.8 GB each; the half-shapes may stream less well (fewer blocks per launch, a larger last wave) |
| Verify attention | 11.2 | ~5.6 | ~5.6 | 2 of 4 KV heads: half the KV read |
| DeltaNet recurrence, conv, gating | 2.1 | ~1.1 | ~1.1 | half the heads and state |
| β/α, norms, FP16 conversion, LM-head top-K | 3.9 | ~3.5 | ~3.5 | norms replicated, launches unchanged |
| Drafter (GEMMs 7.4 incl. its head pass, small kernels 3.0) | 10.4 | 10.4 | ~6.7 | replicated, or its GEMMs halved |
| Commit (DeltaNet replay, drafter context KV) | 3.4 | ~2.0 | ~1.7 | half the state; Form A keeps the drafter's KV whole |
| GPU idle | 1.1 | 1.1 | 1.1 | |
| **Without the exchange** | **113.8** | **64.6** | **60.6** | |
| Exchange, 12 rows, L = 5–30 µs | — | 1.9–5.1 | 2.0–5.5 | section 3 |
| **Step** | **113.8** | **66.5–69.7** | **62.6–66.1** | |
| **Speed-up of decode** | | **1.63–1.71×** | **1.72–1.82×** | |

The pessimistic corner is the half-shapes at 200 GB/s (+3.1 ms) with the byte-identical exchange of section 5 (×4 the bytes) at L = 30 µs (~8.8 ms, ~9.5 in Form B): 76.6 ms in Form A and 73.3 ms in Form B, 1.49× and 1.55×. Hence **about 1.5–1.8× on decode**, as the owner was told; tokens per step are unchanged (the same proposals, the same verify), so decode tokens/s scales by the same factor: ~45–48 tokens/s of decode today becomes about 70–85.

- **At two streams** the ratio is about the same or a little higher: the weight bytes are shared by the batch as on one machine, and the per-sequence work (attention, the recurrence, the commit) splits by head; the exchange grows to 3.2–6.4 ms (section 3). This is measured at M4 stage 4's shape, not assumed.
- **Prefill gains too, less precisely:** it is compute bound (~2,100 tokens/s), and each machine does half the FLOPs. The exchange per 2,048-token chunk is 128 × 2,048 × 20 KB ≈ 5.4 GB, ~0.21 s each way at 25 GB/s, against ~0.5 s of halved compute per chunk, so prefill gains ~1.4× without overlap and approaches 2× with it (chunk n's exchange beside chunk n + 1's GEMMs).

**What overlap is possible.** Within a layer the sum is on the critical path: the residual, the RMSNorm and the next projection need it. But the next projection's *weights* do not depend on it. `stream_gemm` already issues its first weight chunks before waiting on its input (PDL, `pdl_begin()` in `stream_gemm.cu`); during an exchange the bus can keep streaming the next GEMM's weights into L2 (~24 MB; 10–30 µs of exchange is 2–6.5 MB at 215 GB/s). Done well, the link's latency is spent reading bytes the step reads anyway, and the exchange mostly leaves the step. Three further levers, each measured alone:

1. Partials sent per finished block of output columns as the row-split GEMM completes them, rather than after the whole GEMM.
2. The RMSNorm fused into the receive.
3. In Form B, the drafter's exchanges beside the commit.

The table above assumes none of this, so its range is the no-overlap case.

## 5. Exactness: can the output stay byte-identical to one machine?

**The column-split parts are exact by construction.** Each output of a by-output split (the in-projections, gate and up, the LM head's logits) is computed whole on one machine with the same K order as on one machine, and every per-head operation is local.

**The row-split projections are where it can break.** The out-projections and FFN down are split by input; each machine sums half of K. `stream_gemm`'s order on one machine (`src/core/stream_gemm.cu`, `kWarps` = 8, `kChunk` = 128):

- warp w accumulates the K-chunks c ≡ w (mod 8) in sequence, in FP32 (the MMAs' own accumulation inside a chunk unchanged);
- the eight warp sums are then added in warp order, `sum = (((w0 + w1) + w2) + … ) + w7`, before `gscale × xinv`.

A plain two-machine all-reduce of two half-K partials groups the sum differently and is **not byte-identical**: FP32 addition is not associative. Three ways to make it exact:

1. **Split K by warp stripe and send the warp sums.** Machine 0 holds the chunks of warps 0–3 (c mod 8 < 4), machine 1 those of warps 4–7, and each computes exactly those warps' sums. The halves are interleaved at the 128-value grain, so this is a layout choice made when the weights are retiled at load. Each sends its four warp sums per output (4× the exchange: 960 KB at 12 rows), and both add the eight in warp order: byte-identical to one machine, by construction.
2. **The same split, relayed.** Machine 0 sends `w0 + w1 + w2 + w3`, summed in order, which is exactly one machine's running sum after four terms. Machine 1 continues the sequence through w7 and sends the result back. That is 1× the bytes each way, but two latencies per all-reduce. The measured latency picks between 1 and 2.
3. **Change the one-machine order first.** Make the warp reduction a fixed two-level sum, `(w0 + … + w3) + (w4 + … + w7)`, on one machine too. Then a plain 1× all-reduce of the two halves is byte-identical to the new one-machine engine; addition is commutative, so both machines get the same bits whichever adds its own partial first. It changes the exact path's rounding on one machine, so it is the owner's decision. It is judged like M1 and M2 were: [MILESTONES §14](../DREAMFERENCE_LING_ENGINE_MILESTONES.md)'s agreement test against production's own spread (the flip line), with `--spec-check`, `--prefix-check` and `--graph-check` unchanged.

**The same rule applies wherever two halves meet:**

- **The LM head.** Under top-k sampling (production's top-k 20) the global top 20 is exactly the merge of the two halves' top 20s, with ties broken by token id as on one machine. A full-vocabulary normalizer (top-k off, as the completions canary sends, and `logprobs`) is a sum over two halves and needs the same two-level order as option 3, or both halves' terms sent.
- **The host's decisions.** The lattice walk and the accept run on both machines from identical logits, with the request's RNG seed and state sent once. They agree as long as the logits do.
- **The reduction runs inside the captured step graph.** Today `LING_KSPLIT` allocates a buffer per step and switches graphs off ([CLI](../DREAMFERENCE_LING_ENGINE_CLI.md)); the exchange must not.

**If no exact form is fast enough**, tensor parallelism is a non-exact mode under §14's exactness policy, off by default. It may become the default only if: mean KL ≤ 0.0008 nats against the exact engine, no growth with position at 25K and 64K, top-5 agreement 100%, tool calls no worse on the 24 replayed agent prompts, `--spec-check` and `--prefix-check` bit-exact within the mode, and a SWE-bench night no worse. Speculation stays exact within the mode either way: both machines verify with the same numerics.

## 6. The measurements to make first

None needs the engine changed. Items 1 and 2 need the direct cable; item 4 needs only one machine and can run any time.

1. **The link, raw.** RDMA write latency and bandwidth between the two Sparks over a direct QSFP cable, at 4 KB to 2 MB (the perftest tools' write latency and bandwidth tests), with the buffers in the memory the GPU reads. Whether the NIC can register the same unified-memory allocation the kernels write, with no copy, is part of this; it is not yet confirmed on GB10. Then whether a GPU-initiated path (the GPU writing the doorbell, polling for arrival) works here and what it saves over a host-proxied one.
2. **A two-node all-reduce microbenchmark at the step's sizes.** The collective library's all-reduce test across the two machines, two ranks, at 240 KB, 320 KB, 480 KB, 640 KB, 960 KB and 1.9 MB (the ×4 exact form at 24 rows). Then the step's shape: 128 dependent all-reduces with a small kernel between each pair, captured in a CUDA graph. Then the same chain beside a weight-streaming kernel, to see whether the NIC's traffic or the polling costs the stream any bandwidth.
3. **The exact forms' cost.** The 4× exchange against the two-latency relay (section 5, options 1 and 2) on the measured link.
4. **The half-shapes on one machine.** `stream_gemm` on every split shape (N halved for the by-output parts, K halved for the by-input parts) at 12, 16, 24 and 32 rows, with GB/s per shape, against the full shapes. This gives the per-machine weight time without a second machine.

**Go rule, to be confirmed when the numbers exist:** item 2's 128-all-reduce chain at 12 rows costs ≤ ~8 ms per step in an exact form (or ≤ ~5 ms in a plain one, if the owner takes option 3), and item 4's half-shapes stream at ≥ 200 GB/s. That predicts ≥ 1.5× on decode by section 4's table. Below it, the note is closed with the numbers.

## 7. What it costs

- **second-puffin stops being a separate machine while the pair serves.** It is the development GB10 where ling-engine is built and benchmarked, and a second node and lane for Mightling (Night Shift, SWE-bench arms, `ling-admin node` jobs). Serving one model across both, neither machine runs a second model server, and no engine development, benchmark or A/B runs on second-puffin. Two independent engines give two lanes at today's per-stream speed; the pair gives one lane ~1.5–1.8× faster per stream, with the per-machine weights halved (~11 GB instead of ~22 GB), so more resident sessions per lane.
- **A direct cable.** Both machines are on Wi-Fi today. The exchange needs the ConnectX-7 ports cabled to each other (QSFP), so the machines sit next to each other. The LAN is not used for it.
- **One failure domain.** Either machine down takes the model down. Host safety runs on both machines: `check_host_safety()` before the load, the PSI watchdog during it. A kill on one stops both halves, and restarts are coordinated.
- **More to prove.** The exactness checks run across two machines; the agreement study and a benchmark night repeat on the pair.

## 8. What the product would need

- **Both machines as one served model.** One machine leads: HTTP, the tokenizer, the scheduler, `/metrics`, `/health`. The other follows: it runs the same step loop on its halves, and receives each request's token ids, sampling parameters and seed from the leader. Clients see one endpoint and one `/v1/models` entry; the launcher, the agents and the API surfaces are unchanged.
- **The node locator.** The pair is advertised as one node, the leader's record carrying the model; the follower is advertised as a member of the pair, not as a model server. Clients find the leader as today, through the tiers ending in the mDNS browse. The peer on the cable is found by node id at use time, never a stored address (the rule for nodes).
- **Pairing.** The two machines are paired with each other as today (management over SSH through the one forced command). The exchange is a new data channel: bound to the cable's interface only, never a port on the LAN, and refusing any peer that is not the paired node.
- **The model gate.** It stays in front of the leader only; the follower serves nothing to clients. `night pause` and the SWE-bench gate act on the pair as on one server.
- **Lifecycle.** `ling-admin server start` on a pair starts the follower, then the leader, after the host-safety checks on both. `server stop` stops both, and status reports both. Both machines hold the checkpoint (model sync to both), each retiling its half at load. The engine takes a launch option naming the peer by node id, placed in the model registry with the other launch flags.
- **The egress audit.** `ling` still talks only to its node as today. On the pair, the audit's allowlist adds the peer on the cable's link-local interface and nothing else.
