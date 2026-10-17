# Prefill beside decode at two streams (design and result, 2026-10-17)

The owner's item of 2026-10-16 (after the cuBLAS removal and the drafter's re-quantization): close the two-stream
aggregate gap without letting the decoding stream stall the way SGLang's does; designed first, exactness-gated,
measured against the same production reference. It is [M4 §6](./M4-sessions-and-batches-2026-10-10.md) item 6
("prefill on a second stream beside decode").

## 1. Where it stood

| M0's main set, seeded 1013, two streams, prompts identical (`--custom-tools drop`) | Engine (main, 2026-10-16) | Production SGLang (stage 5) |
| --- | --- | --- |
| Aggregate, output tokens per second over the wall | 46.9 and 47.0 | 53.1 and 50.6 |
| Server-side prefill, all requests | 266 s | 188 and 195 s |
| Decode per stream | 33.3 | 36.1 and 33.0 (34.6 pooled) |
| A 35K-token prefill beside a decode | 36.9-38.2 s, the other stream at 282-285 ms per step | 19.8 s, the other stream at 2,107 ms per step |

The decode side was level; what separated the aggregates was prefill time at two streams, and with the prompts
identical that is how the prefill is scheduled. Stage 2's policy: the running requests take turns, one unit
each; a prefill chunk that runs while the other request decodes is `--prefill-chunk-contended` tokens (256), then
the decoding request takes one step, then the next chunk. Two costs follow: small chunks are inefficient (every
chunk reads every weight once whatever its length), and the alternation doubles a long prefill's wall time while
the decoding stream runs alone at one-stream efficiency.

## 2. Options

**A. The budget, policy only:** a bigger contended chunk. Exact without new checks (`--chunk-check` and
`--interleave-check`: a prefill's state does not depend on its chunking).

**B. The prefill on a second stream, at the same time as the decode.** As built (branch `steps/prefill-side`):
the engine holds a second set of everything one pass writes (the activations of a 2,048-token chunk, the quantized
copies, the attention partials, the drafter's feature and context rows, the last row's logits in a pinned buffer)
and swaps it in for a side pass (`Engine::prefill_chunk_launch`, then `prefill_chunk_finish` once the chunk's event
has fired); the side stream belongs to a green context of N SMs (`--side-prefill N`), so the prefill's kernels run
on those SMs only, while the decode stays on the primary context, its captured graphs unchanged, and takes whatever
SMs are free. (A probe on 2026-10-16 showed why the decode is not confined instead: a graph captured on the primary
context and launched into a green context's stream runs on every SM; only a graph captured on the green stream
itself is confined. Kernels launched into a green stream stay on its SMs.) The side chunk waits for the main
stream's queued work at launch (begin_prefill's reset, state copy and donor KV for its session run there; the first
build without that wait gave the side session a different state), binding or prefilling another session waits for
the side stream, and a request that ends with a chunk in flight waits for its pass. ling-serve's scheduler launches
a full chunk on the side stream whenever another request is decoding, goes on with that request's steps, and
finishes the chunk (and launches the next) as soon as its pass is done. No kernel chooses its grid or split from
the SM count, so every unit computes the same bits on 32, 40 or 48 SMs.

**C. One mixed pass (prefill rows and decode rows in one forward):** rejected; the decode rows would take the
prefill path's numerics, which the exactness policy (MILESTONES §14) does not allow for decode and verify.

## 3. The bar (set before measuring)

Two-stream aggregate at least production's lower run (50.6), per-stream decode no lower than today's, and the
other stream's period during a 35K prefill no longer than today's 282 ms.

## 4. Measurements (2026-10-17, second-puffin; M0's main set, seeded 1013, two streams, custom tools dropped)

| Configuration | Aggregate | Server prefill | Decode per stream | 35K prefill beside a decode | Other stream during it |
| --- | --- | --- | --- | --- | --- |
| Contended 256 (main) | 46.3 | 273 s | 33.3 | 38.2 s | 285 ms |
| Contended 512 | 51.6 | 199 s | 34.4 | 25.8 s | 374 ms |
| Contended 1,024 | 53.3 (two runs) | 172 s | 34.3 | 20.5 and 20.7 s | 573 and 581 ms |
| Contended 2,048 | 53.3 | 168 s | 34.2 | 18.4 s | 997 ms |
| Side, plain second stream (no partition) | 52.6 | 205 s | 35.3 | 21.4 s | 402 ms |
| **Side, green context of 40 SMs** | 52.3, 52.3, 52.2 and 52.2 | 230 s | 36.6, 36.6, 36.5 and 36.5 | 26.6 s | 205 ms |
| Side, 32 SMs | 50.8 and 51.2 | 268 and 263 s | 37.5 and 37.4 | 32.8 and 30.7 s | 159 and 161 ms |
| Side, 24 SMs | 49.4 | 304 s | 38.4 | 41.8 s | 137 ms |
| Production SGLang (stage 5) | 53.1 and 50.6 | 188 and 195 s | 34.6 pooled | 19.8 s | 2,107 ms |

The 32-SM rows' second figures are after the scheduler change that finishes a side chunk and launches the next as
soon as its pass is done (the first build waited out the other request's step). The policy alone (A) reaches
production's best aggregate at 1,024 but lets the other stream's step reach about 0.6 s during a long prefill
(1 s at 2,048); the side stream (B) trades a slower long prefill for a decoding stream that hardly notices it.
Every configuration produced the same 20,507 warm tokens in the same 4,231 passes.

## 5. Decision

`ling-serve --side-prefill 40` is the default: it meets the bar on every line (aggregate 52.3 against 50.6, per
stream 36.6 against 33.3, the other stream at 205 ms against 282) and is better than the engine before on every
column but the 35K prefill's own time, which is slower than production's (26.6 against 19.8 s; it was 36.9-38.2
before). It is at production's aggregate (53.1 and 50.6) and about 6% faster per stream, and it bounds the other
stream at a tenth of SGLang's stall. `--side-prefill 0 --prefill-chunk-contended 1024` is the alternative when the
aggregate matters more than the other stream's latency (53.3 against 52.3; 0.58 s against 0.2 s during a long
prefill). The gate run on the branch is section 6.

## 6. Gates on the branch (`steps/prefill-side`, 2026-10-17)

- The 19 `ling-run` checks and the six unit suites (the side path off: the refactor of the scratch allocation and
  the split of `prefill_chunk` into its pass and the rest), same as main greedy and seeded: all pass.
- `--interleave-check --side-prefill 40` on the long and the short pairs, eager and with graphs, PDL and bulk
  attention; also at 32 SMs and on a plain second stream: every sequence the same tokens, state, steps and
  accepted drafts as alone, greedy and sampled, both ways round.
- `tests/server` against `ling-serve` with the side prefill on by default: 58 passed, 3 xfailed (as main).
- Greedy replay at one and at two streams, every request compared: the first gate run found 3 of 144 requests
  of one session different at two streams (and different again on a rerun), with the side off none. The cause: a
  prompt whose chunk ran on the side stream and ended at the end-of-prompt position queued the snap copy there;
  the other request had ended, so the next chunk ran on the main stream and overwrote the session's state under
  the copy, and the session's next turn resumed from a broken snap. `--interleave-check` had the same split (A
  ended before B's prefill) but nothing in it resumes from that snap. Fixed (a main-stream chunk waits for the side
  stream first); then greedy at two streams equals one stream on every request, twice (20,771 tokens in 3,718
  passes each), and the seeded replays are unchanged (52.2 and 36.5).
- The seeded replay at one stream: the same 20,507 tokens in 4,231 passes at 101.0 ms as main (the side path
  never engages there).
