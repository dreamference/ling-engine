# Moonshot P0, the GPU half: what a verify costs by row count

October 13, 2026 · second-puffin (GB10), nothing else on the GPU (`m0-prod8000` stopped for the run and started again after) · `bench/rowsweep.py` as written, from the build of `moonshot/p1` at d37d3bc (`--copy` off, which leaves the measured paths as on main) · production flags `--graphs --pdl`, two repetitions · 18:27–19:19 WEST.

This is the GPU half that [the P0 statistics report](moonshot-p0-copy-stats-2026-10-13.md) §6 prepared. The logs and `summary.json` are in [moonshot-p0-rowsweep-2026-10-13/](moonshot-p0-rowsweep-2026-10-13/); the prompts are left out because the script regenerates them. **Step C was not run**: the drafter's own rate on copied spans at block 12 needs a replay of M0's main set through `ling-serve`. The break-even below therefore uses M1's two rates (8.2 and 10.9 tokens per step on copied spans, from block 16).

## Findings

1. **A production step costs 101 / 107 / 117 ms at ~8K / ~32K / ~56K context.** This is block 12: draft 10, verify 88–103 and commit 3 ms.
2. **The rows path's verify grows by 1.1–1.3 ms per row from 16 to 32 rows.** Below 16 rows it grows by only 0.3–0.6 ms per row, and from 24 to 32 rows it is steepest (1.0–1.4 ms).
3. **A 32-row pass through the prefill path costs 126–142 ms.** Each further row adds only 0.14–0.52 ms up to 128 rows, depending on context, but the pass has a large fixed cost.
4. **c(64), the cost of a 64-row copy step against a production step, is 1.29–1.33 by the prefill path and 1.39–1.48 by extrapolation.** The prefill path should be the upper bound, yet it comes out *lower*, so continuing the 16→32 slope overestimates. The best estimate is c(64) ≈ 1.3–1.5. At ~32K, which is the context the decision rule reads, it is 1.33–1.48.
5. **c(128) is 1.39–1.64 by the prefill path and 2.0–2.25 by extrapolation.**
6. **Under §6's decision rule, c(64) at ~32K is "between" (1.0–1.5): the edit-span feature is marginal.** It is not a no-go, but the extrapolated value (1.48) sits at the no-go edge.
7. **The measured c(64) cancels most of the realistic proposer's whole-session gain.** Interpolating the statistics report's table at c = 1.33–1.48:
   - realistic K=64: about +2.3% to +0.4%, against +6.8% at c = 1;
   - realistic gated K=64: about +10% to +9%;
   - oracle m1 K=64: about +13% to +11%;
   - edit scripts alone (realistic): about +13% to +10%.
   P1's gate (≥ +7% on whole sessions, byte-identical) is therefore reachable only with gating, or with a 64-row path cheaper than these numbers.
8. **Break-even copy length: about 11 to 16 accepted tokens per 64-row copy step.** A copy step beats the drafter's step only when its accepted run is longer than c(K) times the drafter's tokens per step on that span:

   | | drafter 8.2 tok/step | drafter 10.9 tok/step |
   | --- | --- | --- |
   | K=64, c = 1.33 | 10.9 tokens | 14.5 tokens |
   | K=64, c = 1.48 | 12.1 tokens | 16.1 tokens |
   | K=128, c = 1.53 (prefill path, ~32K) | 12.5 tokens | 16.7 tokens |

   - The realistic proposer averages 13.4 tokens per copy step at K=64, right at break-even, which is why it gains almost nothing.
   - The gated proposer averages 26.5 tokens, well above it.
   - `--copy-min` should therefore stay at or above a 16-token anchor, with gating.
   - The statistics report's break-even c for +20% (0.18 realistic, 0.62 oracle, K=64) is far below every measured c, as expected once the 20% bar was dropped.

## The numbers

**A, the rows path** (median of two repetitions, ms; at ~8K each prompt ran 75–84 spec steps, but at ~32K and ~56K the model stopped after 2–3 steps on filler text, so those cells rest on few steps. The repetitions agree within 1 ms.):

| Context | Block 8 | 12 | 16 | 24 | 32 | Slope 16→32 per row |
| --- | --- | --- | --- | --- | --- | --- |
| ~8K verify | 85.0 | 87.6 | 89.6 | 96.9 | 107.9 | 1.14 |
| ~32K verify | 92.0 | 93.4 | 94.7 | 104.8 | 115.0 | 1.27 |
| ~56K verify | 100.8 | 103.5 | 104.4 | 112.8 | 122.8 | 1.15 |
| draft (all contexts) | 9.7–10.0 | 10.1–10.4 | 10.5–10.8 | 11.8–11.9 | 14.1–14.4 | |
| commit (all contexts) | 3.0–3.3 | | | | | |

**B, one N-row pass through the prefill path** (ΔT/Δchunks between the prompt pair, ms):

| Midpoint | N=32 | 64 | 96 | 128 |
| --- | --- | --- | --- | --- |
| 7,295 | 126.5 | 129.9 | 133.7 | 139.8 |
| 31,158 | 133.2 | 142.4 | 152.9 | 163.7 |
| 55,871 | 141.7 | 155.6 | 171.5 | 191.8 |

Every chunk-check also reported the state and the next token identical to the reference chunking.

**c(K)** (copy step ÷ production step at block 12):

| Context | Step 12 (ms) | K=64 extrap. / prefill | K=96 | K=128 |
| --- | --- | --- | --- | --- |
| ~8K | 100.8 | 1.46 / 1.29 | 1.83 / 1.33 | 2.19 / 1.39 |
| ~32K | 106.9 | 1.48 / 1.33 | 1.87 / 1.43 | 2.25 / 1.53 |
| ~56K | 116.9 | 1.39 / 1.33 | 1.70 / 1.47 | 2.02 / 1.64 |

## What is still open

- **P1's own 33–64-row verify is not timed here.** It is the rows-path kernel that runs 33–64 rows as eight 8-token groups. P1's GPU slot measures it directly (`--spec --copy`, §8.5 of the draft). If it lands below the prefill path's 130–156 ms at 64 rows, c(64) drops toward 1.2 and the realistic gain recovers to about +4%.
- **C, the drafter's rate on copied spans at block 12**, still decides which column of the break-even table applies (the statistics report §6).
- **Batch 2 is not covered.** A batched verify (M4 stage 3) makes rows cost more, and it should be swept when it lands.
