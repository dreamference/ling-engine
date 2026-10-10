# ling-engine: Milestones, status, the agreement gate and the risks

Part of the ling-engine specification; the index is [README.md](./README.md). Section numbers are those of the single-file `SPEC.md` this was split from (Oct 8, 2026 · revised the same day against Mightling's measured workload), so a citation of the form `§n` or `S§n` in the reports and the code still names the same text.

## 14. Milestones

Six milestones, each gated by a measurement on the Spark; M1 is the long pole, because everything after it multiplies the pass time it establishes. Rough effort for one engineer with review, labelled as an estimate: about four months end to end, M1 taking a third of it, M0 two weeks.

![M1's single-row pass gates every speculation milestone](../images/roadmap.svg)

M1 starts as soon as M0's breakdown shows the engine can be meaningfully faster than tuned SGLang, without a separate review (the maintainer's decision, 2026-10-08). All milestone work runs on the development Spark, never on the production machine.

*roadmap · 6 milestones, 6 gates*

Each milestone ends at a measured gate; the highlighted M1 sets the pass time that M2 and M3 multiply.

M0 produces the numbers every later target is restated against (plus the five measurements of [section 16.1](./DREAMFERENCE_LING_ENGINE_RESEARCH.md)): the measured streaming bandwidth; the replay harness and the replay of production; a profile of production's step that splits its ~160 ms; a tuned SGLang and its replay; production's aggregate throughput at 2 and 4 streams; and the BF16 quality reference. If tuning alone closes much of the gap, M0 says so and the targets move up with it. M1 is the single-row fused path and must reach 90% of measured bandwidth before any speculation work starts, so that acceptance is measured against a tight step and not against overhead. M2 and M3 add speculation in two stages, exact first and fast second. M4 turns the engine into Mightling's model server: the endpoints, parsers and template of [section 12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md), shared-prefix checkpoints, FP4 prefill, images and batches of 2–4. M5 hardens it and tries the agent-tuned drafter.

**Status (9 October 2026).** M0 is done ([reports/M0.md](../reports/M0.md)). M1 is done ([reports/M1.md](../reports/M1.md)): exact speculation with the DFlash2 drafter, 45.5 tokens/s on the replay, 97% of production's decode. Prefill came next, ahead of M2's tree speculation, by the maintainer's choice ([reports/M2-prefill.md](../reports/M2-prefill.md)): from M4, the prompt template of [section 12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md) (ling-serve's prompts are now token-identical to production's on 396 replayed requests), FP4 prefill ([section 11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)'s numerics: NVFP4 activations for the FFN, FP8 for the projections, on block-scaled tensor cores; 2,640 / 2,580 / 2,110 tokens/s at 1K / 8K / 35K tokens against production's 2,140-2,460 / 1,930 / 1,680) and shared-prefix checkpoints (a new session resumes after the shared system message). The DeltaNet prefill runs the chunked form in 32-token chunks rather than 64, the KV cache stays BF16, and prefill attention is the rows path's tensor-core kernel. M3, the decode step, is done ([reports/M3.md](../reports/M3.md)): the step's draft, verify and commit as CUDA graphs, programmatic dependent launch, and an L2 prefetch in the verify attention take the step from 115.5 to 110.5 ms on the replay with every output bit unchanged; production's is 105.5, and M3 §7 lists what closes the rest (the weight stream first). Open: M2's tree speculation, the rest of M4 (paged FP8 KV, images, batches of 2-4) and M5. The API and serving fixes from the issue survey are merged (branch `api-fixes`): the four tool-call parser fixes, stop strings held back on UTF-8 boundaries, context overflow in the shapes the clients recognise, the NaN/inf guard, penalties over output tokens only, and chat streams that end properly after an error ([section 18](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) names each guard). Checked on the GPU after merging M3: the kernel, stream and prefill tests, `--spec-check` and `--prefix-check` with and without `--graphs --pdl`, `--graph-check`, and 23 server cases against ling-serve with `--graphs --pdl`.

**Agreement gate (standing, from 9 October 2026).** It replaces the 98.6% top-1 target on validate_v0.py's six prompts, whose 288 tokens cannot detect a change smaller than about 4.5 points ([reports/M2-prefill.md](../reports/M2-prefill.md) [section 6](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)). Measured with `bench/agreement/` on 54 prompts (validate_v0.py's six, 24 short chat prompts, 24 replayed agent prompts), teacher-forced on production's greedy 48-token continuation, against production at concurrency 1 with a cold cache; a flip is a position where the engine's top token scores strictly below production's top:

- flip rate not significantly above production's own self-noise, its greedy decode against its own prefill on the same positions: a one-sided Fisher exact test (the hypergeometric upper tail, observed table included) at α = 0.05. With production's 65 flips in 2,490 positions the threshold is 85 flips (3.41%): 85 gives p = 0.058, 86 gives p = 0.049;
- every flip's gap (production's top against the engine's choice) ≤ 1.75 nats;
- top-5 agreement 100%;
- no late concentration of flips by position in the continuation or by prompt length;
- tool-name match on the 24 agent prompts' first answers ≥ production's match with itself;
- exact-argument match judged only against production's match with itself (13 of 24 when measured).

M1 (56 flips, 2.25%, p = 0.82), `44c92c5` (68, 2.73%, p = 0.43) and M2 (66, 2.65%, p = 0.50) all pass the flip line; M2 is as far from production as production is from itself. Every other line passes for M1 and M2. `bench/agreement/analyze.py` reports counts, not verdicts; anything that turns them into pass/fail applies this test, with the threshold recomputed from production's own count whenever the prompt set changes.

**Exactness policy.** The engine's numerics are the exact path: decode and verify on the rows path, prefill as M2 built it. Every non-exact mode (a change of numerics, not of speed) sits behind a switch and is off by default. One may become the default only if:

- mean KL(exact engine ‖ mode) ≤ 0.0008 nats (twice production's own warm-against-cold spread at concurrency 1);
- the KL shows no growth with position at 25K and at 64K tokens of context;
- top-5 agreement with the exact engine is 100%;
- tool calls on the 24 replayed agent prompts are no worse;
- `--spec-check` and `--prefix-check` are bit-exact within the mode;
- a SWE-bench night is no worse.

BF16 prefill intermediates may be built behind a switch now. FP8 KV (§16.10) and the all-NVFP4 projections (§16.2) wait until the exact engine has beaten SGLang in a night.

**Backlog (9 October 2026).** The surveys of 9 October are merged and ranked in [reports/backlog-2026-10-09.md](../reports/backlog-2026-10-09.md); only measured wins move from it into this spec.

**Target: a benchmark night (set 10 October 2026).** The Code Mode A/B benchmark night (50 SWE-bench tasks, two in parallel, about 16 hours) runs the day ling-engine can be the benchmark harness's model server for a whole night. The bar (the owner's decision, 2026-10-10): two tasks served at once at no less than production's two-stream numbers ([reports/M0.md](../reports/M0.md) §4: 33.9 tokens/s per stream, a 145 ms step period), the harness admitting two tasks, no restart over 8 hours, 35K-token prefills at the harness's pace, and the harness's probes and metrics answered. Where ling-serve stands at `2288f21` is in [reports/benchmark-night-gap-2026-10-11.md](../reports/benchmark-night-gap-2026-10-11.md): the probes and metrics are answered (`/health`, the one-token chat canary, `/v1/models`, `/metrics` with the admission gauges; established from the harness's source, not yet exercised by the harness against ling-serve). The gaps: (1) one request at a time, so two streams are never decoded together: each decodes alone at the single-stream rate (55.5 tokens/s on the tuning set at 12 drafts; 48.3 on the soak's first cold request) and waits behind the other (16 s of wait and prefill in a 24 s request), against 33.9 tokens/s per stream with no wait in production; (2) `sglang:max_total_num_tokens` reports 65,536, one resident history, so the harness's parallelism is 65,536 × 0.9 // 49,152 = 1 against the 2 the night needs (a pool of at least 109,227 tokens); (3) one resident history, so two interleaved sessions evict each other and re-prefill from the last checkpoint every turn ([BACKLOG §19](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) item 8), and a 35K prefill (17 s at 2,110 tokens/s) stalls the other stream for its whole length; (4) the 8-hour run without a restart is unproven until the soak of 10 October (M0's main set at two streams, pass after pass) is read in the morning. All of it closes in M4: resident sessions and batches of two, which the owner decided to build together (2026-10-10), with the pool gauge reporting sessions × context once sessions are resident and a prefill chunk interleaved between decode steps ([section 11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)). With the engine as it is, the harness runs the 50 tasks one at a time; M0 states no request count per task, so the night's length is to be measured, between about 19 and 32 hours by the report's bounds, past the 16-hour window either way.

## 15. Risks and open questions

The two risks that can sink the targets are the bus and the toolchain; the third is that the gap closes from the other side.

| Risk | Effect if it lands | Mitigation | Decided by |
| --- | --- | --- | --- |
| Sustained LPDDR5x bandwidth well below 273 GB/s (CPU traffic shares the pins) | Every tokens/s target scales down with it | Measure in M0; restate targets; keep the CPU idle during decode (no tokenization or copies on the hot path) | M0: **did not land** (262 GB/s, 96%); CPU streams take bandwidth byte for byte, so the CPU-idle rule stands |
| Production's 65 ms of non-byte step time is mostly something the engine cannot remove (attention at long context, for example) | Step target slips toward 100 ms; decode toward 60 tokens/s | M0's profile splits it before M1 starts; the flash-decoding attention kernel is built and measured in M1 | M0: **moot** — the single-stream step is 105 ms, not 160, and its non-GEMM part is ~19 ms (DeltaNet 7.5, attention 6.1 at bandwidth, drafter 8.6 incl. GEMMs, commit and sampling 3.5); the GEMMs themselves run at 78% of the bus |
| Tuned SGLang closes much of the gap first | The engine's margin over the tuned baseline shrinks | Every gate is stated against the tuned number; M0's tuned run is the go/no-go for M1; useful tunings ship in production at once | M0 |
| CUTLASS and the CUDA toolchain lack mature block-scaled NVFP4 and FP8 MMA kernels for sm_121 at small M | M1 slips | Prototype the FFN kernel first, both NVFP4 and FP8, before committing the layout; FP8 is the documented fallback precision | M1 |
| The FP8 projections fail the quality gate at NVFP4 | They stay FP8; step at 25k is ~88 ms instead of ~76 | Targets are already stated both ways; move groups one at a time | M1 |
| The lookup branch gains less than simulated (both sources succeed on the same spans) | Acceptance ~5.3 instead of 5.8; decode ~62 tokens/s | Measured on the replay in M2; the 64-token branch and an agent-tuned drafter are the next levers | M2 |
| The drafter's 32k-token head costs acceptance | Decode falls by a few percent | Measure separately; 64k head as fallback | M3 |
| The DeltaNet replay interacts badly with tree speculation and batches (per-branch replays exceed the L2 budget) | Tree capped at 16 nodes; batches at 2 | Measure state traffic in M2; the chain path is unaffected | M2 |
| Shared-prefix checkpoints miss because Codex's prefix varies per session (dates, paths early in the prompt) | New sessions start slower than 1 s | M4 measures the hit rate on replay; the prefix layout is Mightling's to change if needed | M4 |
| Thermal throttling at the 140 W TDP during long decodes | Pass time drifts up after minutes | Record clocks in telemetry; benchmark runs of 10 minutes, not 10 seconds | M0 |
| The Qwen3.8 chat template or tool-call format changes in a model update | Prefix check triggers re-prefill on every turn | Pin the template hash in the manifest; re-prefill path is already required | M4 |
| An agent-tuned drafter overfits one user's sessions | Acceptance falls on new kinds of work | Hold out recent sessions; keep the shipped drafter as the fallback and pick per request class by measured acceptance | M5 |

Open questions:

- Does the SM120 block-scaled MMA path on sm_121 reach 90% of bandwidth at M = 1 and M = 16, or does the 1-row bucket need a separate dequant-and-FMA kernel over the same tile layout? (M0: the instructions exist and work on sm_121a — `mma.sync` `kind::mxf4nvf4.block_scale` and `kind::f8f6f4`, TMA, clusters of up to 8 with distributed shared memory, programmatic dependent launch; at M = 16 production's CUTLASS NVFP4 GEMMs reach 208–225 GB/s and cuBLASLt's FP8 ones 140–218.)
- ~~Where do production's ~65 ms of non-byte step time go?~~ Answered by M0 ([reports/M0.md](../reports/M0.md) [section 5](./DREAMFERENCE_LING_ENGINE_MODEL.md)): one stream's 105 ms step is 86 ms of GEMMs at 204 GB/s plus ~19 ms of DeltaNet, attention, drafter and sampling; the GPU idles ~1 ms. It moved lever 5 (batching) up and lever 3 (short requests) down.
- What is the first-token latency floor once tokenization and HTTP are included: is 0.25 s for a warm agent request comfortable, and how close to one pass can a short follow-up get?
- How much of the copyable output does the lookup branch capture once the two draft sources overlap, and does a cross-session suffix store (proposals from earlier sessions' outputs, not just this one's context) add to it?
- Is a 4-bit KV cache needed for sessions above 64k tokens, or is FP8 enough for the users this engine serves?
