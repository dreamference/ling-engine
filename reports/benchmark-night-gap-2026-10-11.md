# The benchmark night: ling-serve against the bar, 2026-10-11

The owner decided on 2026-10-10 that the Code Mode A/B benchmark night (50 SWE-bench tasks, two in parallel, about 16 hours) runs the day ling-engine can be the benchmark harness's model server for a whole night, and set the bar below. This report states the bar, what the harness needs from the server item by item, what ling-serve (`2288f21`) answers today and what it measures, the gap against the bar, and what closes it. The harness's needs were read from its source on 2026-10-10 (file paths in section 2, repository `dgxcoder`); the reference numbers are [M0](M0.md) §4 and §6; the engine's single-stream numbers are [RESEARCH §16.1](../specs/DREAMFERENCE_LING_ENGINE_RESEARCH.md) item 5, [M1](M1.md) and [M3](M3.md); the two-stream numbers come from the soak started on second-puffin on 2026-10-10 at 15:33 (section 3).

## 1. The bar

Production's two-stream numbers are the floor: M0 §4, "The replay reproduces that condition at 33.9 tokens/s and 145 ms" (the table's 144.8 ms is a per-stream step *period*, not a batch step; M0 §6's pooled warm row at 16 drafts reads 33.9 tokens/s, 5.05 accepted, 149.0 ms, and 34.2 / 142.8 ms at 12). The owner's 145 ms is the §4 figure. So:

| Item | Bar |
| --- | --- |
| Two tasks served at once | ≥ 33.9 tokens/s per stream, ≤ 145 ms step period, no stream waiting for the other |
| Admission | the harness admits 2 tasks from what `/metrics` reports |
| Endurance | no restart over 8 hours |
| Prefill | 35K-token prefills at the harness's pace, the other task not stalled |
| Probes and metrics | readiness, model discovery, admission gauges, the model gate's read-only routes |

## 2. What the harness needs from the server

Every "answered" row below was established by reading the harness's source and ling-serve's routes ([API](../specs/DREAMFERENCE_LING_ENGINE_API.md) §2, §6), not yet by the harness running against ling-serve; that run is the first thing the night's dry run does.

| Need | Harness source (`dgxcoder`) | ling-serve `2288f21` | Status |
| --- | --- | --- | --- |
| Readiness: `/health`, then a one-token chat completion with the model's repo id | `dreamference/runner/vllm_readiness_waiter.py`, `_pre_warm` | `/health` answers `ok` once the model is loaded; `/v1/chat/completions` answers, and `parse_request` (`src/serve/api.cpp`) never reads `model`, so the canary passes under any served name | answered |
| Model discovery: `/v1/models` gives `id` and `max_model_len` | `dreamference/night_shift/night_shift_host.py`, `served_model` | `data[0].id` = `--served-model-name`, `max_model_len` = `--max-context` (65,536) | answered |
| Admission: refuse to start while a request runs that is not the night's; wait for idle | `night_shift_runner.py`, `start_blocker` and `admit` (`wait_for_idle`): sums `sglang:num_running_reqs` + `sglang:num_queue_reqs`, watches `sglang:prompt_tokens_total` move | all three exported since `2288f21` (API §6); running is 0 or 1, queued counts the waiting requests, so the night's own two requests count as 2 and block nothing | answered |
| Parallelism: `sglang:max_total_num_tokens` × 0.9 // `task_context`, at most `max_parallel` | `night_shift_host.py`, `parse_metrics`, `parallelism`, `task_budget` (`NIGHT_POOL_SHARE` 0.9, `LAUNCHER_POOL_SHARE_PERCENT` 60); defaults `task_context` 49,152, `max_parallel` 3 (`dreamference/swe_bench/swe_bench_settings.py`) | the gauge reports the context size, 65,536 (one resident history): 65,536 × 0.9 // 49,152 = **1** task, with a 39,321-token compaction limit | **gap** |
| Model gate: read-only routes pass while it is closed | `dreamference/vllm_server/model_gate_service.py`, `READ_ONLY_PATHS` | the gate is a proxy in front of the server; `/health`, `/v1/models`, `/metrics` are ling-serve routes; nothing else is required of the server | answered |
| `/get_model_info`, `/get_server_info` | listed in `READ_ONLY_PATHS`, read by nobody | 404 (API §2) | not needed |
| Two tasks served at once | the night runs two `ling exec` tasks against one server | one request at a time; a second waits in the queue (API §1, §7) | **gap** |
| Long prefills while the other task decodes | each task's turn carries up to its compaction limit | prefill at 2,110 tokens/s at 35K (M2-prefill, [MILESTONES §14](../specs/DREAMFERENCE_LING_ENGINE_MILESTONES.md)), about 17 s per 35K prefill; the other stream waits for all of it | **gap** |
| Several sessions resident | two tasks alternate their turns | one resident history with prefix checkpoints: interleaved sessions evict each other and re-prefill from their last checkpoint ([BACKLOG §19](../specs/DREAMFERENCE_LING_ENGINE_BACKLOG.md) item 8) | **gap** |
| 8 hours without a restart | the night's length | being measured by the soak (section 3) | to fill |

## 3. Measured

**Single stream** (RESEARCH §16.1 item 5, measured 2026-10-10: ling-serve `7f02d8f`, 12 draft tokens, M0's tuning set, beside the resident reference server): **55.5 tokens/s** decode at a **103.6 ms** host step, 5.82 tokens and 4.82 accepted drafts per step, 1,336 steps; the ceiling bin is 16.8%. On M0's main set at 16 drafts: M1 45.5 tokens/s at 114.5 ms; M3 110.5 ms, 43.5 tokens/s at M2's acceptance. Production single stream: 47.1 tokens/s, 105.5 ms (M0 §4).

**Two streams: the soak.** On second-puffin against ling-serve on port 18080: M0's main set (12 sessions × 12 requests, 144 requests) at two concurrent streams, pass after pass for 8 hours from 15:33:05 on 2026-10-10, one file per pass, `~/m0/runs/soak-N.jsonl`, the log in `/tmp/soak.log`. Each file's last line is the pass summary (requests, errors, wall seconds, concurrency, per-request statistics); the lines before it are per-request records (input, cached and output tokens, time to first token, total time, tokens/s). **At 16:46 WEST, 73 minutes in, no pass had finished**: `soak-1.jsonl`'s last line was a per-request record written at 16:46:01. The table is left for the morning; the file names are the ones to read.

| Pass | File | Requests / errors | Wall (s) | Tokens/s per stream (decode) | Warm wait (s): total − output ÷ 55.5 | Cached on warm rows | Restart |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `~/m0/runs/soak-1.jsonl` | to fill | to fill | to fill | to fill | to fill | to fill |
| 2 | `~/m0/runs/soak-2.jsonl` | to fill | to fill | to fill | to fill | to fill | to fill |
| … | `~/m0/runs/soak-N.jsonl` | to fill | to fill | to fill | to fill | to fill | to fill |

Memory over the passes is not in the soak's files; the morning reads it from the server process beside the pass summaries. Any restart shows in `/tmp/soak.log`.

**The one record read.** A single cold request (worker 0, the first request of a session), not a pass measurement: 16,929 input tokens, 10,464 cached (the shared-prefix checkpoint), 449 output tokens, time to first token 14.81 s, total 24.08 s, **48.3 tokens/s** while decoding. By the brief's rule the wait is 24.08 − 449 ÷ 55.5 = **16.0 s**; for a cold request that includes the prefill of 6,465 uncached tokens, about 3 s at M2's rates, so about 13 s of it is inferred to be the queue behind the other stream's request. The decode rate is the single-stream rate: with two streams ling-serve does not decode two sequences in one step, it decodes one while the other waits.

## 4. The gap

| Item | Bar | ling-serve today | Gap |
| --- | --- | --- | --- |
| Per-stream decode, two tasks | 33.9 tokens/s, each stream always decoding | 48.3–55.5 tokens/s while decoding, 0 while waiting; on the one record, 449 tokens in 24.08 s = 18.6 tokens/s end to end | each stream waits for the other's whole request; the end-to-end rate falls below the bar as soon as the wait exceeds a third of the request |
| Step period | 145 ms per stream | 103.6 ms single stream; no batch step exists | a batch of two is not built |
| Admission | 2 tasks | 1 (pool 65,536 → 65,536 × 0.9 // 49,152 = 1) | the pool gauge must reach ≥ 109,227 tokens for 2 (≥ 163,840 for 3): 2 × 65,536 = 131,072 gives 2 tasks at a 58,982-token limit |
| Prefill | 35K at pace, the other task decoding | 17 s per 35K prefill, the other stream stalled for all of it | no interleaving of prefill chunks with decode steps |
| Sessions | two resident | one resident, two interleaved sessions re-prefill from the last checkpoint each turn | the morning's `cached` column on warm rows is the measurement |
| 8 hours, no restart | proven by a night | soak in progress | to fill |
| Probes and metrics | answered | answered by source reading | exercise the harness against ling-serve once |

## 5. What closes each gap

All of it is milestone M4 (MILESTONES §14), which the owner decided on 2026-10-10 to build as one piece: resident sessions and batches of two, together.

- **Batches of two** (SESSIONS §11): one weight pass per step for both sequences, each adding its KV, DeltaNet state and rows. From M1 §5's breakdown at 31K (weights 81.7 ms once; attention 11.5 and state 2.1 ms per sequence), a two-sequence step is roughly 125–130 ms, under the bar's 145 ms per stream at today's acceptance; an estimate to be measured, not a result.
- **Resident sessions**: each task's history stays resident (about 4 GB per full 65,536-token session in today's BF16 KV, half in M4's paged FP8), keyed by `prompt_cache_key` (Codex sends the thread id; BACKLOG §19 item 8), so the two tasks stop evicting each other and warm requests prefill only their new tokens.
- **The pool gauge** reports sessions × context once sessions are resident: 2 × 65,536 = 131,072 admits two tasks, 3 × 65,536 admits three (API §6 says so; today's value is the one-session case).
- **Prefill interleaving** (SESSIONS §11): a prefill chunk between decode steps, so a 35K prefill costs the other stream a chunk's time per step, not 17 s.
- **The dry run**: the harness's own admission, readiness and task budget against ling-serve, once, before the night.

## 6. The night with the engine today

With parallelism 1 the harness runs the 50 tasks one at a time, each at the single-stream rate. M0 states no request count per SWE-bench task (its sessions are 12-request windows of recorded `im-*` runs, M0 §3), so the night's length is **to measure**: one task through the harness against ling-serve, its request count and wall time, then × 50. The bounds from the numbers in hand, derived from the owner's figure rather than from the soak: 50 tasks two at a time in about 16 hours is about 38 minutes per task at 33.9 tokens/s per stream; one at a time at 55.5 tokens/s is about 19.5 hours if the tasks are all decode (2 × 33.9 ÷ 55.5 × 16 h) and about 32 hours if their time is mostly tools and tests (50 × 38 min). Either is past the 16-hour window.

## 7. What the morning reads

`ssh second-puffin 'cat /tmp/soak.log'` for the passes and any restart; `ssh second-puffin 'for f in ~/m0/runs/soak-*.jsonl; do echo "== $f"; tail -1 "$f"; done'` for each pass summary; the per-request lines of each file for the warm wait (total time minus output tokens ÷ 55.5), the per-stream decode rate, and `cached` against `in` on warm rows (a low ratio is the eviction of section 4). Those fill the table in section 3 and settle the ranking of the gaps.
