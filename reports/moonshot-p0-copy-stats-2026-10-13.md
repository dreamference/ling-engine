# Moonshot P0: how much of the agent's output is a long copy

Night of October 12–13, 2026 · the statistics half of P0 of [the repository-aware decoding draft](../specs/drafts/MOONSHOT-repository-aware-decoding-2026-10-13.md) (§5), approved that night. CPU only, on the production GB10 beside night 4, from recorded sessions read in place; only aggregates are reported. The GPU half (the cost of a 64- and 128-row verify) is prepared in §6 and not yet run.

## Summary

1. **Long copies exist, but they are a minority of the output.** Of 16.25M output tokens in 972 SWE-bench sessions, the session's own history (S1) could have supplied **45.9% in runs of 16 tokens or more, 23.9% in runs ≥32, 14.1% ≥64 and 7.8% ≥128**. Edits and file writes copy most: an `exec_command` edit script has 29% of its tokens in runs ≥64, a file write 28%, prose 2%.
2. **The other sources add almost nothing at the lengths that matter.**
   - **Emission views (S2):** +0.1 to +0.3 points at every length. The agent here re-reads a file with `sed`/`cat` before it edits, so its edits already copy the raw read. The `-`/` ` diff view matters for `apply_patch` (runs ≥64: 22% → 33% of its tokens), but `apply_patch` is 0.3% of the output (181 calls).
   - **The JSON-escaped view is irrelevant for this model.** Qwen3.8's chat template emits a tool call's string arguments verbatim inside `<parameter=…>`, so a patch is never JSON-escaped in what the model writes, as a function tool or not.
   - **Repository files (S3):** not measurable here: no checkout of the tasks' repositories is cached on this machine (the code index stores symbols, not bodies). A proxy, the base-commit tool outputs other sessions of the same task saw before editing, adds +0.1 point.
   - **Other sessions (S4, leave-one-task-out):** 28.9% of the output in runs ≥16 but only 2.9% ≥32 and nothing ≥64. That is shared boilerplate (call syntax, `cd /testbed && python -m pytest …`), not code.
3. **Tokens per step, with copies verified at 64 rows (S1+S2), against DFlash2's ~5.** The drafter's real rate on the copied spans decides the answer, and M1 §6 measured it:
   - **Oracle start, M1-calibrated** (the drafter gets 10.9 tokens per step on those spans, M1's 32+ bucket): **5.8 tokens per step, +16%** (K=128: +17%).
   - **The same with M1's 16–31 rate (8.2):** +26% (K=128: +27%).
   - **Naive** (the drafter at its average 5 everywhere): +64%. This is the number the draft's moonshot target implicitly assumes, and M1 §6 already showed it is wrong.
   - **A realistic proposer** (anchored on the last 16 tokens written, follows the most recent earlier occurrence, no knowledge of the future): **+6.8% at K=64, +7.8% at K=128**. With an oracle gate that takes the copy only when it beats the drafter (the bound for a copy branch beside the drafter's block): +12% / +13%.
   - **At K=15 (M1's cap)** the same model gives +3% (oracle) and −0.4% (realistic), which reproduces M1's +4%: the method agrees with the GPU measurement where the two overlap.
4. **Go/no-go: it points to no-go for the 20% bar on whole sessions.** These are tokens per *step* with a copy step costing what a production step costs; the bar is tokens per *millisecond*.
   - **Conservative calibration:** the oracle reaches +20% only if a 64-row copy step costs at most **0.62×** a production step (0.89× with every source at K=128). The weight read alone is ~70% of a step and a copy step saves only the drafter's ~9%, so that is out of reach.
   - **The 16–31 calibration:** the oracle reaches +20% only if a 64-row copy step costs at most **1.5×** a production step.
   - **The realistic proposer:** would need copy steps at under a fifth of a step's cost (0.18× at K=64), so in practice it never reaches the bar.
   - **The cap was not the main reason for M1's +4%:** lifting it from 15 to 64 takes the realistic proposer from −0.4% to +7%.
   - **What could flip it:** the drafter's own rate on copied spans at production's block 12, which M1 measured only at block 16. If it is ≤ 8.5 tokens per step, the oracle clears the bar up to a 1.5× copy step. The GPU session measures it in the same sitting (§6, step C).
5. **Edits are where it pays:**
   - **Edit scripts at K=64:** +26% (oracle, conservative) and +18% (realistic) on their own tokens; `apply_patch` +25% and +20%. P1's gate for edit spans (≥2× tokens/s) is out of reach on this workload; +15–30% on edit spans is what is there.
   - **The deciding unknowns:** what a 64-row copy step costs (§6; at 1.2× a production step the realistic edit gains fall to +15–18%, at 1.5× to +10–14%), and whether interactive sessions edit in larger, copy-heavier blocks than SWE-bench's `old = """…"""` snippets (§5).

## 1. Corpus and method

**Sessions.** Every finished session of every SWE-bench run on this machine:
- **What is in:** nights 1–3 (275 sessions), im100 (300), the `im-*` runs (240; M0's main and tuning sets were drawn from them), the `ab-*`, `acc-*` and `idx*` runs (154), and night 4's first 3 finished instances. That makes **972 sessions, 95,883 responses, 16.25M output tokens**.
- **What is left out:** night 4's two running instances and the smoke runs.
- **M0's main set:** the exact 12-session list is not stored on this machine; §2 reports the `im-*` population it was drawn from.
- **Reasoning:** the agent sends `reasoning.effort = none` (M0 §3), so there are no hidden reasoning tokens: the visible output is the whole output.

**Rendering and tokenization.**
- **Output:** each response is rendered as the checkpoint's chat template renders it: the content, then each call as `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>…`, with string values verbatim, then `<|im_end|>`. It is tokenized once with the Qwen3.8 tokenizer.
- **Fidelity:** the rendered count equals the recorded `output_tokens` for **99.65% of 95,872 responses** and is within 2 tokens for 99.96% (median difference 0).
- **Context:** rendered in the same form, as one stream: system prompt, developer and user messages, every earlier output, and tool outputs inside `<tool_response>`.

**Classes** (each output token by the span it falls in):

| Class | What | Share of output | Responses |
| --- | --- | --- | --- |
| `shell` | other `exec_command` commands, one-liners and test scripts | 35.7% | 75,021 |
| `prose` | the assistant's message text | 21.7% | 30,322 |
| `call_syntax` | the tool-call wrapper, parameter names, `<|im_end|>` | 17.3% | 95,883 |
| `file_write` | `cat`/`tee` heredocs into a file, Python writing a file without a replace | 16.5% | 5,589 |
| `edit_script` | `exec_command` edits: Python `old`/`new` replace-and-write, `sed -i` | 7.5% | 4,015 |
| `tool_other` | other tools' arguments (code index, `write_stdin`, `wait`) | 1.0% | 17,853 |
| `edit_patch` | `apply_patch`: 80 as a function tool, 101 through the shell, none freeform | 0.3% | 181 |

The agent edits almost entirely through `exec_command`. `apply_patch` as a function tool, the form night 5 tests, barely occurs in these runs (80 calls; 101 more through the shell), so its rows rest on 181 calls in 99 sessions. The classes come from a heuristic on the command text; a test script written with `cat > /tmp/t.py` counts as a file write.

**Sources.**

| | Corpus at the position | Notes |
| --- | --- | --- |
| S1 | Everything before the output, plus the output so far | The resident session's whole history, across compactions |
| S2 | Every tool output in that history in three views, each tokenized as a whole span | `num`: line-number prefixes removed (`cat -n`, `grep -n`, `nl`, `code_show`)<br>`diff`: every line prefixed with ` ` and with `-`<br>`json`: JSON-escaped |
| S3p | Proxy for S3: the tool outputs (raw and viewed) that up to 2 other sessions of the same task, in other runs, saw before their first edit | Base-commit content, but only what other agents chose to read |
| S4 | Every other session's outputs; all sessions of the same task (across all runs) excluded | Over the whole corpus, and within the same repository |

**Matching** is exact on token ids.
- **Oracle:** at each position, the run is the longest extension from the most recent or the first earlier occurrence of the next 16 tokens (S4: the first occurrence in another task). The proposer is told where a run starts, but not which occurrence is best.
- **Runs:** a run never reaches past the tokens that exist at that moment. Runs below 16 tokens are not counted.
- **Shares:** "share in runs ≥T" is a greedy left-to-right parse that takes every run of at least T and jumps past it.

**The step model.** DFlash2 alone makes N/5 steps for N tokens (production: 4.95–5.24 tokens per step, M0/M1).
- **The copy step:** a run of L tokens verified at K rows takes ⌈(L+1)/(K+1)⌉ steps, each accepting up to the first mismatch plus the bonus token. It replaces the drafter steps those L+1 tokens would have cost at the drafter's rate on copied spans:
  - **naive:** 5;
  - **m1_16:** 8.23, M1 §6's 16–31 bucket;
  - **m1:** 10.89, the 32+ bucket.
- **Gain:** (N/5) / (N/5 − saved + copy steps × c), with c the copy step's cost relative to a production step. The tables give c = 1, and the **break-even c**: the largest step inflation that still leaves +20%.
- **The realistic proposer** decides on the backward match (≥16, or ≥32) and is charged M1's drafter rate for that bucket. A copy step that accepts nothing costs a whole step: 15% of its copy steps at K=64.

## 2. How much is copyable

**Share of all output tokens in runs of at least T tokens:**

| Sources | ≥16 | ≥32 | ≥64 | ≥128 |
| --- | --- | --- | --- | --- |
| S1 | 45.9% | 23.9% | 14.1% | 7.8% |
| S1 + `num` view | 46.0% | 24.1% | 14.2% | 7.9% |
| S1 + `diff` view | 46.0% | 23.9% | 14.1% | 7.8% |
| S1 + `json` view | 45.9% | 23.9% | 14.1% | 7.8% |
| **S1 + S2** | **46.1%** | **24.2%** | **14.3%** | **7.9%** |
| S1 + S2 + S3p | 46.2% | 24.2% | 14.3% | 7.9% |
| S4, same repository | 28.0% | 2.7% | 0.0% | 0.0% |
| S4, all repositories | 28.9% | 2.9% | 0.0% | 0.0% |
| All sources | 53.0% | 25.4% | 14.3% | 7.9% |

**Per class, S1 + S2** (S1 alone in brackets where S2 changes it):

| Class | ≥16 | ≥32 | ≥64 | ≥128 |
| --- | --- | --- | --- | --- |
| `edit_patch` (181 calls) | 61.0% (49.1) | 48.5% (33.3) | 33.3% (21.9) | 18.1% (11.8) |
| `edit_script` | 63.7% (61.9) | 49.1% (47.1) | 29.9% (28.6) | 15.5% (14.8) |
| `file_write` | 51.0% | 39.1% | 28.4% | 17.3% |
| `shell` | 49.3% | 29.4% | 17.8% | 9.7% |
| `tool_other` | 55.8% | 19.1% | 1.9% | 0.6% |
| `call_syntax` | 73.2% | 13.9% | 2.9% | 1.1% |
| `prose` | 8.9% | 3.7% | 1.9% | 0.9% |

**Run lengths** (S1+S2, share of all output in runs of each length): 16–31: 23.7%, 32–63: 8.8%, 64–127: 6.0%, 128–255: 4.7%, 256–511: 2.2%, 512+: 0.7%. Half of the copyable text sits in runs of 16–31 tokens, two or three steps of DFlash2's 12-token block.

**By group** (S1+S2):

| Group | Output tokens | Runs ≥16 | Runs ≥64 | Gain at K=64, m1 |
| --- | --- | --- | --- | --- |
| Nights 1–3 (and night 4's first 3) | 5.76M | 48.9% | 15.2% | +16.9% |
| im100 | 4.78M | 45.4% | 15.7% | +15.8% |
| `im-*` (M0's replay population) | 3.07M | 42.1% | 11.2% | +14.1% |
| ab, acc, idx | 2.64M | 46.3% | 13.5% | +15.8% |

The prompt variants (default, high-SWE, refine, tests-first, issue-first) move the shares by a few points, not the conclusion.

## 3. Tokens per step

**Oracle start, all classes**: tokens per step, gain over DFlash2 alone at c = 1, and the break-even c for +20% tokens/ms.

| Sources, K | naive (drafter 5 on copies) | m1_16 (8.23) | **m1 (10.89)** |
| --- | --- | --- | --- |
| S1+S2, K=15 | 7.0, +39%, c ≤ 1.62 | 5.5, +11%, c ≤ 0.63 | 5.1, +3%, c ≤ 0.26 |
| S1+S2, K=32 | 8.0, +59%, c ≤ 3.09 | 6.2, +23%, c ≤ 1.21 | 5.7, +13%, c ≤ 0.50 |
| **S1+S2, K=64** | 8.2, +64%, c ≤ 3.84 | 6.3, +26%, c ≤ 1.50 | **5.8, +16%, c ≤ 0.62** |
| S1+S2, K=128 | 8.3, +66%, c ≤ 4.16 | 6.4, +27%, c ≤ 1.63 | 5.8, +17%, c ≤ 0.67 |
| All sources, K=64 | 9.0, +79%, c ≤ 3.89 | 6.5, +30%, c ≤ 1.68 | 5.9, +18%, c ≤ 0.84 |
| All sources, K=128 | 9.1, +81%, c ≤ 4.15 | 6.6, +31%, c ≤ 1.79 | 5.9, +19%, c ≤ 0.89 |

**The gain against the copy step's cost c** (S1+S2):

| | c = 0.9 | 1.0 | 1.2 | 1.5 |
| --- | --- | --- | --- | --- |
| Oracle K=64, m1_16 | +27% | +26% | +24% | +20% |
| Oracle K=64, m1 | +17% | +16% | +14% | +11% |
| Realistic K=64 | +8.3% | +6.8% | +4.0% | +0.1% |
| Realistic gated K=64 | +13% | +12% | +11% | +9% |
| Realistic K=128 | +9.2% | +7.8% | +5.1% | +1.2% |

**Realistic proposer (S1+S2, 16-token anchor, most recent occurrence), drafter rates from M1's buckets** ("—": +20% is out of reach even for free copy steps):

| Policy | Copy steps | Mean tokens per copy step | Gain at c = 1 | Break-even c |
| --- | --- | --- | --- | --- |
| Copy when anchored ≥16, K=15 | 596K | 9.2 | −0.4% | 0.07 |
| Copy when anchored ≥16, K=32 | 451K | 12.2 | +4.9% | 0.14 |
| **Copy when anchored ≥16, K=64** | 409K | 13.4 | **+6.8%** | 0.18 |
| Copy when anchored ≥16, K=128 | 394K | 13.9 | +7.8% | 0.22 |
| Anchored ≥32, K=64 | 184K | 16.3 | +2.9% | — |
| Gated (copy only where it beats the drafter), K=64 | 176K | 26.5 | +12.0% | — |
| Gated, K=128 | 163K | 28.7 | +13.0% | — |

**Edit classes on their own tokens, K=64** (oracle conservative / realistic / realistic gated):

| Class | Oracle (m1) | Realistic | Gated |
| --- | --- | --- | --- |
| `edit_script` | +26% | +18% | +22% |
| `edit_patch` | +25% | +20% | +22% |
| `file_write` | +20% | +14% | +17% |
| `shell` | +18% | (not split) | |
| `prose` | +3% | | |

At K=128 the edit classes gain 2–3 points more.

## 4. Reading against the gate

P0's gate (draft §5): long copies must add **≥20% tokens per millisecond on agent sessions over DFlash2 alone**. M1's +4% was capped at 15 tokens, and P0 decides whether the cap was the reason.

- **The cap was a reason, not the reason.**
  - **Lifting it from 15 to 64** moves the realistic proposer from −0.4% to +6.8% and the oracle from +3% to +16%.
  - **Beyond 64** almost nothing comes: runs ≥128 hold 7.9% of the output, and K=128 adds +1 point over K=64.
  - **The limit is how often long runs occur** (14% of the output is in runs ≥64) and how well DFlash2 already does inside them.
- **Under the calibration M1 measured, the bar is not reached** unless a 64-row copy step costs ≤0.62× a production step (≤0.89× with every source at K=128); the weight read alone is ~70% of a step.
  - **The bar is met only on two assumptions:** the drafter does no better than 8.2 tokens per step on copied spans, *and* a 64-row copy step costs ≤1.5× a production step (oracle start, perfect choice of occurrence).
  - **No realistic proposer** we simulated comes within reach.
- **The reading is no-go for P1 as specified** (whole sessions ≥ +20%), and a narrow yes for an edit-only feature.
  - **The narrow yes:** a copy chain inside edit-shaped spans is worth +18–26% of those spans' decode time at K=64, if the 64-row step is nearly free.
  - **Its weight in a session:** edit scripts and patches are 7.8% of the output here (24% with file writes), so the session-level effect is a few percent.
- **The GPU half sizes the narrow yes; it cannot overturn the session-level no.** The oracle's edit gains hardly move with the step cost (edit scripts +26% at c = 1, +20% at 1.5), because copy steps are few. The realistic proposer's do move: edit scripts +18% at c = 1, +15% at 1.2, +10% at 1.5, because of its failed copy steps.

## 5. Caveats

- **SWE-bench, not interactive use.** This agent edits through Python `old = """…"""` / `new = """…"""` scripts: short spans copied from a raw re-read, not long patches or whole-file rewrites. Interactive sessions may rewrite larger blocks (copy-heavy) or write more new code (copy-poor); EfficientEdit's and Cursor's gains come from full-file rewrites, a workload absent here. Night 5's `apply_patch`-as-function arm will give the first real patch-heavy sample.
- **Harness prompts.** Every session shares the same system prompt, task wording and test commands. That inflates S1's and S4's short runs (call syntax is 73% copyable at ≥16) and barely touches ≥64.
- **The drafter's rate on copied spans** comes from M1 §6 at block 16 (up to 16 tokens per step); production runs block 12. With fewer draft rows the drafter's rate on long copies is lower, so the truth for production lies between the `m1_16` and `m1` columns, nearer `m1` on the 32+ spans that make up most of the copy-step savings.
- **Oracle start.** The oracle is told where a run begins; the realistic proposer is the honest lower side. A suffix-tree proposer with frequency scoring (SuffixDecoding) would sit between them; how far is not measured here.
- **The fluid model.** It ignores block granularity and the drafter's warm-up after a copy step, and treats a copy step's cost as c times a production step whatever the run length. A copy step needs no drafter pass (~11 ms of a ~116 ms step, M1 §5), so c below 1 is possible for short copies.
- **S3 and S4 are partial.** S3 is a proxy (what other agents read before editing); a checkout at the base commit was not available. S4 uses the first occurrence in another task only, over ~1,000 sessions, not a user's months of history in one repository.

## 6. The GPU half: script and steps

**Run on 2026-10-13 (A and B; C not run):** [moonshot-p0-rowsweep-2026-10-13.md](moonshot-p0-rowsweep-2026-10-13.md). c(64) ≈ 1.3–1.5 against a production step (1.33 by the prefill path, 1.48 extrapolated, at ~32K): "between" under the rule below, the edit-span feature marginal.

`bench/rowsweep.py` measures, on second-puffin with nothing else on the GPU:

- **A, the verify itself, rows path:** blocks 8, 12, 16, 24, 32 (`ling-run --spec --block B`, greedy, production's `--graphs --pdl`) at ~8K, ~32K and ~56K context, two repetitions. It reads the step's draft, verify and commit milliseconds.
  - **Why it stops at 32:** today's rows path stops at 32 rows (`kMaxStreamRows`), so 64 and 128 cannot be verified yet.
  - **The estimate:** the 16→32 per-row slope, extended to K rows.
- **B, N-row passes through the prefill path:** N = 32, 64, 96, 128 (`ling-run --chunk-check N`) on prompt pairs (4K, 12K), (24K, 40K), (48K, 64.5K).
  - **The cost of one N-row pass** at the pair's midpoint is ΔT/Δchunks, which cancels fixed costs.
  - **It is an upper bound:** this path also commits the DeltaNet state and writes the KV, and its GEMM is built for 2,048-row chunks.
- **C, the drafter's own rate on copied spans at block 12**, the number the go/no-go hinges on (8.2 or 10.9 tokens per step, §3, both from M1 at block 16). ling-serve keeps the context lookup in shadow mode by default and reports the drafter's accepted drafts per match-length bucket in `/spec_stats`. One replay of M0's main set at production's block 12 gives the 16–31 and 32+ rates.
- **The summary:** c(K) = copy step (verify + commit at K rows, no drafter) / production step (block 12: draft + verify + commit), both ways, beside this report's break-even values (`--stats`).

Prompts are filler built from this repository's own specs and sources; no recorded session leaves this machine. Two streams are not covered: a batched verify (M4 stage 3) is what makes rows cost something at batch 2, and it should be swept when it lands.

**Steps (≈1.5–2 GPU hours):**

```bash
ssh second-puffin
cd ~/ling-engine && git fetch -q && git checkout --detach origin/main && cmake --build build -j
# one model server on the machine: stop the reference server, say so in the chat, restart it after
docker stop m0-prod8000
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # must print nothing
python3 bench/rowsweep.py --build build --out /tmp/rowsweep --dry-run   # the plan
setsid nohup python3 bench/rowsweep.py --build build --out /tmp/rowsweep \
    > /tmp/rowsweep.log 2>&1 < /dev/null &
# when /tmp/rowsweep/summary.json exists:
docker start m0-prod8000
```

For C, the reference server still stopped:

```bash
bench/serve_bg.sh 18081                          # block 12, --graphs --pdl: production's settings
curl -s 127.0.0.1:18081/spec_stats > /tmp/c-before.json
python3 bench/replay/replay.py --url http://127.0.0.1:18081 --tools <tools.json> --tools-index <tools-index.json> \
    --sessions-file <main-set sessions> --out /tmp/c-replay.jsonl      # M0's main set, as M1 ran it
curl -s 127.0.0.1:18081/spec_stats > /tmp/c-after.json
python3 bench/replay/spec_report.py /tmp/c-before.json /tmp/c-after.json   # "shadow": per-bucket drafter rates
bench/serve_bg.sh stop 18081
```

Then, here: `scp second-puffin:/tmp/rowsweep/'*.log' DIR && python3 bench/rowsweep.py --summarize DIR --stats reports/moonshot-p0-copy-stats-2026-10-13.json`. The decision rule:

- **First, C.** If the drafter accepts ≤ 7.5 drafts per step in the 32+ bucket at block 12 (≤ 8.5 tokens per step), read §3's `m1_16` column; above ~9.5 drafts, the `m1` column. Between, interpolate.
- **A copy step at K=64 costs ≤1.0× a production step** (the extrapolated column at ~32K): the edit-span feature is worth +14–26% on those spans (realistic to oracle). Session-level stays below +20% (§4).
- **≥1.5×:** P0 closes no-go; the lookup stays off and the copy branch, if any, joins M2's tree at its own row budget.
- **Between:** the edit-span feature is marginal (+10–18% on edit spans, realistic); decide with night 5's patch-heavy sessions re-run through `copy_stats.py`.

## How to run

```bash
# the statistics (CPU, read only; ~15 min with 4 workers; RLIMIT_AS 4 GB per worker, 14 GB for S4)
nice -n 19 .venv/bin/python bench/copy_stats.py all --work /path/outside/repo --workers 4 > p0.json
nice -n 19 .venv/bin/python bench/copy_stats.py report --work /path/outside/repo   # re-aggregate only
```

`bench/copy_stats.py` reads `~/.local/share/dreamference/swe-bench/runs/*/scratch/*/codex-home/sessions/` (finished instances only) and the Qwen3.8 tokenizer from the Hugging Face cache. It writes token caches only under `--work`. The aggregate this report quotes is [moonshot-p0-copy-stats-2026-10-13.json](moonshot-p0-copy-stats-2026-10-13.json): counts and shares, no text.
