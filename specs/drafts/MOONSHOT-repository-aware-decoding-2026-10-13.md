# Moonshot: repository-aware decoding

*Draft, 2026-10-13. The owner asked for a night of reading on one question: how can decoding get faster, cleanly, by using what the repository and the session already contain? This is the result: a proposal, nothing built. Numbers measured on our machines say so; numbers from papers are the authors' own, read on the date above; everything else is marked as an estimate or unverified.*

**The thesis in one paragraph.** A coding agent mostly writes things that already exist: lines of the file it is editing (a patch's context and removed lines, a rewritten file), names and paths from the repository, earlier tool calls with small changes, its own output from earlier turns. A speculative engine can hand those spans to the target model to *verify* instead of *generate*, many tokens per pass. The output does not change at all: every proposed token is checked by the target, so the only thing that changes is how many tokens one weight read pays for. On a GB10, where a step is bound by reading 17.6 GB of weights and extra verify rows are almost free up to the compute bound, long verified copies are the cheapest tokens the machine can produce.

**"Clean" means four rules, kept throughout:**
1. Exact. Greedy output is identical to plain decoding; under sampling, the standard rejection rule with deterministic proposals (probability 1) keeps the target distribution, as for today's lookup branch ([SPECULATION §10](../DREAMFERENCE_LING_ENGINE_SPECULATION.md), "Exactness").
2. No approximate reuse of computed state. Position-independent KV reuse is excluded (§3.6).
3. Optional and ignorable from the client's side ([API §8](../DREAMFERENCE_LING_ENGINE_API.md), the speed-only `ling` extension, decided 2026-10-12).
4. Nothing leaves the machine. Every corpus is local, private and deletable; the product is air-gapped and this keeps it so.

---

## 1. What we already know here (measured)

- **Agents copy.** 46% of the visible tokens Mightling's agent writes continue a match in its own context ([SPECULATION §10](../DREAMFERENCE_LING_ENGINE_SPECULATION.md)).
- **DFlash2 copies well, inside its block.** In M1's shadow measurement ([reports/M1.md](../../reports/M1.md) §6, main set, block 16, 4,500 steps) the drafter beat the context lookup in every match-length bucket, even on 32-token-plus verbatim matches: 9.89 accepted against 6.87. A second branch was worth at most +4% (oracle).
- **But both were capped at the block.** That measurement scored the lookup's proposal up to B − 1 = 15 tokens, and DFlash2 can never accept more than its block (12 drafts in production since 2026-10-09, 16 in M1). A 300-token unchanged region of a file costs at least 20 steps today, however predictable it is. **Nothing we have measured says what happens when a proposal may run to 64 or 128 tokens.** That is the gap this spec is about.
- **The matcher is naive.** `lookup.cpp` takes the last 3 tokens and the most recent earlier occurrence. It has no longest-match structure, no frequency statistics, no view of text in the form the agent re-emits it (§3.2), and no corpus beyond the session.
- **Extra rows are nearly free, up to a point.** A step is ~108 ms, of which ~82 ms is the weight read at ~215 GB/s ([reports/M1.md](../../reports/M1.md), M4 stage 2). A 64-row verify needs ~3.5 TFLOP for the GEMMs (27B parameters × 2 × 64), ~13 ms at the prefill GEMM's measured 262 TFLOPS, so the GEMMs stay bound by the weight read. Attention and the DeltaNet recurrence grow with rows; their cost at 64 and 128 rows is to be measured (§5, P0).

## 2. What the literature says (read 2026-10-12/13)

| Work | What it does | Their result (self-reported) | What we take |
| --- | --- | --- | --- |
| **SuffixDecoding** (Oliaro et al., NeurIPS 2025, [arXiv 2411.04975](https://arxiv.org/html/2411.04975v3)) | Model-free drafting from two suffix trees: one over the request's prompt and output, one global over previous outputs; draft size adaptive, `MAX_SPEC = α·p` for a match of length p; falls back to EAGLE-3 below a score threshold | SWE-bench (OpenHands, 32B): 7.8 accepted per step against PLD's 3.2; 1.8–4.5× end to end over plain decoding; AgenticSQL 5.3×; ~20 µs per drafted token on CPU; 572M tokens in ~6 GB of host memory | Longest-match structure, frequency scoring, match-length-proportional draft length, a global corpus of past outputs |
| **AgSpec** (Oct 2026, [arXiv 2610.01108](https://arxiv.org/html/2610.01108v2)) | Retrieval drafting for coding agents over three corpora: session (prompts, tool outputs, generated text), workspace (files opened, indexed *as-is and in the agent's emission format*: each line prefixed with a space and with a minus for unified diffs, plus a JSON-escaped copy for whole-file tool calls), global (past trajectories); draft length `K = clip(α·m, 1, K_max)` with K_max profiled offline at break-even acceptance and α adapted online | 2.27–4.37× over plain decoding at batch 1, 1.08–4.76× at batch 16 (Qwen3.6-27B INT4 among the models, one RTX PRO 6000 Blackwell); emission-format indexing +13–39% acceptance in the first turn; session corpus 1.6–2.1× the acceptance of a per-call corpus; fixed K=32 was 14% faster than K=8 at batch 1 but 30% slower at batch 16 | **Emission-format views**, a session corpus that spans the whole session, the cost-aware length control, and the warning that long drafts hurt batched serving |
| **EfficientEdit** (ASE 2025, [arXiv 2506.02780](https://arxiv.org/html/2506.02780v2)); Cursor's "speculative edits" ([Fireworks](https://fireworks.ai/blog/cursor)) | The original code is the draft: verified in bulk; a rejection marks an edit; after new content, prefix matching finds where the original resumes | 5.4–13.1× over plain decoding on CanItEdit and CodeIF-Bench, where ordinary speculative decoding gave ~1.6–1.8×; Fireworks reported ~1,000 tok/s for Cursor's full-file rewrites | **Edit mode**: anchor on the file being edited, re-synchronise after each edit. (EfficientEdit's entropy-relaxed acceptance is lossy; we do not take it.) |
| **CopySpec** (EMNLP 2025, [arXiv 2502.08923](https://arxiv.org/html/2502.08923v1)), **SAM-Decoding** ([arXiv 2411.10666](https://arxiv.org/html/2411.10666v3)), **REST/CREST** ([arXiv 2311.08252](https://arxiv.org/pdf/2311.08252), [2408.04678](https://arxiv.org/pdf/2408.04678)), **Token Recycling** ([arXiv 2408.08696](https://arxiv.org/html/2408.08696v2)), **FastCoder** ([arXiv 2502.17139](https://arxiv.org/pdf/2502.17139)) | Copy from context (hash of the last γ tokens), longest suffix match with a suffix automaton (O(1) amortised per step), retrieval from a datastore, recycling the model's own candidates, repository-level retrieval | CopySpec gets *faster* with longer context and adds ~49% on top of a model drafter on redundant turns; SAM +3–11% on top of EAGLE-2 | A suffix automaton is the right per-session structure; copying composes with a model drafter |
| **Empirical study of SD on SE tasks** ([arXiv 2604.26469](https://arxiv.org/abs/2604.26469)) | Model-based and model-free methods across generation, editing, repair | Abstract: model-free methods fit repository-level repair and editing best, model-based ones generation; a review summary (unverified against the paper) gives Suffix Decoding ~2.3× on SWE-bench and ~1.5× on Aider Polyglot, and says acceptance does not translate linearly to speed on repository-level tasks | Measure tokens per millisecond, not acceptance |
| **STree** (NeurIPS 2025, [arXiv 2505.14969](https://arxiv.org/html/2505.14969v2)), **TreeWY** ([arXiv 2608.20961](https://arxiv.org/pdf/2608.20961)), **Trees from Marginals** ([arXiv 2607.06763](https://arxiv.org/abs/2607.06763)) | Tree verification for hybrid models whose recurrent layers (Gated DeltaNet, ours) cannot share state across branches: accumulated transition matrices, a tree WY transform with one triangular solve, rollback-free GDN tree verification in SGLang | Trees from Marginals: +24.7% over an optimised DFlash baseline, 4.37× over plain decoding | How to verify *several* proposals in one pass on our hybrid model (§3.5) |
| **HYPIC** ([arXiv 2607.01299](https://arxiv.org/abs/2607.01299)), **LinearKV** ([arXiv 2608.11231](https://arxiv.org/pdf/2608.11231)), **MiniPIC** ([arXiv 2606.13126](https://arxiv.org/pdf/2606.13126)) | Reuse cached KV and recurrent state of a file at a new position | HYPIC: 3.25× lower time to first token, with a 1.71-point quality gap from full recompute | **Excluded**: not exact (§3.6) |
| **Context-aware sparse head** (vLLM [#59740](https://github.com/vllm-project/vllm/pull/59740), [#60226](https://github.com/vllm-project/vllm/pull/60226)) | The drafter's output head over 32k fixed + 16k per-token rows | +5.9% decode on this model and box (MTP k=3) | Already planned ([BACKLOG §19.19](../DREAMFERENCE_LING_ENGINE_BACKLOG.md)) |

What the field has not done, as far as we found: long verified copies (64–128 tokens) on a bandwidth-bound single-node machine where the verify's extra rows are nearly free; copying for a *hybrid GDN* model at batch sizes of one and two; and a repository corpus supplied by the client of a local, air-gapped product. Those are ours to try.

## 3. Design

### 3.1 One proposer interface, five sources

Every source returns a chain (later a tree) of proposed tokens with a score, through one interface ([reports/backlog-2026-10-09.md](../../reports/backlog-2026-10-09.md) B36: nodes with parent indices and per-node q). The verify and the accept stay as they are.

| Source | Corpus | Lives | Built from |
| --- | --- | --- | --- |
| **S1 Session** | The resident session's whole token history: system prompt, every turn's input and tool output, every output | Engine, per resident session (M4's sessions make this natural: it persists across turns, unlike a per-request corpus, which AgSpec measured at 39–51% less acceptance) | Incrementally, as tokens are prefilled and accepted |
| **S2 Emission views** | For every file body that appears in the context (a tool output from reading a file), the token sequences the agent would emit when quoting it: lines prefixed with a space and with `-` (unified diff and `apply_patch` context and removed lines), the body JSON-escaped (a function-tool argument: the form `apply_patch` takes as a function tool, which night 5 tests on production), and as-is (a whole-file write) | Engine, beside S1 | Detected from the rendered context (a tool output that is a file read); each view tokenized as it would be emitted, so token boundaries match |
| **S3 Workspace** | Repository files the session has *not* read, and the repository's identifiers | Client → engine through `ling.lookup_corpus` ([API §8](../DREAMFERENCE_LING_ENGINE_API.md)); uploaded once per session and named by hash afterwards, cached by `prompt_cache_key` | The `ling` launcher, which runs where the repository is: `ling-code`'s symbol index for identifiers; file bodies on demand, size-capped |
| **S4 Personal history** | The user's own past outputs (all sessions, partitioned by repository), as SuffixDecoding's global tree | Engine host memory, LRU-capped (SuffixDecoding: ~6 GB for 572M tokens); written to disk under the user's home, never elsewhere | Online, after each finished request |
| **S5 Model** | DFlash2 (with the sparse head, §19.19, and the drafter tuned on the user's sessions, §19.20) | GPU, as today | — |

S1–S4 are CPU structures (suffix automaton per session, suffix tree for S4): microseconds per step, off the GPU's critical path; the proposer runs during the previous step's verify.

### 3.2 Why emission views matter for us specifically

Codex's `apply_patch` repeats the file's own lines with a one-character prefix (` ` for context, `-` for removed). Tokenizers split `" foo(bar)"` and `"foo(bar)"` differently, so the session's copy of the file (a tool output, unprefixed) does not match the patch the agent writes, token for token. A token-level matcher misses exactly the spans that matter most. AgSpec measured +13–39% acceptance from indexing the prefixed forms. Night 5 (2026-10-13) tests `apply_patch` as a function tool on production; in that form the patch travels as a JSON string: newlines become `\n`, quotes `\"`, and the same file line tokenizes differently again. S2 indexes each view in the form the template renders it, so the match is exact at the token level.

### 3.3 Edit mode (speculative edits)

When the output enters an edit-shaped span (inside an `apply_patch` call, a file write, a diff), the proposer switches to the file being edited (the last read of that path in S1/S2, else S3):
- **Anchor:** after an `@@` hunk header or the first context line, align to the matching line of the file.
- **Copy long:** propose the file's next lines in the emission view, up to the long-copy cap (§3.4).
- **Re-synchronise:** a rejection marks an edit; while the model writes new content, S5 drafts; when the output's last k tokens match a line of the file again, copying resumes from there (EfficientEdit's prefix matching, kept exact: we change only what is proposed, never what is accepted).

### 3.4 Long copies, and when to stop

The step's cost depends on the verify's row count; the rows' value depends on how many are accepted. The proposer chooses the draft length from both:
- **Length from the match.** As SuffixDecoding and AgSpec: proportional to the match length (`K = clip(α·m, 1, K_max)`), α adapted online from verify outcomes, per output class (edit, tool call, prose).
- **K_max from our cost model.** Measured on GB10 per row bucket (16, 32, 48, 64, then 96 and 128) at the session's context length and batch size: K_max is the largest bucket whose expected accepted tokens per millisecond beats the next smaller one. At batch 2, rows are no longer free (the batched verify, M4 stage 3), so K_max is lower, as AgSpec found at batch 16.
- **The verify for long copies.** Today's rows path stops at 32 rows (`kMaxStreamRows`); 64 and 128 need a mid-M GEMM between `stream_gemm` and the prefill GEMM, attention for 64–128 query rows over the session's KV (reading the KV once for all rows), and the DeltaNet recurrence over the accepted run (the prefill's chunked form, then commit).

### 3.5 Several proposals at once (trees, later)

A chain is exact and simple: pick the best single proposal per step (by expected accepted tokens) and verify it. A tree (S5's block plus a copy branch, or two copy candidates) needs tree verification through the Gated DeltaNet layers, whose recurrent state cannot be shared across branches. STree, TreeWY and Trees from Marginals show it can be done exactly (transition-matrix accumulation; one triangular solve over the tree; rollback-free verification in SGLang, +24.7% over DFlash). It is the same machinery M2 tree speculation needs, so the copy branch joins M2's tree rather than building its own.

### 3.6 What is deliberately left out

- **Position-independent KV reuse** (HYPIC, LinearKV, MiniPIC): prefilling a file once and reusing its KV and recurrent state at another position. It would cut time to first token sharply, but it is not exact (HYPIC reports a 1.71-point quality gap), and rule 2 forbids it. Exact prefix reuse (M4's resident sessions and checkpoints) stays the only reuse.
- **Relaxed acceptance** (EfficientEdit's entropy-aware top-τ): lossy. Out.
- **Changing what the agent writes to make it faster** (for example, preferring whole-file rewrites because they copy well): a product decision with quality consequences; it is measured (§4), not assumed.

## 4. The product side (Mightling)

- **`ling` supplies S3.** The launcher already runs `ling-code`; when the model server says `owned_by: ling-engine`, it sends the repository corpus in the `ling` extension, once per session, then by hash. Off for every other server.
- **S4 belongs to the user.** Stored under `~/.mightling/`, partitioned by repository, deletable with one command, never synced. The egress audit is unchanged (no new socket).
- **Edit format, measured for speed as well as quality.** Night 5 compares `apply_patch` as a function tool against freeform on production; with edit mode in place, a later night compares patch-style edits against whole-file rewrites for small files, on resolved tasks *and* wall time. The default changes only if quality holds.

## 5. Phases, gates and the moonshot target

| Phase | What | Cost | Gate to continue |
| --- | --- | --- | --- |
| **P0: measure, no build** | From recorded agent sessions (night transcripts, M0's replay sets, SWE-bench trajectories): the length distribution of runs that continue a match in S1, S2, S3 (the session's repository) and S4 (earlier sessions), per output class; oracle acceptance with K_max 15, 32, 64, 128; and the verify's cost per row bucket (16–128) at 8k, 32k and 64k context, one and two streams, on second-puffin | CPU for the corpus analysis; ~2 GPU hours for the row sweep | The oracle says long copies would add ≥ 20% tokens per millisecond on agent sessions over DFlash2 alone (M1's +4% was capped at 15 tokens; this decides whether the cap was the reason). **Dropped by the owner on 2026-10-13: GO** (below the table) |
| **P1: S1 + S2, chains, edit mode** | Suffix automaton per resident session with emission views; edit-mode anchoring; long-copy verify to 64 rows; cost-model K_max | Engine | On the replay: edit spans ≥ 2× tokens/s against DFlash2 alone; whole sessions ≥ +7%, output byte-identical (restated by the owner on 2026-10-13, the low end of P0's measured +7–13%; was ≥ +20%) |
| **P2: S3 + S4** | The `ling.lookup_corpus` extension end to end; the personal history tree | Engine + launcher | Acceptance beyond P1 on held-out sessions of a repository the user works in repeatedly |
| **P3: trees** | Copy branches inside M2's tree (GDN tree verification) | Engine (with M2) | Beyond the best single chain |
| **P4: 128-row copies and the batch rule** | Longer copies at batch 1, the K_max rule at batch 2 | Engine | Tokens per millisecond at both batch sizes |

**Decided (the owner, 2026-10-13): GO.** P0's statistics half ([reports/moonshot-p0-copy-stats-2026-10-13.md](../../reports/moonshot-p0-copy-stats-2026-10-13.md)) measured +7–13% on whole sessions (a realistic proposer +6.8% at K=64 and +7.8% at K=128, +12% and +13% with an oracle gate beside the drafter), short of the 20% bar above; the owner judged that gain good enough and dropped the 20% bar, so the moonshot goes ahead. The GPU half of P0 (`bench/rowsweep.py`, about 2 hours, still to run) is now design input, not a gate: it measures what a 64- and a 128-token check cost and how well the drafter predicts copied spans at 12 drafts. P1's whole-session gate is restated the same day as ≥ +7% with the output byte-identical (the low end of P0's +7–13%, in place of ≥ +20%); no later phase's gate repeats the 20% figure. Its place in the engine's order (the owner's decision, later the same day): P1's CPU parts are written now, on branch `moonshot/p1`; P1's GPU work (its GPU tests and first numbers) runs after the single-step savings block and before M4's stage 5, so stage 5 measures the savings and P1 together against production ([MILESTONES §14](../DREAMFERENCE_LING_ENGINE_MILESTONES.md), "Order").

**The moonshot target.** On edit-heavy agent sessions, the tokens of edits verified at 20–40 per step instead of ~5, and the session's decode time halved against DFlash2 alone. With the plan's other levers (the step savings, M2's tree, the tuned drafter), an agent that spends most of its output on edits would decode roughly 3× faster than today's production. This is an estimate built on two unmeasured quantities, how often long copies occur and what a 64-row verify costs here, which is why P0 comes first and costs almost nothing.

## 6. Risks

- **DFlash2 already copies most of what is copyable.** M1 showed it winning every bucket up to its block. If long runs are rare in our sessions, P0's oracle says so and the work stops at P0.
- **Long context taxes long verifies.** Verify attention grows with rows × context; the empirical SE study found acceptance stops translating into speed at long context. P0's row sweep at 32k and 64k context answers it for us.
- **Batched serving.** Rejected rows cost real time at batch 2. The K_max rule is per batch size; at worst, long copies are a batch-1 feature.
- **The recurrent layers.** A long accepted run must be committed through the DeltaNet recurrence (the prefill's chunked path); a rejected run must leave the state untouched. The exactness checks (`--spec-check`, `--prefix-check`, `--graph-check`) extend to long runs.
- **Tokenization at view boundaries.** A copied line can tokenize differently next to its neighbours; S2 indexes whole rendered spans, not line by line, and P0 measures how often boundaries break a match.
- **Privacy.** S1 and S2 die with the session; S3 is sent only to the local engine; S4 is the user's own and deletable. No corpus is shared between users of a node.

## 7. Where this sits in the plan

It extends [BACKLOG §19.20](../DREAMFERENCE_LING_ENGINE_BACKLOG.md) (more tokens per step: lookup per output class, edits, the repository corpus) and uses [API §8](../DREAMFERENCE_LING_ENGINE_API.md) (speed-only extensions). P0 is cheap and can run whenever second-puffin has two free GPU hours; P1 runs after M4's batching stages and the single-step savings, before M4's stage 5 (the owner's decision of 2026-10-13, section 5); P3 sits beside M2, whose tree verification it shares.

## 8. P1 as planned (2026-10-13, branch `moonshot/p1`)

*Status: the CPU half is built on `moonshot/p1`, not merged. It contains the proposer, its wiring into the speculative step behind `--copy` (off by default), and tests that run on the CPU. Nothing has run on the GPU. The GPU checks are written and wait for a free GPU (§8.6). §8.7 lists the choices left to the owner.*

### 8.1 Where proposals come from

- **S1 only in this build.** The corpus is the resident session's history plus the step's anchor: the system prompt, every turn's input and tool outputs, the earlier outputs, and this request's output so far. M4's sessions persist across turns, so the corpus spans the whole session, as AgSpec recommends.
- **The structure** is `CopyIndex` (`src/core/copy_proposer.hpp`), one per session:
  - It hashes every 8-gram that has a next token and keeps, per hash, the most recent end position, chained to the earlier ends with the same hash.
  - It is not a suffix automaton. It has the same constant-time lookup per step, plus two properties the engine needs. When a new turn re-renders the tail, the truncation is an exact undo, newest entries first. And the index's state is a function of the tokens alone, so a resumed or interleaved session proposes exactly what a fresh one would.
- **Selection:**
  - Among the 16 most recent occurrences of the last 8 tokens, it takes the one that agrees with the output furthest back (up to 256 tokens). Ties go to the most recent.
  - A proposal is made only when that agreement is at least 16 tokens. This is P0's realistic proposer, whose ≥32 variant measured worse.
  - The proposal is the occurrence's continuation, up to K_max.
  - A length rule `clip(α·m, …)` is implemented but off (α = 0), because P0 found the full length best.
- **Cost on second-puffin's CPU** (a 64K-token history):
  - 27 µs per step for sync and proposal (median; p99 30 µs). Most of it is the comparison that confirms the history only grew.
  - 3.3 ms to index 60K tokens from scratch.
  - 0.07 ms when a turn re-renders its last 400 tokens.
- **S2 (emission views): not built.**
  - P0 measured +0.1 to +0.3 points from S2.
  - The JSON view never occurs with this chat template.
  - The diff view matters only for `apply_patch`, which is 0.3% of the output.
  - Adding S2 later is cheap: `CopyIndex::propose(query, …)` answers a query against any corpus, so a view (or S3's corpus) is a second index asked with the session's tail. Decision D2 below.
- **S3 and S4** stay in P2, as planned.
- **Edit mode (§3.3): no separate mechanism.** The matcher re-synchronizes on its own:
  - A rejection is an edit.
  - While the model writes new text, the last 8 tokens find nothing and the drafter drafts.
  - Once the output's last 16 tokens match the file again, copying resumes from there.
  - P0's realistic proposer (+6.8%) is this anchor-free behaviour. Explicit `@@` anchoring would need span detection at the tokenizer level inside the engine (decision D3).

### 8.2 How a copy meets the DFlash2 drafter

Proposals are chains, not trees; trees come in P3, with M2. There are two modes, `--copy 1|2` (`EngineOptions::copy_mode`).

**Replace (1).**
- When the copy is longer than the drafter's chain (more than block − 1 tokens), the verify takes [anchor, copy] instead of the drafter's chain, and the drafter's pass is skipped (~11 ms of a ~116 ms step, M1 §5).
- The risk: a copy that fails early wastes the step. P0's realistic proposer had 15% of its copy steps accept nothing at K=64.

**Extend (2).**
- The drafter drafts as it always does. When its whole chain equals the copy's first block − 1 tokens, the copy's continuation is appended, up to K_max.
- A failed extension costs only the extra verify rows.
- It copies only where the drafter already agrees. That makes it a realistic stand-in for P0's "gated" proposer, which copies only where the copy beats the drafter (+12% at K=64, with an oracle gate).

**Limits and wiring:**
- **K_max is 31 tokens (32 rows)**, the rows path's limit (`kMaxStreamRows`). The 64-row verify is GPU work (§8.5, D4). At K=32, P0's realistic replace gains +4.9% at c = 1, against +6.8% at K=64. So the +7% gate needs either the 64-row verify or extend doing better than the realistic replace.
- **Batches:** a batched step never copies (P4's batch rule). Copies happen in one-sequence steps (`speculate()`), which is every step while one request runs.
- **Graphs:** each session's verify graph is now keyed by rows and top-K, instead of top-K alone, so a copy step also replays graphs. The draft graph is unchanged, and the commit graphs are still keyed by the accepted count.
- **The lookup:** copying takes precedence over `--lookup 2`. The lookup's shadow scoring leaves out steps that hold copied tokens, because it compares the lookup with the drafter alone.
- **Counters:** `/spec_stats` reports `copy_steps`, `copy_drafted` and `copy_accepted`, and `ling-run` prints them.

### 8.3 Exactness

- **Greedy: byte-identical.**
  - Every row is verified on the rows path (≤ 32 rows), which is row-invariant: each row's logits are bit for bit what plain decoding computes. So the accepted tokens are plain greedy's, for the same reason today's speculation is exact. `--spec-check` compares the tokens and the recurrent state.
  - A 64-row verify must keep this property. The prefill path's quantized GEMM is not row-invariant, so it cannot verify copies.
- **Sampled: the target's distribution.**
  - The rejection rule (`accept.cpp`) is unchanged. A copied token is a deterministic proposal (q = 1 on it), as the lookup's tokens are.
  - In extend, the drafted tokens keep the drafter's q, and the appended copy is a deterministic function of the drafted prefix, which the rule allows.
  - The proposal is computed from the history alone, before the drafter samples.
- **Seeded sampling: reproducible.**
  - The same seed and the same flags give the same bytes, because the index depends only on the history. Prefill splits, resumes, interleaving and graphs therefore cannot change a proposal.
  - Turning copies on changes which random draws are made, so the seeded sample changes, exactly as `--lookup 2` or a different block does today. The distribution does not change.
  - "Byte-identical" in P1's gate therefore means the greedy comparison, as for every earlier speculation change.

### 8.4 Built and tested on the CPU

**Code:**
- `src/core/copy_proposer.{hpp,cpp}`: the index and `extend_with_copy`.
- `accept_greedy` in `accept.cpp`, so the engine and the CPU simulation share one accept rule.
- `speculate.cpp`: `plan_begin` makes the copy proposal; `plan_draft` extends the chain; `plan_accept` uses q = 1 from the first copied position.
- `EngineOptions::copy_*`, and the flags `--copy`, `--copy-min`, `--copy-max` and `--copy-ngram` on `ling-run` and `ling-serve`.
- `ling_spec_host`, a CUDA-free library holding lookup, accept and copy, so the host tests build without the kernels.

**`ling-copy-tests`** (all pass, under 1 s):
- Proposals: longest agreement over most recent, the gates, the caps, the α rule, a side corpus.
- Incremental against fresh: random appends, truncations, regrowths and two-span syncs at n = 1, 3 and 8 must propose exactly what an index built in one go proposes.
- Extend's chain.
- Greedy end to end: a toy target over agent-like text (files read, then rewritten with edits) is decoded with the engine's step logic. The output must equal the text token for token in all three modes.
  - Toy tokens per step: drafter 5.1, replace 11.7–12.7, extend 6.4–6.6.
  - These are a check that copies happen, not a forecast: about 70% of the toy text is copyable, against 14% of real output in runs of 64 or more.
- Sampled: a chi-square test of the joint output distribution for replace chains of 3 and 6 tokens and for extend chains (1 drafted + 2 copied; 2 drafted + 3 copied). The test asserts that copied positions are actually reached, and a deliberately wrong q fails every case (checked by hand).
- Cost (§8.1).

**Existing host tests** (`ling-spec-tests`, `ling-sampling-tests`, `ling-api-tests`) still pass, and the whole tree builds (`ling-run`, `ling-serve`, the kernel tests).

### 8.5 What P0's GPU half settles for P1

`bench/rowsweep.py`'s A, B and C steps set P1's parameters.

- **c(K), what a copy step costs against a production step:**
  - The source: A measures 16, 24 and 32 rows; A's slope and B's upper bound give 64 and 128; all at ~8K, ~32K and ~56K context.
  - **It decides whether the 64-row rows path is built (D4).** P0's realistic replace gains +6.8% at c = 1, +4.0% at 1.2 and +0.1% at 1.5 (K=64). So build it if c(64) ≤ ~1.2 at the contexts sessions run at; stop at 32 rows if c(64) ≥ 1.5.
  - **It sets K_max per context length:** the largest row bucket whose accepted tokens per millisecond beats the next smaller one (§3.4).
- **C, the drafter's accepted drafts at block 12 on copied spans** (per match bucket: 8–15, 16–31, 32+):
  - **It sets the minimum copy length for replace.** Copy at agreement m only where the copy's expected accepted run beats the drafter's bucket rate times c(K). If the drafter makes ≥ 10 tokens per step on 16–31 matches, raise `--copy-min` to 32 for replace; at ≤ 8.5 keep 16.
  - **It sizes extend's reach.** Extend pays off where the drafter accepts its whole block on copied spans, so the share of full blocks in the 32+ bucket bounds it. `/spec_stats` does not split the accept histogram by bucket, so the first `--copy 2` replay measures it directly (`copy_steps` over `steps`).
- **Verify block sizes:** the drafter's block stays at 12 (production's). Copy chains use 32 rows now, then 48 or 64 if c allows.

**After the GPU half, the order:**
1. `bench/copy_exact.sh` (§8.6).
2. M0's main-set replay with `ling-serve --copy 0`, `--copy 1` and `--copy 2`, measuring tokens per millisecond per session and comparing the greedy outputs byte for byte with `--copy 0`. That replay is P1's gate (whole sessions ≥ +7%), with edit spans read separately by output class.
3. The 64-row verify, if c(64) allows.

### 8.6 GPU checks written, not run

`bench/copy_exact.sh` (ling-run, behind `flock ~/gpu.lock`, one model on the machine). Prompts come from `bench/copy_prompts.py`: this repository's files with a small edit asked for, so the greedy answers are mostly long copies.

- **`regress`:** today's `--spec-check` and `--batch-check` without `--copy`. The default path must be unchanged, including the verify graphs' new key.
- **`spec`:** `--spec-check --copy 1|2`, eager and with `--graphs --pdl`, on the copy prompts and on the M3 set (greedy tokens and recurrent state equal to plain decoding).
- **`graph`:** `--graph-check --copy 1|2` (graphs against eager, greedy and sampled; the verify graphs of copy steps' row counts).
- **`prefix`, `chunk`, `interleave`:** `--prefix-check --sessions 2`, `--chunk-check 256` and `--interleave-check --sessions 2`, each with `--copy 2`. These check that the index gives the same proposals after resumes, chunked prefills and interleaving (same steps, same accepted drafts).
- **`batch`:** `--batch-check` without `--copy`. A batch never copies, so with `--copy` the runs alone would copy and their steps would differ by design.
- **Not yet written:** the existing GPU test binaries (`ling-kernel-tests`, `ling-stream-tests`, `ling-prefill-tests`) need re-running as a regression (no kernel changed), and `tests/server` against a `ling-serve --copy 2`.

### 8.7 Open for the owner

- **D1. The default mode for the gate replay.**
  - (a) replace;
  - (b) extend;
  - (c) both: replace on agreement ≥ 64, extend below.
  - *Recommended: run (a) and (b) on the replay and choose by tokens per millisecond.* Expectation: extend is the safer default, because its loss is bounded by the extra rows' cost, which A measures. Replace can do more on long runs but pays for its early failures. (c) is cheap to add once the per-bucket numbers exist.
- **D2. S2 (emission views) in P1.**
  - (a) leave it out;
  - (b) the ` `/`-` diff view only;
  - (c) all three views, as §5 planned.
  - *Recommended: (a).* P0 measured +0.1 to +0.3 points in total. Revisit with night 5's `apply_patch` transcripts through `bench/copy_stats.py`.
- **D3. Explicit edit mode (`@@` anchoring, §3.3).**
  - (a) none: the matcher's re-synchronization covers it;
  - (b) build it now.
  - *Recommended: (a) until the replay's edit classes show copies the matcher misses.* (b) needs span detection at the tokenizer level inside the engine.
- **D4. Verifies beyond 32 rows.**
  - (a) extend the rows path to 64 rows: the stream GEMM's M tile, the row-invariant attention for 64 query rows, the step buffers and the commit's saved inputs. This keeps exactness bit for bit.
  - (b) stay at 32 rows.
  - (The prefill path is excluded: it is not row-invariant.)
  - *Recommended: (a) if c(64) ≤ ~1.2× a production step; otherwise (b).* At K=32 the realistic replace is at +4.9%, short of the gate.
