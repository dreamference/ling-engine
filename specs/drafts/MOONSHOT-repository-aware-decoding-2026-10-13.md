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
| **P0: measure, no build** | From recorded agent sessions (night transcripts, M0's replay sets, SWE-bench trajectories): the length distribution of runs that continue a match in S1, S2, S3 (the session's repository) and S4 (earlier sessions), per output class; oracle acceptance with K_max 15, 32, 64, 128; and the verify's cost per row bucket (16–128) at 8k, 32k and 64k context, one and two streams, on node-b | CPU for the corpus analysis; ~2 GPU hours for the row sweep | The oracle says long copies would add ≥ 20% tokens per millisecond on agent sessions over DFlash2 alone (M1's +4% was capped at 15 tokens; this decides whether the cap was the reason). **Dropped by the owner on 2026-10-13: GO** (below the table) |
| **P1: S1 + S2, chains, edit mode** | Suffix automaton per resident session with emission views; edit-mode anchoring; long-copy verify to 64 rows; cost-model K_max | Engine | On the replay: edit spans ≥ 2× tokens/s against DFlash2 alone; whole sessions ≥ +20%; byte-identical output |
| **P2: S3 + S4** | The `ling.lookup_corpus` extension end to end; the personal history tree | Engine + launcher | Acceptance beyond P1 on held-out sessions of a repository the user works in repeatedly |
| **P3: trees** | Copy branches inside M2's tree (GDN tree verification) | Engine (with M2) | Beyond the best single chain |
| **P4: 128-row copies and the batch rule** | Longer copies at batch 1, the K_max rule at batch 2 | Engine | Tokens per millisecond at both batch sizes |

**Decided (the owner, 2026-10-13): GO.** P0's statistics half ([reports/moonshot-p0-copy-stats-2026-10-13.md](../../reports/moonshot-p0-copy-stats-2026-10-13.md)) measured +7–13% on whole sessions (a realistic proposer +6.8% at K=64 and +7.8% at K=128, +12% and +13% with an oracle gate beside the drafter), short of the 20% bar above; the owner judged that gain good enough and dropped the 20% bar, so the moonshot goes ahead. The GPU half of P0 (`bench/rowsweep.py`, about 2 hours, still to run) is now design input, not a gate: it measures what a 64- and a 128-token check cost and how well the drafter predicts copied spans at 12 drafts. The later phases' own gate lines are left as written. The engine's order is unchanged ([MILESTONES §14](../DREAMFERENCE_LING_ENGINE_MILESTONES.md), "Order").

**The moonshot target.** On edit-heavy agent sessions, the tokens of edits verified at 20–40 per step instead of ~5, and the session's decode time halved against DFlash2 alone. With the plan's other levers (the step savings, M2's tree, the tuned drafter), an agent that spends most of its output on edits would decode roughly 3× faster than today's production. This is an estimate built on two unmeasured quantities, how often long copies occur and what a 64-row verify costs here, which is why P0 comes first and costs almost nothing.

## 6. Risks

- **DFlash2 already copies most of what is copyable.** M1 showed it winning every bucket up to its block. If long runs are rare in our sessions, P0's oracle says so and the work stops at P0.
- **Long context taxes long verifies.** Verify attention grows with rows × context; the empirical SE study found acceptance stops translating into speed at long context. P0's row sweep at 32k and 64k context answers it for us.
- **Batched serving.** Rejected rows cost real time at batch 2. The K_max rule is per batch size; at worst, long copies are a batch-1 feature.
- **The recurrent layers.** A long accepted run must be committed through the DeltaNet recurrence (the prefill's chunked path); a rejected run must leave the state untouched. The exactness checks (`--spec-check`, `--prefix-check`, `--graph-check`) extend to long runs.
- **Tokenization at view boundaries.** A copied line can tokenize differently next to its neighbours; S2 indexes whole rendered spans, not line by line, and P0 measures how often boundaries break a match.
- **Privacy.** S1 and S2 die with the session; S3 is sent only to the local engine; S4 is the user's own and deletable. No corpus is shared between users of a node.

## 7. Where this sits in the plan

It extends [BACKLOG §19.20](../DREAMFERENCE_LING_ENGINE_BACKLOG.md) (more tokens per step: lookup per output class, edits, the repository corpus) and uses [API §8](../DREAMFERENCE_LING_ENGINE_API.md) (speed-only extensions). P0 is cheap and can run whenever node-b has two free GPU hours; P1 belongs after M4's batching stages, beside M2, whose tree verification P3 shares. Nothing here changes the order the owner set on 2026-10-12 until P0's numbers are in.
