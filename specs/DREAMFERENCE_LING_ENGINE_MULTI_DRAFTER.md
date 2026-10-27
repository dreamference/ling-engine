# ling-engine: Several drafters, one per kind of work

Part of the ling-engine specification; the index is [README.md](./README.md). Written on 2026-10-27 at the owner's decision of that day: **Mightling 2.0 ships several speculative drafters, each tuned for a different kind of work.** This document is the specification. Nothing has been trained or built yet. It has no section number in the former `SPEC.md`.

It builds on:

- [BIG_DRAFTER](./DREAMFERENCE_LING_ENGINE_BIG_DRAFTER.md): a bigger drafter is no-go, and tuning today's size is the lever;
- [SPECULATION §10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md) and [SESSIONS §11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md);
- [BACKLOG §19](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) items 15 (speed-only extensions), 19 (sparse head), 20 (the agent-tuned drafter), 26 (tree verification) and 35 (the prefix KV store);
- the memo on training the drafter on agent sessions (`dgxcoder-memos/drafter-agent-sessions.md`, outside this repository);
- Mightling's `specs/DREAMFERENCE_MODELS.md` (the registry and the drafter pin).

## 0. Summary

- **Output never changes.** A drafter only proposes tokens; the target verifies them. Under greedy decoding a draft is accepted only if it equals the target's argmax, so the text is identical whichever drafter runs, or none. Under sampling the rejection rule keeps the target's distribution. A drafter is purely a speed choice. §2.6 says how this is tested.
- **Two specialists, one fallback.** 2.0 ships three drafters: the stock `general` drafter (today's), a **`work`** drafter for coding-agent turns, and an **`ask`** drafter for prose (Ask, phone answers, thinking, long-document answers).
  - JSON tool arguments and structured output need no drafter of their own: they are already the fastest class, and they make up most of the agent's output, so `work` covers them.
  - A drafter per programming language is not proposed until a measurement shows a gap (§1).
- **The evidence is a gap, not a cause.** Prose runs at about half the speed of code on this model on every engine where it was measured (§1.1), but the engine has no measured prose acceptance of its own.
  - Part of that gap is the target's own entropy, which no drafter can remove.
  - The stock drafter was trained on short, target-regenerated chat. So the larger tuning gain is expected where Mightling's traffic is *out of* that distribution: agent turns and long contexts.
  - Phase 0 measures the split before anything is trained.
- **Picking a drafter: declared first.** The client declares its workload per request: Ask, Work, phone, SWE-bench or Night Shift. The hint travels in BACKLOG 15's `ling` object, or in Codex's per-turn metadata, which Mightling's app-server clients can already set.
  - A rule on the request's own shape covers clients that declare nothing.
  - Switching inside a turn and online switching on acceptance come later, and only if the declared mode leaves a measured gap.
- **Engine cost.**
  - Each resident drafter takes 1.61 GB (measured in BIG_DRAFTER §3). Two extra drafters add 3.2 GB to the engine's budget; the 16 GB host reserve stays as it is.
  - The drafter's context KV ring is per drafter, and the target features it is built from are not kept. So 2.0 switches drafters **per session**, at request boundaries, which in Mightling means almost never.
  - Switching within a turn needs a per-session window of target features (105 MB per session). That is phase 4.
- **Training: local by default.** Each specialist is trained on about 20M tokens that our own target writes for public tasks, never the owner's sessions.
  - On one GB10 that is about 3-5 nights of generation and 2-4 nights of training per drafter, with the model server stopped for the training.
  - Renting a GPU instead is an owner decision. It costs tens of dollars per drafter.
- **Shipping.**
  - **The bar:** a specialist ships only if it beats `general` by at least 5% tokens/s on its own held-out set, no class is more than 3% worse, and every exactness gate passes. The general case may not slow down.
  - **Distribution:** release assets pinned by SHA-256 (the hosting decision of 2026-10-27), listed in the model registry.

## 1. Which specialisations pay

### 1.1 What is measured

| Source | Engine and drafter | Code or agent | Prose | Other |
|---|---|---|---|---|
| ling-serve, agentlong (16 agent turns at 25-64K, one stream; BIG_DRAFTER §1) | DFlash2 NVFP4, 12 drafts, tree of 16 rows | **6.7 tokens/step greedy** (65.5 tok/s); 5.5-6.2 sampled | not measured | — |
| Production SGLang, short context, greedy (OVERVIEW §3, Mightling MODELS §2) | DFlash2, 16 drafts | 50.3 tok/s | **25.5 tok/s** | JSON **87.0 tok/s** |
| Community DFlash2 figures (SPECULATION §10) | DFlash2 | ~6.0 tokens/step (math) | **~3.3 tokens/step (long prose)** | — |
| MiaAI-Lab, SGLang on a GB10 (`reports/github-survey-2026-10-09.md`) | DFlash2 | 50.9 tok/s | 25.4 tok/s (long essay) | — |
| hasso5703, SGLang on a GB10, thinking on (same survey) | DFlash2 | agentic coding 40 / 34 tok/s | EN / FR / DE 22 / 20 / 18 tok/s | — |
| DFlash paper, Table 6 (BIG_DRAFTER §4), 5 layers | DFlash | HumanEval 4.94 tokens/step | MT-Bench 3.37 | Math500 5.99 |
| AMD box, llama.cpp, the model's own MTP with 4 drafts (Mightling `specs/DREAMFERENCE_MIGHTLING_AMD_NODES.md`) | MTP, a different drafter and engine | **80% acceptance**, 23.7-28.6 tok/s | **37% acceptance**, 14.1-14.6 tok/s | — |

Every source tells the same story. Prose decodes at about half the rate of code (0.45-0.55×), and JSON at about 1.7×. Thinking traces count as prose: with heavy thinking, the agent's output becomes prose-like, and acceptance falls toward the prose figure ([OVERVIEW §7](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)).

**What is not measured.** The engine has no prose or chat acceptance of its own: every engine number is agentlong, which is coding-agent turns. Its 6.7 already includes tree verification and the agent's copying, and the lookup branch feeds on that copying (46% of visible output continues a match in the context; OVERVIEW §2). So the 6.7-against-3.3 comparison mixes engines and workloads. Phase 0 (§6) measures the engine's own split.

### 1.2 Why a gap is not yet a gain

Acceptance is bounded by two things:

- **The target's entropy.** If the target itself hesitates between continuations, no drafter can predict its argmax reliably. That part of the prose gap is irreducible.
- **The drafter's mismatch with the target** on that kind of text. Only this part can be tuned away.

The stock drafter (`incoai/Qwen3.8-27B-DFlash2`) was trained by SpecForge's recipe on about 1.07M **target-regenerated conversations of about 4.5K tokens** (BIG_DRAFTER §6). That means:

- **Short generic chat is in its distribution already.** An `ask` drafter trained on more of the same would gain little there.
- **Mightling's agent turns are out of its distribution.** Their format is the tool-call wrapper, `exec_command` heredocs, patches and status lines, at 25-64K tokens of context. That is the memo's case for +5-15% from tuning.
- **Long contexts are out of its distribution for every class.** The drafter's own attention sees a window of 2,048 keys, but it conditions on the target's features, and the target's hidden states at 30K tokens were never in its training data. BACKLOG item 10 records DFlash falling from 71 to 16 tok/s at 185K tokens on vLLM.
- **Mightling's prose is not generic chat.** Ask and phone answers follow Mightling's prompts, quote `ling-search` results and local documents, write Markdown, and often run with thinking on. That is the case for `ask`, and it is weaker than the case for `work`.

The literature agrees on the direction. arXiv 2503.07807 measured a fixed generic drafter's acceptance falling by 4-38% when the *target* was replaced by a domain-tuned one (function calling, biology, Chinese). That is a bigger shift than ours, where the target stays and only the traffic changes. What transfers here is the paper's data finding: distillation on the target's own outputs recovers the loss. Data the target generates for synthetic prompts gives 80-93% of the gain of training on real user queries, and offline distillation beats online distillation by 11-25%. MetaSD (arXiv 2604.05417) selects among several domain drafters as a multi-armed bandit and beats every single drafter, which is the online selection of §2.4.

### 1.3 The candidates

| Candidate | Evidence | Verdict |
|---|---|---|
| **Coding-agent turns** (`work`): shell, edits, tool-call syntax, status text | out of the stock training distribution in format and context length; the memo's +5-15%; tool-call arguments are 71% of visible output (OVERVIEW §2) | **Build first.** It is also the single-drafter upgrade the memo recommends (option b), so it pays even if nothing else in this document is built. |
| **Prose** (`ask`): Ask, phone answers, thinking traces, answers over long documents | the largest speed gap (0.5× code), but partly entropy; out of distribution at long context and in Mightling's formats | **Build second, behind a pilot gate.** It ships only if the pilot shows a real gain (§6, phase 3). Long-document reading and thinking are part of its training mix, not drafters of their own. |
| **Structured output and JSON tool arguments** | already the fastest class (87 tok/s); grammar-constrained decoding (BACKLOG 17) masks every verify row, so the accepted tokens are determined by the grammar plus the target | **No drafter of its own.** Tool-call JSON is in `work`'s data, and `response_format` JSON in `ask`'s. Reconsider only if phase 0 shows `call_syntax` or JSON accepting below the agent average. |
| **Per programming language** | no measurement. The agent's traces are mostly Python (SWE-bench). Code in other languages is also prose-like to the drafter in places (comments, strings). | **Not proposed.** Phase 0 splits `work` acceptance by language on a multilingual held-out set (SWE-rebench has several languages). A per-language drafter is reconsidered only if a language with real traffic accepts at least 10% below Python after `work` is trained. |
| **Natural language other than English** | the hasso5703 figures show prose falling from English (22 tok/s) to French (20) to German (18); the static sparse head cost −33% on multilingual prompts (BACKLOG 19) | Part of `ask`'s data mix, weighted by the languages the owner expects (open question 5). Not a drafter of its own. |

## 2. How the engine picks a drafter

### 2.1 The workload hint

The client says what kind of work a request is. The values are `work`, `ask`, `phone`, `swe-bench`, `night` and `general`. The hint follows BACKLOG 15's rules for speed-only extensions:

- it is optional;
- it never changes the output;
- it is sent only to a server whose `/v1/models` says `owned_by: ling-engine`;
- unknown values are ignored.

**Two carriers:**

1. **`"ling": {"workload": "ask"}`** in the request body, on `/v1/responses`, `/v1/chat/completions` and `/v1/completions` alike. This is the documented form, for any client.
2. **Codex's per-turn metadata.** Every Responses request Codex sends carries `client_metadata`, including the `x-codex-turn-metadata` JSON blob. App-server clients can add their own keys to that blob per turn (`responsesapi_client_metadata`, which lands under the blob's extra fields; Codex keeps its reserved keys to itself). So `ling web`, the desktop app and the phone bridges can set `ling_workload` per thread and per turn without a Codex patch. `ling exec`'s unattended callers (SWE-bench, Night Shift) set it the same way, or through the launcher's config.
   - The exact key path is checked against the pinned Codex at build time.
   - `client_metadata` is listed as "accepted and ignored" today ([API §3](./DREAMFERENCE_LING_ENGINE_API.md)), so reading it is an API change, recorded there in phase 2.

**Workload to drafter.** The mapping is a table in the registry's recipe (§5.2), not code:

| Workload | Drafter |
|---|---|
| `work`, `swe-bench`, `night` | `work` |
| `ask`, `phone` | `ask` |
| `general`, or none | the rule below, else `general` |

**SWE-bench and A/B runs must pin the drafter.** The harness sends the workload explicitly on every request, and the report records which drafter served each one (§2.5). Otherwise a run compares drafters without saying so.

### 2.2 Requests that declare nothing

Cline, Continue, OpenHands and plain `curl` send no hint. A rule on the request's own shape decides, and it is recorded in `/spec_stats` as `inferred`:

- tools offered and the last input item a tool result: **`work`**;
- tools offered and the last item a user message: **`work`** if the tools include a shell or patch tool, else `general`;
- no tools: **`ask`**, once `ask` exists and has passed its gate; else `general`.

A small learned classifier (a few features of the rendered prompt, microseconds on the host) replaces the rule only if phase 0's replays show the rule misroutes more than 10% of requests. A wrong guess costs speed, never correctness.

### 2.3 When the drafter changes, and what a change costs

**What a session holds per drafter.** Each resident session keeps the drafter's context K and V for the last 2,048 positions: a ring of 5 layers × 2,048 × 8 KV heads × 128 × K and V × 2 bytes, about **42 MB** per session (`dkc_`, `dvc_`; positions `[dkv_lo, history)`, the last `ring_` of them).

- The ring is projected from the target's features at layers 5, 19, 33, 47 and 61. Those features exist only during the pass that wrote them: `cap_` holds one pass's rows, not the window.
- So the ring belongs to the drafter that built it. A different drafter has nothing to rebuild its own ring from, short of running the target again.
- The prefix KV store and the checkpoints already hold no drafter context (BACKLOG 35): a resumed prefill rebuilds the ring from its own pass. They are unaffected by having several drafters.

**Three ways to switch:**

| Option | How | Cost | When it pays |
|---|---|---|---|
| (a) A ring per drafter | every commit projects the accepted rows into every resident drafter's ring | +0.8 ms per step per extra drafter, always (the commit is 0.81-0.84 ms today), and 42 MB per session per drafter | never at 2.0's switch rate: it charges every step for switches that almost never happen |
| (b) A window of target features | each session keeps the last 2,048 positions' features (5 × 5,120 × BF16 = 51 KB per token, **105 MB per session**); a switch builds the new drafter's ring from them in one pass (the fc and five K/V projections over 2,048 rows, about 0.75 TFLOP, **an estimated 10-20 ms**, measured in phase 2) | 105 MB per session, a copy of the accepted rows' features per step (microseconds), one pass per switch | switching inside a turn (phase 4) |
| (c) Rebuild at the boundary | a session that changes drafter re-prefills the last 2,048 tokens from the nearest checkpoint, as a resume past the window does today | at least 0.7 s once (2,048 tokens at ~3,000 tok/s), more when the nearest kept state lies before the window | rare switches: **2.0** |

**2.0 uses (c), and binds the drafter per session.**

- A new session takes the drafter its first request names. Its prefill builds that drafter's ring as it does today, so a new session costs nothing extra, prefix-store hits included.
- A Mightling thread keeps its workload across turns (a Work thread stays Work), so a resident session changes drafter only if a later request names another. Each such change is counted in `/spec_stats`.
- Option (b) is built in phase 4, and only if switching inside a turn has earned it (§2.4).

**What else changes with several drafters:**

- **CUDA graphs.** The step graphs are per session and bake in the drafter's weight and ring pointers, so each drafter gets its own draft and commit graphs. They are captured at first use, and the verify graphs are shared. Graph memory is small; phase 2 measures the capture time.
- **Batches.** `draft_batch_launch` runs one drafter over all the sequences in a batch. When two sequences that decode together use different drafters, the step runs one draft pass per drafter: about +8 ms on a step of 110-120 ms. Every sequence's result stays bit for bit what it gets alone, as the batch checks already require. The scheduler prefers to pair sequences that use the same drafter; phase 2 measures whether that preference is worth having.
- **Tree verification.** The tree is built from the drafter's lattice, and its knobs (`LING_TREE_T`, `LING_TREE_CHAIN`, the 16 rows) were tuned on the stock drafter. They become per-drafter values in the drafter's manifest (§5.1), defaulting to today's. The verify kernels do not change.
- **The sparse head and the lookup branch.** The sparse head is factors of the target's head, so it is shared by every drafter of the same shape. The lookup branch does not depend on the drafter. Neither changes.

### 2.4 Later: switching inside a turn, and online selection

- **At span boundaries (phase 4).** A Work turn may think in prose, then write a tool call. The token history tells the engine deterministically where `<think>`, `</think>`, `<tool_call>` and message text begin, and the engine can switch drafter there (`ask` for thinking and message text, `work` for calls) with option (b). This is worth building only if phase 0's per-class split shows that `work` accepts thinking or message text at least 10% worse than `ask` does, and if those spans are a large enough share of Work output to move its tokens/s by at least 3%.
- **On acceptance (phase 4, after spans).** A bandit over the resident drafters, rewarded by each step's accepted tokens, as in MetaSD. Measuring the drafters it does not use would mean running them too (+8 ms a step each), so the bandit learns only from the drafter it runs, with periodic exploration.
- **The rule that keeps seeded runs reproducible.** Every selection is a function of the request's own token history and hint. It never depends on the batch partner, the clock or another session's counters. A seeded request replayed with the same hint then picks the same drafters at the same steps.

### 2.5 What the engine reports

- `/spec_stats`: the counters per drafter (drafted, accepted, steps, draft and commit ms), the switches, and how each request's drafter was chosen (`declared`, `inferred`, `default`).
- `/metrics`: `ling:spec_drafted_tokens_total` and `ling:spec_accepted_tokens_total` gain a `drafter` label.
- A response header, `x-ling-drafter: work`. It is outside the body, so tokens, logprobs and usage stay what they would be without the hint.
- `bench/agentlong/run.py` and the new sets record the drafter for each turn, and `paired.py` refuses to compare arms whose drafters differ unless that difference is the arm.

### 2.6 Exactness, and how it is tested

- **Greedy.** A drafted token is accepted only if it equals the target's argmax. The output is therefore token-identical to plain decoding with every drafter, with any switch, and with no drafter at all.
- **Sampled.** The accept kernel's rejection rule uses the drafter's own probabilities, so the target's distribution is preserved whichever drafter proposed, including the tree's siblings (checked against recursive rejection sampling, BACKLOG 26 follow-up 3).
- **One caveat, about seeds.** A given seed's draw can differ between drafters, because the accept consumes uniforms differently. That is already true today between drafters, tree shapes and draft blocks. The rule in §2.4 makes the choice reproducible for a fixed hint.

**Tests (phase 2):**

1. `ling-run --spec-check` and `--graph-check` with each shipped drafter: token identity against plain decoding on the fixed prompt set (code, prose, JSON, arithmetic, an 11K-token replay).
2. **`ling-run --switch-check`** (new, modelled on `--interleave-check`). Decode each prompt with drafter A, switch to B at a request boundary (option c), and at step *k* ∈ {1, the middle, a `</think>`} once option (b) exists. Check that:
   - the tokens equal plain greedy;
   - the target's state (KV, DeltaNet, conv) is bit for bit that of a run that never switched;
   - B's rebuilt ring is bit for bit the ring B builds when it runs from the start. If the 2,048-row rebuild and the per-step commits disagree in the last bit, report it and require only token identity, because the ring changes proposals and never acceptance.
3. `--interleave-check` and `--batch-check` with the two sessions on different drafters.
4. The sampled exactness check of BACKLOG 26's follow-up 3, rerun with each drafter and with a switch.
5. The VALIDATION §13 exactness set (500 prompts plus 500 replays), run once per shipped drafter before release.

## 3. Training data and cost per drafter

### 3.1 Data

Every training token is written by **our own served target** (Qwen3.8-27B NVFP4, Mightling's template and prompts) for a **public** prompt or task. So the drafter learns the distribution it will meet, and no private text enters a shipped file.

| Drafter | Sources | Shape |
|---|---|---|
| `work` | `ling exec` sessions on **SWE-smith** (MIT, 50K instances) and **SWE-rebench** (CC-BY-4.0, each repository's licence recorded per instance), through the SWE-bench harness. Instances after a cut-off date are held out for evaluation. Some sessions run with thinking on, so the drafter sees what Work looks like with effort raised. | about 12K written tokens per session at 25-64K of context |
| `ask` | public chat and instruction prompts; public long-document question-answering sets (the document as context, 8-64K tokens); search-style questions answered through `ling-search` in a sandbox with recorded results; phone-style short questions; non-English prompts in the languages the owner names. Candidate sets include oasst2 (Apache-2.0), Dolly 15k (CC BY-SA 3.0) and WildChat (ODC-BY). Phase 1 fixes the list and records every set's licence before use, keeping only sets whose licence allows training weights that are then distributed. About 30% of samples have thinking on. | short prompts with answers of 300-1,500 tokens; long-document samples capped at 20% of the tokens, because their prefill dominates the cost |

**The owner's own sessions are never used**, unless the owner later opts in (the memo's option d). Such a drafter is trained and kept on the owner's machines only, and never ships: not in a release, not in `node sync-model`, not in adapter publishing. The standing rule of 2026-10-09 applies.

The SWE-bench Verified nights are not used for `work`'s training. Verified is the benchmark, and a drafter tuned on it would make the night's tok/s a measurement of the training set (Mightling's benchmark choice of 2026-10-09).

### 3.2 Recipe

1. **Warm start from the stock drafter** (BF16 `incoai/Qwen3.8-27B-DFlash2`): same shape, same block, same selector.
2. **Supervised distillation** on the target's outputs, with the target's features captured online. ling-engine already writes the five layers' features (`cap_`) during every pass, so the target and the trainer share the GPU and no second runtime is needed (BIG_DRAFTER §6).
3. **Then on-policy distillation (Draft-OPD, arXiv 2605.29343)** from the verify's rejected rows. The engine already dumps a step's candidates and lattice (`LING_DUMP_STEPS`) and hidden states (`LING_DUMP_HIDDEN`). The memo reports +11-13% for this over plain supervised fine-tuning.
4. **NVFP4 re-quantization with the ScaleSearch scale search** (BACKLOG 22), as the served drafter is made today.

### 3.3 Volume and local cost

The volume is **20M written tokens per specialist** (a 2M-token pilot first, §6). That is between the few million that published same-size adaptations use and the 30-100M BIG_DRAFTER estimates for growing a drafter, which has to learn new parameters from nothing.

Measured rates on node-b's GB10 (BIG_DRAFTER §6):

- the trainer runs at **3,438 tokens/s** at 5 layers (30 GB peak);
- online feature capture runs at about **2,400 prefill tokens/s**;
- generating agent sessions yields about **180K written tokens per stream-hour**.

Features cannot be stored at 51 KB per token (20M written tokens of agent data would need about 5 TB), so every epoch prefills its transcripts again.

| Per drafter, 2 epochs | `work` | `ask` |
|---|---|---|
| Generation (model server up; can run on idle days) | 20M ÷ (4 streams × 180K/h) ≈ 28 h, **3-4 nights** | prose decodes slower and in parallel requests: ~100-200 tok/s aggregate, 28-56 h, **4-7 nights** |
| Feature prefill per epoch | ~1,700 transcripts × ~60K tokens ≈ 100M tokens, ~12 h | ~60M tokens (long documents capped), ~7 h |
| Training per epoch | 20M ÷ 3,438/s ≈ 1.6 h | 1.6 h |
| Training total, model server stopped | ~27 h, **3-4 nights** | ~17 h, **2-3 nights** |
| Draft-OPD round, ScaleSearch, evaluation | ~1 night | ~1 night |

So about **two weeks of node-b nights per drafter**, most of it generation. The training nights displace SWE-bench and Night Shift, as the memo says. Adam on a 2B drafter needs 32-36 GB, and the target needs 21 GB beside it, so the model server must be stopped: this is not continual learning.

### 3.4 Cloud option (owner decision)

At BIG_DRAFTER's ratios (an H100 trains about 7× faster than the GB10 and prefills about 10× faster), the training share of one drafter is about 3-5 H100-hours: **roughly $10-50 per drafter** at $2.69-3.99 per hour, with setup. Generation would stay on node-b, since it is our own model serving.

- **What leaves the machine:** the transcripts, which are public tasks and the model's own text. The memo calls that an acceptable exposure if the owner agrees.
- **A precision caveat.** An H100 cannot run the NVFP4 target, so the features would come from a BF16 target and differ slightly from those the served drafter sees. A rented B200 (which has FP4) avoids that.

## 4. Evaluation

### 4.1 Sets

Every set is a fixed manifest of request bodies, each identified by its SHA-256. The prompts stay on node-b, as agentlong's do (`bench/agentlong/select_set.py`).

| Set | Content | Role |
|---|---|---|
| `agentlong` (existing) | 16 agent turns, 25-64K context | the general drafter's regression set, unchanged; `work`'s second set |
| `workheld` (new) | 24 turns from held-out SWE-rebench and SWE-smith sessions run by `ling`, in the same context bins, two thirds of them edits, at least four outside Python | `work`'s ship set |
| `asklong` (new) | 24 prose turns: 8 short chat answers, 4 phone-style answers, 4 with thinking on, 4 over long documents (16-64K), 4 non-English. Prompts are held out from `ask`'s training sets. | `ask`'s ship set; the phase 0 prose measurement |
| `structset` (new, small) | 8 `response_format` JSON and tool-heavy turns | the check that JSON does not regress under either specialist |
| mixed replay | two streams, one replaying a Work thread and one an Ask thread, interleaved as in production | the batch and switch check |

### 4.2 Method

`bench/agentlong/arms.sh` and `run.py` run each set at one stream, in the same GPU window, against the same engine commit. Each arm runs greedy and with seeds 1013 and 2026, with the tree on and every production flag.

- The arms are compared turn by turn with `bench/drafter/paired.py`: tokens per step and tok/s, as ratios of sums, with bootstrap 95% intervals over turns.
- The per-class split (`shell`, `prose`, `call_syntax`, `file_write`, `edit_script`, `tool_other`, as in `reports/moonshot-p0-copy-stats-2026-10-13.md`) is computed from greedy step dumps.

### 4.3 The bar to ship a specialist

The bar extends the memo's rule (SELF_SPEEDING §6.3) to specialists:

1. On its own held-out set, at least **+5% tokens/s** against `general` greedy, with the lower end of the 95% interval above 0, and sampled not worse.
2. On its own set, **no class more than 3% worse** than under `general`.
3. **The misroute cost is known and bounded.** On each other set, a specialist is at most 10% slower than `general`, so a wrong hint costs little. If it is slower than that, the routing rule of §2.2 sends doubtful requests to `general`.
4. **Exactness:** every check of §2.6 passes with it.

### 4.4 Regression checks: the general case must not get worse

- An engine with all three drafters resident, routing everything to `general`, runs `agentlong` at **at least 99% of the single-drafter engine's tok/s**, within the measurement's interval. This checks memory, graphs and the step's bookkeeping.
- With both streams on the same drafter, the mixed replay's aggregate is not below today's two-stream figure. With the streams on different drafters, it is not more than 5% below that.
- Option (c)'s switch cost is measured and reported per switch. It re-prefills from the nearest kept state at or before the window's start: at least 2,048 tokens, more when the last message boundary or prefix-store node lies earlier. Phase 2 reports the distribution on the mixed replay, and each switch is counted.
- `structset` is no slower under any shipped drafter than under `general`.
- Production's start canary and the tool-call canary (Mightling MODELS §2.1) pass with each drafter.

## 5. Distribution

### 5.1 The asset

Each drafter is one release asset: a tarball of the NVFP4 drafter directory (`config.json`, the safetensors, and a `drafter.json` manifest).

- **Size.** About 1.5-1.9 GB, under a GitHub release's 2 GiB limit per asset.
- **The manifest.** `drafter.json` holds the drafter's id, its workloads, its base (`incoai/Qwen3.8-27B-DFlash2` at a commit, Apache-2.0, with the base's attribution), the hash of its training-data manifest, its tree knobs (`tree_t`, `tree_chain`) and the evaluation numbers it shipped with.
- **The pin.** Each asset is pinned by SHA-256, per the hosting decision of 2026-10-27, and listed in the release's signed `SHA256SUMS` (the release signing of `specs/DREAMFERENCE_RELEASE_SIGNING.md`).
- **The engine refuses mismatches.** `ling-serve` refuses a drafter whose `drafter.json` names a target, block, selector shape or head dimension other than the one it serves. `DraftConfig::from_file`'s checks stand.

### 5.2 Registry fields (Mightling)

The ling-engine recipe in `ModelMatrixRegistry` gains two fields. They follow the shape that branch `models/scalesearch-drafter` already gives engine assets (`engine`, `feature`, `flag`, gated by the recipe's `engine_features`):

```python
"drafters": [
    {"id": "general", "url": "<release asset>", "sha256": "<64 hex>", "size_gb": 1.61,
     "engine": "ling", "feature": "multi-drafter"},
    {"id": "work", ...},
    {"id": "ask", ...},
],
"drafter_routes": {"work": "work", "swe-bench": "work", "night": "work",
                   "ask": "ask", "phone": "ask", "*": "general"},
```

- **One matrix entry per drafter**, as `qwen3.8-27b-dflash2-draft` is today, so the memory gates know each size before the file is on disk. The recipe's memory floor rises by the extra drafters' 3.2 GB, and `check_host_safety()` counts it before a load.
- **SGLang keeps one drafter.** It cannot switch, so its recipe's `speculative_config` names a single drafter: `work` once it ships, since the agent is most of the traffic. That is a one-drafter speed-up for SGLang users too.
- **A launch never runs with an empty or mismatched pin.** The tests check the pins, as they do for the drafter pin today.

### 5.3 Download

`ling-admin model download`, `server start` and `node sync-model` fetch the recipe's drafters as they fetch its drafter and engine assets today, and only when the recipe's engine is `ling` and its `engine_features` include `multi-drafter`. So a build that cannot read them never downloads them.

- **Verification.** The SHA-256 is checked before extraction. The file goes into the model directory keyed by its hash, and a mismatch deletes the download and stops the start with the asset's name.
- **The air gap.** The download happens at install or update, never while serving. The egress audit is unaffected.
- **The engine side.** `ling-serve --drafter ID=DIR` (repeatable) and `--drafter-routes FILE` load and route the drafters. `--draft DIR` stays and means `general` alone.

### 5.4 Memory

Each extra drafter is **1.61 GB** of weights (BIG_DRAFTER §3, measured in the engine) plus 42 MB of ring per resident session. Option (b) adds 105 MB per session when phase 4 builds it.

[INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)'s budget is about 57 GB, with the prefix KV store sized at start from what is left above a 16 GB reserve. In that budget:

- **The extra drafters come out of the engine's declared budget**, which grows by 3.2 GB for two specialists. The 16 GB reserve and Mightling's `container_headroom_gb` are not reduced.
- The prefix store's arena is not silently shrunk to make room.
- Every drafter is allocated at start, as the sessions are. Nothing is loaded while serving, because a load on unified memory is a host-safety event.

## 6. Phases toward 2.0

| Phase | What | Gate to the next |
|---|---|---|
| **0. Measure** (1-2 GPU windows, no training) | build `asklong`, `workheld` and `structset`; run the stock drafter on all four sets; split acceptance per class (from `LING_DUMP_STEPS` dumps, offline), per context bin, and per language for code; replay recorded requests through §2.2's rule to measure misroutes | the per-class and per-set table exists. A prose specialist goes ahead only if `asklong` accepts at least 15% fewer tokens per step than `agentlong`. If not, `ask` is dropped and 2.0 ships `general` and `work`. |
| **1. The pipeline and `work`** (the memo's option b; ~1-2 weeks of engineering plus ~2 weeks of nights) | generation on public tasks; online feature capture from `cap_`; the trainer (SpecForge's DFlash2 trainer, or `bench/drafter/train_step.py` grown into one); Draft-OPD; ScaleSearch; the evaluation harness. A 2M-token pilot first. | **Pilot:** at least +3% tokens per step on `workheld`, else stop and report. **Full:** the bar of §4.3 on `workheld` and `agentlong`. `work` can then ship alone, as the single drafter, on both engines, before 2.0. |
| **2. The engine holds several drafters** (~1-2 weeks) | `--drafter`, `--drafter-routes`; drafters bound per session; option (c) at request boundaries; per-drafter graphs and tree knobs; mixed-drafter batches; the `/spec_stats`, metrics and header of §2.5; `--switch-check`. Mightling side: the hint from app-server clients and `ling exec`, the registry fields, the download. | §2.6's checks pass; §4.4's regression checks pass; memory is within §5.4. |
| **3. `ask`** (pilot, then ~2 weeks of nights) | the data mix of §3.1, the same recipe | **Pilot:** at least +3% tokens per step on `asklong`. **Full:** the bar of §4.3 on `asklong`, and misroutes bounded. If it fails, 2.0 ships `general` and `work`. |
| **4. Switching inside a turn, then online** (optional) | option (b)'s feature window; span switching (§2.4); then the bandit | each one ships only if it adds at least 3% tok/s on the mixed replay, with exactness passing |
| **5. Release** | assets uploaded and pinned, `SHA256SUMS` signed, the registry's pins filled, the VALIDATION §13 exactness set per drafter, the docs | the release checks; the egress audit unchanged |

Phases 0 and 1 do not depend on phase 2, and phase 1's result is useful even if the rest stops: one better drafter for every user. Phase 2 can be built while phase 1's nights run.

## 7. Open questions for the owner

1. **Hosting.** Release assets on which repository's releases? The branch `models/scalesearch-drafter` names a hosted model repository (`dreamference/qwen3.8-27b-dflash2-nvfp4-scalesearch`) with a commit pin. Should that drafter move to a release asset with a SHA-256 pin as well, so that every drafter is fetched the same way?
2. **Local or rented training.** Local only, at about two weeks of nights per drafter that displace SWE-bench and Night Shift, or the training share on a rented GPU (tens of dollars per drafter, with the public-task transcripts leaving the machine)?
3. **The owner's own sessions.** A later opt-in tuning of `work` on the owner's machines only (the memo's option d), never shipped: wanted at all?
4. **Memory.** Three resident drafters add 3.2 GB to the engine's budget. Is that accepted, or should 2.0 hold two (`general` and one specialist)?
5. **Languages for `ask`.** Which languages besides English should its data cover, and in what proportion?
6. **`work` before 2.0.** Should `work` replace the stock drafter in a 1.x release as soon as phase 1 passes, on SGLang and ling-serve alike?
7. **The engine in 2.0.** Several drafters need ling-serve (SGLang serves one). Does 2.0 make ling-serve the default engine, or do several drafters apply only to `server start --engine ling`?
