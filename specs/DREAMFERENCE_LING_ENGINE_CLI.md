# ling-engine: Command lines

Part of the ling-engine specification; the index is [README.md](./README.md). Written on 2026-10-09 from the code (`src/serve/main.cpp`, `tools/*.cpp`, `src/core/engine.cpp`), which is the source of truth: when the two differ, the code is right and this document is behind.

Four programs are built from the repository (`CMakeLists.txt`): `ling-serve`, the HTTP server; `ling-run`, the offline generator and checker; `ling-tokenize` and `ling-template`, two small reference tools the Python test harness drives. The test binaries (`ling-kernel-tests`, `ling-stream-tests`, `ling-prefill-tests`, `ling-spec-tests`, `ling-sampling-tests`, `ling-api-tests`) take no arguments. Every program parses its arguments by hand: an unknown argument prints `unknown argument …` and exits 2, a flag that needs a value and has none prints `… needs a value` and exits 2, and a missing required argument prints the usage line and exits 2.

## 1. `ling-serve`

```
ling-serve --model DIR [--host H] [--port P] [--served-model-name N] [--max-context N]
           [--draft DIR [--draft-block N] [--lookup 0|1|2] [--lookup-min N]]
           [--pretokenizer production|checkpoint] [--prefix-checkpoints N]
           [--graphs] [--pdl] [--attn-bulk] [--attn-prefetch N] [--keepalive-seconds N]
           [--prefill-bf16] [--sessions N] [--session-protect-seconds S]
           [--batch N] [--prefill-chunk-contended N]
```

| Flag | Default | What it does |
|---|---|---|
| `--model DIR` | required | The checkpoint directory: the safetensors, `config.json` and `tokenizer.json`. |
| `--host H` | `0.0.0.0` | The listening address. Every interface by default, as production's engine listens; Mightling's gate or the Docker network is the boundary, not this flag. |
| `--port P` | `8000` | The listening port. |
| `--served-model-name N` | the last path component of `DIR` | The id `GET /v1/models` reports and every response carries. Mightling's tools ask the server which model runs, so this must be the id the registry expects ([INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)). |
| `--max-context N` | `65536` | The context length: `max_model_len` in `/v1/models`, the limit of `context_length_exceeded`, and the size of the KV and state allocations. |
| `--draft DIR` | none | The DFlash2 drafter's checkpoint. Without it every request decodes plainly. |
| `--draft-block N` | `12` | Draft tokens per speculative step. The default follows production, which runs 12 since 2026-10-09 (measured +7% single-stream over the drafter's published block of 16; the owner's decision of 2026-10-10 made 12 the code default too). The M0 reference container on the development machine keeps 16 on purpose. |
| `--lookup 0\|1\|2` | `1` | The context-lookup draft source ([SPECULATION §10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md)): 0 off, 1 on, 2 shadow mode (counted in `/spec_stats`, not used). |
| `--lookup-min N` | `8` | The shortest match the lookup accepts. |
| `--pretokenizer production\|checkpoint` | `production` | The pre-tokenizer pattern: production's (how SGLang tokenizes this checkpoint) or the checkpoint's own `tokenizer.json` pattern. Any other value exits 2. |
| `--prefix-checkpoints N` | `8` | How many shared-prefix state checkpoints the engine keeps ([SESSIONS §11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)). |
| `--graphs` | off | Each speculative step's draft, verify and commit run as captured CUDA graphs (M3). Needs a drafter; refused, with a notice, while `LING_PROFILE` or `LING_KSPLIT` is set. |
| `--pdl` | off | Programmatic dependent launch for the rows path's kernels (M3). |
| `--attn-bulk` | off | The rows path's attention loads its tiles with bulk copies (M3). |
| `--attn-prefetch N` | `2` | How many KV tiles ahead the rows path's attention prefetches into L2; 0 switches it off (M3). |
| `--keepalive-seconds N` | `15` | How often a streamed request that has produced no token yet (waiting in the queue, or in prefill) is sent a keep-alive, so a client with an idle timeout keeps the stream: a `response.in_progress` event on `/v1/responses`, an SSE comment line on the chat and completions routes ([API §3, §4](./DREAMFERENCE_LING_ENGINE_API.md)). 0 switches them off. `LING_KEEPALIVE_SECONDS` overrides it. |
| `--sessions N` | `2` | Resident sessions, each with its own BF16 KV cache, DeltaNet and conv state and drafter context (about 4.8 GB each at 65,536 tokens of context), all allocated at start (M4 stage 1, [drafts/M4 §1](./drafts/M4-sessions-and-batches-2026-10-10.md)). A request runs in the session it extends (its `prompt_cache_key`'s session looked at first, the token prefix deciding), else in a new session that takes a free slot or evicts the least recently used one. `sglang:max_total_num_tokens` reports sessions × `--max-context`. Requests still run one at a time. |
| `--session-protect-seconds S` | `30` | A session bound with a `prompt_cache_key` whose last request ran less than S seconds ago is never evicted: a new session's request waits in the queue (`sglang:num_queue_reqs`) while requests that continue a resident session run (M4 §7 question 6). Sessions without a key are protected only while a request of theirs runs or waits. 0 turns protection off. The default was 180 s in stage 1; 30 s by the owner's decision of 2026-10-11. |
| `--batch N` | `2` | Requests that run at once, each in its own resident session (at most `--sessions`; M4 stage 2, [drafts/M4 §3](./drafts/M4-sessions-and-batches-2026-10-10.md)). The engine thread interleaves them: the running requests take turns, one prefill chunk or one decode step each, every unit in its own session's state, so each request's output is bit for bit what it is alone (`ling-run --interleave-check`). A request that continues a running request's session waits for it. 1 runs requests one at a time, as before stage 2. |
| `--prefill-chunk-contended N` | `256` | The prefill chunk, in tokens, while another request decodes (a multiple of 32 in [32, 2,048]; rounded down to a multiple of 32): each chunk runs between two of the decoding request's steps, so it bounds that request's step period during a long prefill (measured in drafts/M4 §6, stage 2). Chunk ends stay where a resumed prefill could start, so the state is the same (`ling-run --chunk-check`). Without a decoding request the prefill runs in its full chunks of 2,048. |
| `--prefill-bf16` | off | The prefill's DeltaNet intermediates (the input norm's copy for the a/b projections, the in_qkv and in_z projections, the conv's output, the recurrence's output) kept in BF16 as production keeps them, instead of FP32 (B34; [reports/M2-prefill.md](../reports/M2-prefill.md) §8 item 3). Not exact: a change of the prefill's numerics, so off until the checks of [MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md) pass (`bench/prefill_bf16_check.sh`). The decode and verify rows path is untouched either way. |

Startup prints `loading DIR …`, then `loaded X GB of weights`, then with a drafter `drafter: X GB, block N`; the server is ready when `GET /health` answers. The HTTP server runs two threads for the connections and one engine thread that runs up to `--batch` requests at once, interleaved one prefill chunk or decode step at a time ([API](./DREAMFERENCE_LING_ENGINE_API.md)); an idle connection is closed after 30 minutes; `SIGINT` and `SIGTERM` stop it. Each finished request logs one line to stderr: its session, prompt tokens (and how many were reused from a kept state), prefill seconds (wall, other requests' steps included), its chunks (and how many ran beside another request's decode) and its own tokens/s of compute, tokens generated, decode seconds and tokens/s, the finish reason, the number of passes and, speculatively, tokens per pass, and its step periods (start to start of its consecutive steps: mean and max, apart from those with another request's prefill chunk in between).

## 2. `ling-run`

```
ling-run --model DIR (--text T | --chat T | --ids 1,2,3 | --prompts-file F)
         [--max-tokens N] [--max-context N] [--temperature T] [--seed N]
         [--ids-out] [--prompt-ids-out]
         [--draft DIR [--block N] [--spec] [--spec-check] [--lookup 0|1|2] [--lookup-min N]]
         [--prefix-check [--sessions N]] [--chunk-check N] [--interleave-check [--prefill-chunk-contended N]]
         [--graphs] [--graph-check] [--pdl] [--attn-bulk] [--attn-prefetch N]
         [--prefill-bf16]
```

Loads the model and generates from a prompt; prints the text, the timings and, asked to, the token ids, for checking against a reference server (`bench/validate_v0.py`, `bench/agreement/`).

| Flag | Default | What it does |
|---|---|---|
| `--model DIR` | required | As for `ling-serve`. |
| `--text T` | | A raw prompt, tokenized as given. |
| `--chat T` | | One user message, rendered through the chat template. |
| `--ids 1,2,3` | | The prompt as token ids. |
| `--prompts-file F` | | One JSON string per line, each a raw prompt; the checks below run over every prompt. |
| `--max-tokens N` | `64` | Tokens to generate per prompt. |
| `--max-context N` | `32768` | The context length. Half of `ling-serve`'s default: the checker's prompts are short and the smaller allocation loads faster. |
| `--temperature T` | `0` (greedy) | Sampling temperature. `ling-serve`'s default is the checkpoint's 1.0; the checker is greedy so its output can be compared token for token. |
| `--seed N` | `0` | Seeds sampling when the temperature is above 0 (0 draws a random seed). |
| `--ids-out`, `--prompt-ids-out` | off | Print the generated ids, or the prompt's ids, one list per prompt. |
| `--draft DIR`, `--block N`, `--lookup`, `--lookup-min` | none, `12`, `1`, `8` | The drafter and its settings, as `ling-serve`'s `--draft`, `--draft-block`, `--lookup`, `--lookup-min`. |
| `--spec` | off | Decode speculatively with the drafter. |
| `--spec-check` | off | Greedy only: decode each prompt plainly, then speculatively, and check that the tokens are identical and that the recurrent state after the speculative run equals a plain run's over the same tokens, bit for bit. Prints `spec check passed` or `SPEC CHECK FAILED`. |
| `--prefix-check` | off | Prefill each prompt after the one before it (resuming from a kept state or a prefix checkpoint), then again from scratch, and check that the DeltaNet and conv state and the next token agree (M2). With `--draft`, also decode 64 tokens speculatively (greedy) after each prefill and check that the tokens, the state, the steps and the accepted drafts agree (the drafter's context ring after a resume, M4). |
| `--sessions N` | `1` | With `--prefix-check`: N resident sessions, each prompt bound to one as `ling-serve` binds a request, so each resumes from its own session's state, the shared checkpoint store, or another session's KV (M4 stage 1). |
| `--chunk-check N` | off | Prefill each prompt from a cold engine twice, in full chunks and in chunks of N tokens (what `ling-serve` runs beside another request's decode), and check that the state and the next token are the same both ways; with `--draft` also a 64-token speculative continuation's tokens, state, steps and accepted drafts (the drafter's context ring, which the prefill's chunks fill). Exit 1 on a difference (M4 stage 2). |
| `--interleave-check` | off | With `--draft` and `--sessions 2`: the prompts in pairs (A, B), each first decoded alone (cold engine, `--max-tokens` speculatively); then A prefills and decodes while B prefills in chunks of `--prefill-chunk-contended` tokens (default 256) between A's steps, and the two take steps in turn, as `ling-serve` interleaves two requests. Per sequence the tokens, the state, the steps and the accepted drafts must equal the run alone; greedy and sampled (`--temperature`, default 1, seeded by `--seed` or 1), and with the pair swapped. Exit 1 on a difference (M4 stage 2, [drafts/M4 §4](./drafts/M4-sessions-and-batches-2026-10-10.md)). |
| `--graphs`, `--pdl`, `--attn-bulk`, `--attn-prefetch N` | off, off, off, `2` | As for `ling-serve`. |
| `--prefill-bf16` | off | As for `ling-serve`: BF16 DeltaNet intermediates in the prefill (not exact). The checks below run within the mode when it is on: `--prefix-check` and `--spec-check` must still pass bit for bit, which `bench/prefill_bf16_check.sh` verifies. |
| `--graph-check` | off | Decode each prompt speculatively twice from the same prefill, without and with step graphs, greedily and then sampling (top-k 20, top-p 0.95, seeded by `--seed` or 1), and check that the tokens, the recurrent state (bit for bit) and the step and acceptance counts agree. |

The three checks are the exactness tests the milestones' gates name ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md)); a failed check is reported and the exit status is 1.

## 3. `ling-tokenize` and `ling-template`

```
ling-tokenize tokenizer.json < strings.jsonl
ling-template < cases.jsonl
ling-template --responses tokenizer.json [checkpoint] < bodies.jsonl
```

`ling-tokenize` reads one JSON string per line and prints each one's token ids, for `bench/tokenizer_test.py`, which compares them with the reference tokenizer. `ling-template` renders chat-template cases (one JSON object per line) to the prompt text the engine would see, for `bench/template_test.py`, which compares them with the patched Jinja template; with `--responses` it takes Responses API request bodies instead, converted the way `ling-serve` converts them, tokenized with production's pre-tokenizer unless the fourth argument is `checkpoint`.

## 4. Environment variables

Read by the engine at start (`src/core/engine.cpp`) and by two kernels, and the last one by `ling-serve`. **A variable set in the environment overrides the matching flag**, so a flag that seems to have no effect was probably overruled by one of these; they exist so a benchmark can switch a path without rebuilding the launch line.

| Variable | Effect |
|---|---|
| `LING_STEP_GRAPHS=0\|1` | Overrides `--graphs`. |
| `LING_PDL=0\|1` | Overrides `--pdl`. |
| `LING_ATTN_BULK=0\|1` | Overrides `--attn-bulk`. |
| `LING_ATTN_PREFETCH=N` | Overrides `--attn-prefetch`. |
| `LING_PREFILL_BF16=0\|1` | Overrides `--prefill-bf16` (also the way to switch it on in a program without the flag, such as `bench/agreement/ling_force.cpp`). |
| `LING_PROFILE` | Set to anything: per-phase timing with a synchronization after each phase. Switches step graphs off (nothing may synchronize inside a captured step). |
| `LING_KSPLIT=N` | Split-K for the streaming GEMM; set and non-zero it allocates a buffer per step, which also switches step graphs off. |
| `LING_GEMM_SHAPE` | Selects the prefill GEMM tile shape (`src/core/prefill_gemm.cu`); a tuning switch for the benchmarks. |
| `LING_ATTN_CHUNK` | The decode attention's key chunk (`src/core/kernels.cu`); a tuning switch. |
| `LING_PDL_ATTN` | Programmatic dependent launch for the attention kernel alone (`src/core/kernels.cu`). |
| `LING_KEEPALIVE_SECONDS=N` | Overrides `--keepalive-seconds` (read by `ling-serve`, `src/serve/main.cpp`). |

Only the last is read by the HTTP layer; the rest of the server's behaviour towards clients is the flags' alone.
