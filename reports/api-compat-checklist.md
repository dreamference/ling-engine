# API compatibility: what the clients send, and what ling-serve does with it

October 9, 2026 · read from each client's source:
- **Codex:** the build Mightling's agent `ling` is made from, at the pinned tag `rust-v0.158.0`, plus Mightling's launcher and patches.
- **Cline, Continue, the OpenHands agent SDK and LiteLLM:** their main branches on 9 October.

ling-serve was checked at `3122b07` in `src/serve/api.cpp` and `src/serve/main.cpp`. The upstream issues behind several rows are in [engine-issues-2026-10-09.md](engine-issues-2026-10-09.md). Items marked *unverified* could not be confirmed from source.

**Status values:**
- **ok**: supported;
- **partial**: supported with a difference that matters;
- **ignored**: accepted and dropped silently;
- **refused**: HTTP 400;
- **missing**: no route.

## 1. `ling` (Codex): the Responses API only

| Feature | What the client sends or expects | ling-serve today | Gap |
| --- | --- | --- | --- |
| Endpoint | `POST /v1/responses`, `stream: true`. A chat wire API is refused by Codex's config parser. The launcher points the provider at `<host>/v1`. | ok (`generate_responses`) | |
| Model discovery | The launcher reads `GET /v1/models`: `id`, `max_model_len` | ok | |
| Request basics | `model`, `instructions`, `input[]`, `tools`, `tool_choice: "auto"` | ok (`parse_responses_request`) | |
| `reasoning` | `{effort}`. The launcher's catalog decides the values (`none` today; `minimal` … `xhigh` when thinking is enabled). `summary` only if the model advertises it. | ok: `none` → thinking off; other strings → the template's effort ladder | |
| `parallel_tool_calls: false`, `store: false`, `include: ["reasoning.encrypted_content"]`, `prompt_cache_key`, `client_metadata`, `service_tier` | Sent on every turn | ignored | Harmless. The prefix cache needs no key while one history is resident. A key becomes useful once several sessions are cached (../specs/DREAMFERENCE_LING_ENGINE_BACKLOG.md §19). |
| `text.format` (JSON schema) | Only for `exec --output-schema` | ignored | `--output-schema` is not enforced |
| Function tools | `{type: "function", name, description, parameters, strict}` | ok (rendered with production's key order) | |
| **Custom (freeform) tools** | Code Mode's `exec` is `{type: "custom", format: {type: "grammar", syntax: "lark", …}}`; history carries `custom_tool_call` / `custom_tool_call_output` | partial since 2026-10-10 (API §4): offered as a one-parameter function, answered as a `custom_tool_call` item with `custom_tool_call_input.delta/.done`; history replays. The grammar is rendered with the description, not enforced. | The patch dialect is the model's choice until the grammar is enforced (BACKLOG §19.13): in the test run of 2026-10-10 the model wrote apply_patch's syntax (`*** Begin Patch` / `*** Add File`) with the grammar rendered, and a unified diff without it. Live under Code Mode (`exec`): the acceptance run of 2026-10-10 passed (three `exec` calls as `custom_tool_call` items; production SGLang never called it). For `apply_patch`, not until the launcher's catalog offers the freeform tool. |
| Namespace / `web_search` / `tool_search` tools | Codex's own namespaces that patch 0020 does not flatten | ignored | Those tools are invisible to the model |
| Input items | `message` (user, developer, assistant), `function_call`, `function_call_output`, `reasoning` | ok: developer → system; items of one turn merged as production merges them (test: `api_tests` merge case) | |
| **Images** | The catalog leaves `input_modalities` at its default (text and image), so a pasted image goes out as `input_image`. `view_image` returns an image inside `function_call_output`. | **refused** (400) for a message image. Inside a tool output, silently reduced to its text. | Codex treats a 400 as terminal: the turn dies. Set `input_modalities: ["text"]` in the launcher catalog now, and answer with a placeholder instead of a 400 until the vision encoder lands (test: `test_image_input_does_not_fail_the_turn`). |
| SSE events read | `response.created` (id), `output_item.added` / `.done`, `output_text.delta`, `reasoning_text.delta` (needs `content_index`), `reasoning_summary_text.*`, `completed`, `failed`, `incomplete`. `function_call_arguments.delta` is ignored by Codex. | ok (test: `test_responses_stream_event_order`) | |
| `response.completed.usage` | `input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens`, `total_tokens` | partial: `reasoning_tokens` is always 0 | The token display under-reports reasoning (test: `test_responses_reasoning_tokens_counted`) |
| Cut at `max_output_tokens` | `response.incomplete` is a stream error to Codex | ling-serve sends `response.completed` with `status: incomplete`, deliberately | Keep; documented in `generate_responses` |
| Idle timeout | 300 s between SSE events | No keep-alive and no `in_progress` events during prefill | Fine at 65K (prefill ≈ 30–40 s at M2’s rates); a risk only with a much larger `--max-context` |
| **Context overflow** | See section 5 | **wrong shape** | The largest gap |

## 2. Cline: chat completions

Cline is now built on the AI SDK's chat-completions-compatible model, through the generic compatible provider or LM Studio's.

| Feature | What the client sends or expects | ling-serve today | Gap |
| --- | --- | --- | --- |
| Model discovery | Generic provider: none. LM Studio provider: `GET /models`, ids only. | ok | Neither reads `max_model_len`. Cline assumes 128,000 tokens against ling-serve's 65,536 unless the user sets it, so overflow is likely (section 5). |
| Streaming | `stream: true`, `stream_options.include_usage: true` | ok | |
| `max_tokens` | `min(32000, window − estimate − 1024)` | ok (clamped to the remaining context) | |
| Sampling | `temperature` only if configured; never `top_p`, `stop` or `seed` | ok (the checkpoint's defaults otherwise) | |
| Tools | Native `tools[]`; `tool_choice` left to the SDK default; no `parallel_tool_calls` | ok | |
| `reasoning_effort` | Only when reasoning is switched on, snapped to the model's advertised levels | partial: `none` → **400** on chat completions | Map `none` to thinking off as the Responses path does (test: `test_reasoning_effort_none_chat`, `api_tests`) |
| Messages | System string; content part arrays; `image_url` data URIs; `role: "tool"` with string content; `reasoning_content` replayed on assistant turns | ok, except **images → 400** | Default model info claims image support |
| Response read | `delta.content`; `delta.reasoning_content` or `delta.reasoning`; `tool_calls` by `index` with `id` and `name`; a `finish_reason`; `usage` with `cached_tokens` | ok (test: `test_stream_chunk_shape`) | |
| Errors after the stream starts | Expects a status code, or a stream that still ends properly | partial: HTTP 200, then `data: {"error": …}`, then end of stream with **no finish chunk and no `[DONE]`** | Validate before sending headers. Anything after them should end with a finish chunk and `[DONE]`. |

## 3. Continue: chat, edit, autocomplete

| Feature | What the client sends or expects | ling-serve today | Gap |
| --- | --- | --- | --- |
| Chat formatting with `provider: vllm` | `vllm` does not count as a provider that applies the template, so for a Qwen model name Continue builds a ChatML transcript itself and sends it as one user message | Templated twice | A client setting. Document: use the LM Studio or generic provider, or `promptTemplates.chat: none`. |
| Chat body | `max_tokens`, `stream`, `stop`, `tools` (+`strict`), `tool_choice`, `parallel_tool_calls: false`, `stream_options.include_usage`; temperature, top_p and penalties only if set; never `reasoning_effort` | ok. `frequency_penalty` **ignored**; `parallel_tool_calls: false` not enforced. | Minor |
| Model info | `GET /v1/models` → `data[0].id`, `max_model_len` | ok | |
| Autocomplete | `POST /v1/completions`: `prompt` is a client-built fill-in-the-middle template (Qwen's FIM tokens when the model name contains "coder"), `stop` lists of FIM and `<|im_end|>` tokens, `max_tokens` up to 4,096, `temperature: 0.01`, `stream: true`; no `suffix`, `echo` or `n` | ok at the protocol level, but **streaming with a stop list can break on non-ASCII text**: the hold-back is in bytes and can split a character (test: `test_streaming_with_stop_and_multibyte_text`) | Fix the hold-back. Whether Qwen3.8-27B completes FIM prompts well is unverified, and with no "coder" in the name the client picks a different template. |
| Edit | ChatML prompt with an assistant prefill through `/v1/completions` | ok | |
| Embeddings | `POST /v1/embeddings` for codebase indexing | **missing** | Point Continue's indexing elsewhere, or add the route with a small embedding model (out of the one-model scope) |
| Rerank | `POST /v1/rerank` | **missing** | Same |

## 4. OpenHands: chat completions through LiteLLM

| Feature | What the client sends or expects | ling-serve today | Gap |
| --- | --- | --- | --- |
| `reasoning_effort` | `"high"` on every call; temperature, top_p and top_k removed | ok (high → xhigh) | Always maximum thinking. That is the client's setting, but costly in latency. |
| `stream` | false | ok | |
| Messages | Every content is a parts list, tool messages included; LiteLLM flattens assistant content to a string | ok (`production_messages` joins tool parts) | |
| Tools | Native by default, no `tool_choice`. The prompt-based fallback uses `stop: ["</function"]`. | ok | With the fallback, streaming plus a stop list meets the UTF-8 hold-back bug |
| `prompt_cache_key`, `seed: null`, unknown params | Sent | ignored or tolerated | |
| Context size | No `/v1/models` call. Condensation is by event count unless the user sets `max_input_tokens`. | | Overflow is likely, so the error shape matters (section 5) |
| Images | Only if LiteLLM thinks the model has vision (probably not; unverified) | refused if sent | Low |

## 5. Context-window overflow: what each client recognises

| Client | Recognised as overflow | ling-serve today | Result today |
| --- | --- | --- | --- |
| Codex | **Only** an in-stream `response.failed` whose `error.code` is `"context_length_exceeded"` (`is_context_window_error` in the pinned source). An HTTP 400 is terminal, with no compaction. An HTTP 500 or other failure code is retried. | `response.failed` with `code: "server_error"` (detected on the engine thread after `response.created`) | Retried up to 5 times with the same prompt, then the turn fails; **compaction never runs** |
| Cline | `context_length_exceeded` in `code`/`type`/`name`, or a message matching patterns such as "context length/window/limit", "maximum context", "too many tokens", "prompt is too long". The status must be 400, 413 or 422, or absent. A 500 never counts. | Streaming: 200 then an error chunk. Non-streaming: 500. | May work while streaming (the message contains "context limit"), but only if the SDK surfaces the chunk as an error (unverified) |
| Continue | No reactive handling; prunes client-side with its own token estimate | 500 or an error chunk | Request fails, no recovery |
| OpenHands (LiteLLM) | Message substrings such as "maximum context length is", "is longer than the model's context length", "exceeds the available context size". The bare code `context_length_exceeded` is not in LiteLLM's generic list. | 500 "the prompt is longer than the context limit" | Matches neither list: `InternalServerError`, retried for about 2 minutes, then the run dies; **no condensation** |

**What satisfies all four:**
- **Chat completions and completions:** the prompt is already tokenized in the HTTP handler, so check its length there and answer **HTTP 400 before any header**:

  ```json
  {"error": {"code": "context_length_exceeded", "type": "invalid_request_error",
             "message": "This model's maximum context length is N tokens. However, your request has M input tokens. Please reduce the length of the messages."}}
  ```

- **Responses API:** keep the error in-stream, as `response.failed` with `error.code: "context_length_exceeded"`. This is a different shape on purpose: it is the only one Codex compacts on.
- Test: `test_context_overflow_is_recognisable`.

## 6. Request features no client needs today

These are ignored without an error. An error (400) is better than silence until each is built.

| Field | Who sends it | ling-serve | Test |
| --- | --- | --- | --- |
| `logprobs`, `top_logprobs` | none of the four by default | ignored | `test_logprobs_are_returned_or_refused` |
| `n > 1` | none | ignored (one choice) | `test_n_is_honoured_or_refused` |
| `response_format` / `text.format` | Codex `--output-schema`; scripts | ignored | `test_response_format_is_honoured_or_refused` |
| `tool_choice: "required"` or a named function | none by default | not enforced (a named choice filters the tool list) | `test_tool_choice_required_yields_a_call` |
| `frequency_penalty`, `logit_bias` | Continue if configured | ignored | |
| `suffix`, `echo` on completions | none (Continue builds FIM itself) | ignored | |
| `reasoning` as the field name (vLLM's current spelling) | Clients written against vLLM may echo it back | Only `reasoning_content` is read and written. An echoed `reasoning` is dropped from the history, which changes the prompt and misses the prefix cache. | |

## 7. Ranked gaps

For one user on one GB10 with 1–4 agents:

1. **Overflow errors** (section 5). A small change with the largest effect: the agent can compact instead of failing the turn.
2. **Images refused with 400.** For Codex, a catalog change in the launcher plus a placeholder in ling-serve.
3. **Stop-list streaming splits UTF-8.** It breaks Continue's autocomplete and OpenHands' fallback on non-ASCII text.
4. **Custom tools dropped** (Codex Code Mode). Built 2026-10-10 (API §4), partial: the dialect question stays with BACKLOG §19.13.
5. **Errors after the stream starts** end without a finish chunk or `[DONE]`.
6. **`reasoning_effort: "none"` on chat completions** is a 400.
7. **Invalid-JSON arguments in a chat history** are a 400 on every later request ([vllm#47761](https://github.com/vllm-project/vllm/issues/47761) is the same class).
8. **`reasoning_tokens` is always 0.**
9. **Silently ignored controls** (section 6): reject what is not built.
10. **No embeddings or rerank** for Continue's indexing. Probably stays out of scope; document it.
