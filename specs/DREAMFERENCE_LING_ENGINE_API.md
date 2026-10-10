# ling-engine: The HTTP API of `ling-serve`

Part of the ling-engine specification; the index is [README.md](./README.md). Written on 2026-10-09 from the code (`src/serve/main.cpp`, `src/serve/api.cpp`, `src/serve/api.hpp`), which is the source of truth. What the four clients actually send, and which of it is handled, is measured in [reports/api-compat-checklist.md](../reports/api-compat-checklist.md); the gaps it found are ranked in [BACKLOG §19](./DREAMFERENCE_LING_ENGINE_BACKLOG.md).

`ling-serve` presents the OpenAI-compatible surface the production SGLang server presents for this model, so that Mightling and the other clients (Cline, Continue, OpenHands, the web chat) switch engines by changing nothing but the server behind the port ([INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)). No authentication: the boundary is the network, as it is for production.

## 1. Model of execution

- **One request at a time.** The HTTP threads (two) parse requests and stream events back; one engine worker thread takes jobs from a queue in arrival order and runs each to completion. A second request waits in the queue; nothing is batched yet ([SESSIONS §11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md) is the plan).
- **Requests are tokenized on the HTTP thread**, so the context check (below) is answered before the job is queued.
- **Cancellation.** A client that disconnects marks its job cancelled; the engine stops at the next emitted token and reports the request as finished with reason `stop`. A job still in the queue when its client leaves runs its prefill and stops at the first token.
- **Limits.** The request body is capped at 64 MB (413 above it); an idle connection is closed after 30 minutes. There is no request timeout: a request runs until its token limit, a stop, the end of the context or a disconnect.
- **State reuse.** A prompt that extends the previous prompt, or a kept prefix checkpoint, resumes from that state instead of prefilling from scratch; `usage.prompt_tokens_details.cached_tokens` says how many tokens were reused.

## 2. Routes

| Method and path | Answer |
|---|---|
| `GET /health`, `GET /healthz` | `200`, body `ok`, as soon as the model is loaded. |
| `GET /v1/models` | `{"object":"list","data":[{"id":NAME,"object":"model","created":…,"owned_by":"ling-engine","root":NAME,"max_model_len":N}]}`. `NAME` is `--served-model-name`; `max_model_len` is `--max-context`. Mightling's launcher reads both. |
| `GET /metrics` | Prometheus text (`text/plain; version=0.0.4`): counters under SGLang's names, so the replay harness reads them unchanged (§6). |
| `GET /spec_stats` | The engine's speculation counters as JSON (§6). |
| `POST /v1/responses` | The Responses API: what Mightling's agent calls (§4). |
| `POST /v1/chat/completions` | Chat completions (§3). |
| `POST /v1/completions` | Completions: a raw prompt, no template; what `ling-admin server start`'s NVFP4 canary and the readiness probes use (§3). |
| anything else | `404` with the error shape of §5, type `not_found`. |

## 3. Chat completions and completions

**Request fields** (`parse_request`). Unknown fields are ignored without error.

| Field | Default | Notes |
|---|---|---|
| `messages` (chat) | required | Rendered through the checkpoint's chat template with production's two patches ([docs/V0.md](../docs/V0.md)). Tool messages whose content is a list of text parts are joined; a trailing assistant message is continued. |
| `prompt` (completions) | required | A string, a one-string list, or a list of token ids. Anything else is a 400. |
| `stream` | `false` | Server-sent events when true. |
| `stream_options.include_usage` | `false` | Streamed: a final chunk with empty `choices` and the `usage` object before `[DONE]`. |
| `max_completion_tokens`, else `max_tokens` | the rest of the context | The generation limit, clipped to what the context has left. |
| `temperature` | `1.0` | 0 is greedy. The defaults are the checkpoint's `generation_config`, as in production. |
| `top_p`, `top_k`, `min_p` | `0.95`, `20`, `0` | |
| `presence_penalty`, `repetition_penalty` | `0`, `1.0` | Set to anything else, the request decodes **plainly**: the drafter is not used, because the penalties change the verify distribution. |
| `seed` | random | An integer seeds the sampler. |
| `stop` | none | A string or a list of strings. Output is held back to a character boundary until a stop can no longer match, so a stop string never leaks in part. |
| `tools`, `tool_choice` (chat) | none, `auto` | `tools` are passed to the template in canonical form (`type`, `function{name, description, parameters, strict}`, `defer_loading`). `tool_choice: "none"` drops them; a named function keeps only that tool; `"required"` is **not enforced**. A tool of any other `type` (a Responses `custom` tool sent here) is dropped: this route offers `function` tools only, as vLLM does. |
| `chat_template_kwargs.enable_thinking`, or top-level `enable_thinking` | template default (on) | Thinking on or off; off, no reasoning is split from the output. |
| `chat_template_kwargs.preserve_thinking`, or top-level | template default | Keeps earlier turns' reasoning in the rendered prompt. |
| `chat_template_kwargs.reasoning_effort`, or top-level `reasoning_effort` | template default (`medium`) | Mapped as production's patched template maps it (`max`/`high` to xhigh, `minimal` to low). |

**Ignored without error, by design until built** ([BACKLOG §19.11](./DREAMFERENCE_LING_ENGINE_BACKLOG.md)): `logprobs` and `top_logprobs` (never returned), `n` (always one choice), `response_format`, `parallel_tool_calls`, `tool_choice: "required"`.

**Unstreamed answer.** `{"id":"chatcmpl-…"|"cmpl-…","object":"chat.completion"|"text_completion","created":…,"model":NAME,"choices":[{"index":0,"message":{…}|"text":…,"finish_reason":…}],"usage":{…}}`. The chat message carries `role`, `content` (null when the answer is tool calls only), `reasoning_content` when thinking produced any, and `tool_calls` (`id` `call_…`, `type: "function"`, `function{name, arguments}` with `arguments` as JSON text). `finish_reason` is `stop` (the end-of-turn token or a stop string), `length` (the token limit or the end of the context), or `tool_calls` (stopped with at least one call). `usage` has `prompt_tokens`, `completion_tokens`, `total_tokens` and `prompt_tokens_details.cached_tokens`.

**Streamed answer.** `Content-Type: text/event-stream`; each event is `data: <chunk>\n\n`; the stream ends with `data: [DONE]\n\n`. Chat chunks are `chat.completion.chunk` objects with one `choices[0].delta`: the first carries `role`, then `reasoning_content` and `content` as they arrive, and `tool_calls` entries (each with `index`, `id`, `type` and the whole `function` at once: a call is emitted when its closing marker has been parsed, not token by token). Completions chunks are `text_completion` objects with `choices[0].text`. A final chunk with an empty delta carries `finish_reason`; then, if asked, the usage chunk; then `[DONE]`.

**Tool calls are parsed from the model's text** (`OutputParser`): a `<tool_call>` marker starts a call only when `<function=` follows and names a function the request offered, and never inside a Markdown code fence, so an example in a fence, the template's placeholder name or a marker quoted in prose stays text ([VALIDATION §18](./DREAMFERENCE_LING_ENGINE_VALIDATION.md)). Arguments are rendered to JSON with the parameter types the tool schema declares.

## 4. The Responses API

`POST /v1/responses` is converted to a chat request the way SGLang converts it (`parse_responses_request`), then runs as §3 with thinking on unless switched off:

- `instructions`, and `developer` messages in `input`, become one leading system message; `input` is a string or a list of items (`message`, `function_call`, `function_call_output`, `custom_tool_call`, `custom_tool_call_output`, `reasoning`); function calls and their outputs become assistant tool calls and tool messages; reasoning items become `reasoning_content`; `function` and `custom` tools reach the template, every other tool type is dropped; `tool_choice: "none"` drops them all.
- **Custom (freeform) tools** (`{type: "custom", name, description, format}`, Codex's `apply_patch` for some model families): the model has no freeform tool format and production SGLang drops `type: custom` tools (read from its source on 2026-10-09 and seen live on 2026-10-10: the same Code Mode `ling exec` carried the custom `exec` tool in every request and the model never called it in eleven turns against production, while against ling-serve it called it three times in seven turns and the task completed), so the rendering is ling-engine's own and the only one in use (the owner's choice, 2026-10-10). A custom tool is offered as a function with the one string parameter `input` (its `description` rendered with the grammar from `format` appended to it as prose, the grammar not enforced), and a call to it comes back as a `custom_tool_call` item whose `input` is that parameter's text (failing that, the single string parameter the model used instead; failing that, the arguments text itself). A replayed `custom_tool_call` renders as that one-parameter call, so the turn renders as the model wrote it; a `custom_tool_call_output` is a tool message like a `function_call_output`. The grammar is rendered because without it the model answered a unified diff where apply_patch's syntax was wanted (2026-10-10); it is not enforced, so the dialect stays the model's choice (`test_responses_custom_tool_call_stream` checks the item, not the dialect; what the model wrote is in `reports/api-compat-checklist.md`). Constraining the output to the grammar is the follow-up if the agent still needs it ([BACKLOG §19.13](./DREAMFERENCE_LING_ENGINE_BACKLOG.md)). On chat completions, the vLLM-compatible route, a `type: custom` tool is dropped: only `function` tools reach the template there.
- `max_output_tokens` is the limit; `stream`, the sampling fields, `stop`, `seed` and `chat_template_kwargs` pass through; `reasoning.effort` becomes the template's effort, except `none`, which switches thinking off.
- A request body that is not an object, or has no `input`, is a 400.

**Streamed** (what Mightling's agent parses), each event `event: <type>\ndata: <json>\n\n` with a `sequence_number`: `response.created` (status `in_progress`); then per output item `response.output_item.added`, its deltas (`response.reasoning_text.delta` for a `reasoning` item, `response.output_text.delta` for a `message` item), `response.output_item.done`; a `function_call` item is added and done at once with `name`, `arguments`, `call_id`; a `custom_tool_call` item is added with `name`, `call_id` and `input` present but empty (Codex parses the item only with `input` present), then one `response.custom_tool_call_input.delta` with the whole input, keyed by `item_id` and `call_id` as Codex reads it, then `response.custom_tool_call_input.done`, then `response.output_item.done` with the full item; then `response.completed` with the whole response object. **A response cut at `max_output_tokens` still ends with `response.completed`**, status `incomplete` and `incomplete_details.reason: "max_output_tokens"`, because the agent treats `response.incomplete` as a failed turn and discards what was written. `usage` has `input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens` (always 0: reasoning tokens are not counted separately yet, [BACKLOG §19.7](./DREAMFERENCE_LING_ENGINE_BACKLOG.md)) and `total_tokens`.

**Unstreamed**: the same response object, `200`.

## 5. Errors

The error shape is OpenAI's: `{"error":{"message":…,"type":…,"code":…}}`, `code` present when there is one.

| Case | Answer |
|---|---|
| Body not JSON, not an object, missing `messages`/`prompt`/`input`, a bad `prompt` | `400`, type `invalid_request_error`. |
| Prompt longer than the context | **Before any header is sent**: `400`, type `invalid_request_error`, code `context_length_exceeded`, a message in the form the clients' libraries recognise ("maximum context length is …"). This is the one error the agents compact on; Codex retries a `server_error` instead. |
| The same, found while a stream is already open | Chat and completions: an error object as a `data:` event, then a chunk whose `finish_reason` is `error`, then `[DONE]`, so no stream is left unterminated. Responses: `response.failed` with `error.code` `context_length_exceeded`, then the stream ends. |
| Logits not finite (NaN or inf) during a request | The request **ends with an error** rather than sampling garbage (sampled anyway they gave `!` forever, or a uniform draw streamed as a healthy answer: [VALIDATION §18](./DREAMFERENCE_LING_ENGINE_VALIDATION.md)): `500`, type `server_error`, or the streamed forms above; counted in `ling:nonfinite_logits_total`. |
| Any other exception in the engine | `500`, type `server_error`, or the streamed forms. |
| Body over 64 MB | `413`, type `invalid_request_error`. |
| Unknown route | `404`, type `not_found`. |

## 6. Counters

`GET /metrics` exports, labelled `{model_name="NAME"}`: `sglang:prompt_tokens_total`, `sglang:cached_tokens_total`, `sglang:generation_tokens_total` (the stop token counted, as SGLang counts it), `sglang:spec_verify_calls_total` (decode passes, speculative or plain), `sglang:e2e_request_latency_seconds_sum` and `_count`, `sglang:time_to_first_token_seconds_sum` and `_count`, and ling-engine's own `ling:spec_drafted_tokens_total`, `ling:spec_accepted_tokens_total`, `ling:nonfinite_logits_total`. `GET /spec_stats` returns the engine's speculation counters since start as JSON: `steps`, `drafted`, `accepted`, `accept_histogram`, the draft, verify and commit seconds, the lookup branch's `lookup_steps` and `lookup_accepted`, and the shadow-mode counters (`shadow_*`) that measure the lookup source without using it ([SPECULATION §10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md)).

## 7. What a client should not expect yet

One request at a time (a second waits); no `logprobs`, no `n > 1`, no structured output (`response_format`, `text.format`), no enforced `tool_choice: "required"`; `reasoning_tokens` reported as 0; images answered as text; the chat template is compiled in (the checkpoint's with production's patches), not read from a file at launch. Each is in [BACKLOG §19](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) with its rank. **The goal (the owner's decision, 2026-10-10) is the complete Codex Responses API on `/v1/responses`, with the chat and completions routes kept as the vLLM-compatible surface for the other clients, and no protocol of ling-engine's own** ([BACKLOG §19.15](./DREAMFERENCE_LING_ENGINE_BACKLOG.md)); the gaps against the pinned Codex source are listed there; custom tools landed on 2026-10-10 (§4); `reasoning_tokens` is next. The regression tests in `tests/server/test_regressions.py` state, per upstream bug class, whether the server handles or refuses the case.
