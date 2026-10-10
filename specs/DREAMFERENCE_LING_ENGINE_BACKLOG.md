# ling-engine: Backlog from user requests

Part of the ling-engine specification; the index is [README.md](./README.md). Section numbers are those of the single-file `SPEC.md` this was split from (Oct 8, 2026 · revised the same day against Mightling's measured workload), so a citation of the form `§n` or `S§n` in the reports and the code still names the same text.

## 19. Backlog from user requests

What users of SGLang and vLLM asked for, and what the four clients send that ling-serve does not handle ([reports/api-compat-checklist.md](../reports/api-compat-checklist.md)). Ranked for one owner on one GB10 running 1–4 agents.

1. **Context overflow the clients can recognise.**
   - Chat completions and completions: HTTP 400 before any header, code `context_length_exceeded`, with a message that matches LiteLLM's patterns ("maximum context length is …").
   - The Responses API: `response.failed` with that code, the only shape Codex compacts on.

   Today Codex retries the same prompt five times and OpenHands for two minutes; neither compacts.
2. **Images.** Answer with a placeholder instead of a 400 until the vision encoder (M4) lands, and set `input_modalities: ["text"]` in the launcher's catalog meanwhile. Codex treats a 400 as the end of the turn.
3. **Stream fixes:**
   - hold back stop-string output to a character boundary;
   - end every started stream with a finish chunk and `[DONE]`;
   - validate before sending headers.
4. **Cancellation before work.** Check the cancel flag before dequeuing and between prefill chunks. With one worker, an abandoned 30-second prefill delays every agent.
5. **A NaN guard** ([section 18](./DREAMFERENCE_LING_ENGINE_VALIDATION.md)) with a counter in `/metrics`.
6. **Parser fixes** from [section 18](./DREAMFERENCE_LING_ENGINE_VALIDATION.md):
   - fences;
   - unoffered names;
   - prose before a call;
   - unclosed middle parameter;
   - unrenderable history arguments rendered instead of refused.
7. **Small request fixes:**
   - presence and repetition penalties over output tokens only;
   - `reasoning_effort: "none"` on chat completions;
   - `reasoning_tokens` counted (done 2026-10-10: from the token ids in the engine's loop, [API §3 and §4](./DREAMFERENCE_LING_ENGINE_API.md));
   - `reasoning` accepted as an alias of `reasoning_content` on input;
   - a negative `max_tokens` refused.
8. **Several resident sessions** ([section 11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)). Two interleaved agents evict each other's history today, and each then re-prefills from its last checkpoint. With several sessions resident, `prompt_cache_key` (Codex sends the thread id) is the natural session key. Upstream's equivalent complaint is lost prefix hits per turn ([vllm#53670](https://github.com/vllm-project/vllm/issues/53670), [vllm#53477](https://github.com/vllm-project/vllm/issues/53477)).
9. **Short requests ahead of long prefills.** Run a short request (autocomplete, a title) between the chunks of a long prefill ([sglang#42530](https://github.com/sgl-project/sglang/issues/42530)).
10. **Speculation measured at long context.** Plain against speculative decode at 8k, 32k, 128k and 200k ([vllm#54691](https://github.com/vllm-project/vllm/issues/54691): DFlash fell to 16 tokens/s against 71 at 185k), with an automatic switch-off if it ever loses.
11. **Reject what is not built.** Answer a 400 for `logprobs`, `n > 1` and `response_format` on the chat and completions routes until each exists. Not on `/v1/responses`: there the fields Codex sends and the engine does not act on are accepted and documented ([API §4](./DREAMFERENCE_LING_ENGINE_API.md), the owner's decision 2026-10-10), because Codex ends the turn on a 400; `tool_choice: "required"` stays unenforced on both routes. When structured output is built, it must still allow tool calls ([vllm#39929](https://github.com/vllm-project/vllm/issues/39929)), bound whitespace ([vllm#38696](https://github.com/vllm-project/vllm/issues/38696)) and mask every verify row ([vllm#60830](https://github.com/vllm-project/vllm/issues/60830)).
12. **A thinking budget** (force `</think>` after N tokens). Users ask for it on both engines ([sglang#25536](https://github.com/sgl-project/sglang/issues/25536)). OpenHands sends effort `high` on every call.
13. **Custom (freeform) tools for Codex. Done 2026-10-10** ([API §4](./DREAMFERENCE_LING_ENGINE_API.md)): a `type: custom` tool is offered to the model as a function with the one string parameter `input`, its grammar appended to its description as prose; the call comes back as a `custom_tool_call` item with the `custom_tool_call_input.delta/.done` events Codex parses, and `custom_tool_call` / `custom_tool_call_output` history replays as that call and its tool response (`ling-api-tests`; `test_responses_custom_tool_call_stream` and `test_responses_custom_tool_replay_matches_what_the_model_wrote` in `tests/server/`). Decided with the owner: the grammar rendered (description only was tried first: the model wrote a unified diff where apply_patch's syntax was wanted), no grammar enforcement, the whole input as one delta, Responses only (on chat completions, the vLLM-compatible route, a `type: custom` tool is dropped). **Open:** the grammar is rendered, not enforced; constrained decoding (item 11's structured output) is the remedy if the model still strays from apply_patch's syntax. Today's `ling` build sends function tools to this model in its default mode, so there the path is exercised by the tests alone until Mightling's launcher catalog offers the freeform `apply_patch`; under Code Mode `ling exec` sends its `exec` custom tool (a 177-byte grammar) today, and the operations stream runs that against `ling-serve` as the acceptance check (2026-10-10). Production SGLang drops custom tools under Code Mode: the same `ling exec` run against the M0 reference never produced a `custom_tool_call` in eleven turns, against ling-serve three in seven (the operations stream's capture, 2026-10-10; recordings under `reports/raw/code-mode-acceptance-2026-10-10/` once copied).
14. **`kernels.cu` made readable, with the SASS unchanged** (the user's request, 2026-10-09; no speed gain, so it is not in the ranked backlog). Each step is accepted only when `cuobjdump -sass` of the touched kernels is byte-identical before and after on second-puffin and `ling-kernel-tests` passes:
   - split the file by subsystem: `device_common.cuh` (`warp_sum`, `block_sum`, the FP8 converters, `silu`), then `gemv.cu` (both GEMVs and the two dequants, which share the FP4 lookup table), `attention.cu`, `gdn.cu`, `elementwise.cu`;
   - name the layouts: a small strided view built inside the kernel from the `__restrict__` parameter (never a struct member, where `restrict` is ignored) replaces the repeated `(pos * Hkv + kh) * D + d` arithmetic;
   - type the attention partial record: `struct AttnPartial { float m, l; float acc[256]; }` in place of `part[0]`, `part[1]`, `part[2 + d]` and the `(D + 2)` stride;
   - one online-softmax merge helper for the three places that rescale `(m, l, acc)`: per key, across warps, across splits;
   - a variadic launch helper that does `<<<>>>` and the launch check in one call, and one `dispatch_rows(M, f)` for the FP4 and FP8 row ladders;
   - argument structs for the wide signatures (`attn_prepare` takes seventeen parameters); a struct passed by value lands in the same constant bank;
   - a header comment per kernel: the thread mapping, each buffer's shape, the invariants the host checks.
   Not touched: the fixed-size register arrays with `#pragma unroll`, the issue-all-loads-first pattern, `__ldg` on `uint4`, `__launch_bounds__`, `fmaf` and `__expf`, the padded shared-memory stride. **Not pursued (the user's decision, 2026-10-09):** merging the FP4 and FP8 GEMV kernels into one body over a codec policy; it changes the loop structure and would need the bandwidth bench to accept, so the two kernels stay separate.
15. **Two API surfaces, no third (the owner's decision, 2026-10-10).** The goal is to match the Codex Responses API completely, and to keep a vLLM-compatible surface (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/metrics`) for the other clients. The gaps found on 2026-10-10 against the pinned Codex source, in order: custom tools (item 13, done 2026-10-10); `usage.output_tokens_details.reasoning_tokens` always 0, which Codex displays and budgets on (item 7, done 2026-10-10); `parallel_tool_calls`, `tool_choice: "required"`, `include`, `store`, `prompt_cache_key`, `reasoning.summary` and `text.verbosity` accepted and ignored (so no `reasoning_summary_text.*` events, which Codex tolerates; item 11 for the refusals, item 8 for `prompt_cache_key`); no rate-limit headers, so Codex's usage display stays empty (harmless). **Not pursued:** a high-performance protocol of ling-engine's own. The time is not in the HTTP layer (a token every 12 to 30 ms against microseconds of framing; M3's gap is the weight stream), every client switches engines by changing nothing but the server behind the port ([INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)), and a third surface is a third place to prove token-identity. Engine-specific routes stay additive beside the two surfaces (`/spec_stats`, the `ling:*` metrics, and later a session handle or "continue this prefix" for [SESSIONS §11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)), each justified by a measured number.
