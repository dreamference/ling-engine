"""Regression tests for bug classes found in SGLang's and vLLM's issue trackers.

Each test names the upstream issue(s) it guards against, the input that triggered their bug, and the
check that would have caught it. The survey and the table of issues are in
reports/engine-issues-2026-10-09.md; specs/DREAMFERENCE_LING_ENGINE_VALIDATION.md §18 ("Known pitfalls") maps each class to its test.

They need a running ling-serve (see conftest.py: LING_SERVE_URL, and the server must say it is
ling-engine); without one every test skips. Tests marked xfail are known gaps: ling-serve does not do
the right thing yet, and the test states what right is. Checks that need two engine configurations
(speculation on and off, two process starts) are not reachable over HTTP and stay with
`ling-run --spec-check` and `--prefix-check`.

    LING_SERVE_URL=http://127.0.0.1:8300 python3 -m pytest tests/server -q
"""
import json
import re
import socket
import time
import urllib.parse

import pytest

from conftest import chat_text, stream_chat_parts

SHELL_TOOL = {"type": "function", "function": {
    "name": "shell", "description": "Runs a shell command and returns its output.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "array", "items": {"type": "string"}, "description": "The command and its arguments"},
        "workdir": {"type": "string"}}, "required": ["command"]}}}
SHELL_TOOL_RESPONSES = {"type": "function", "name": "shell", "description": SHELL_TOOL["function"]["description"],
                        "parameters": SHELL_TOOL["function"]["parameters"], "strict": False}
NO_THINK = {"enable_thinking": False}
TEMPLATE_TOKENS = ("<think>", "</think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>")


def degenerate(text):
    """The signature of NaN logits or a broken sampler: a run of token 0 ('!') or of one character."""
    return bool(re.search(r"!{8,}", text)) or bool(re.search(r"(.)\1{40,}", text))


def greedy_chat(server, messages, max_tokens=48, **extra):
    body = {"messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": NO_THINK}
    body.update(extra)
    return server.post("/v1/chat/completions", body)


def greedy_completion(server, prompt, max_tokens=24, **extra):
    body = {"prompt": prompt, "temperature": 0, "max_tokens": max_tokens}
    body.update(extra)
    return server.post("/v1/completions", body)


def evict(server):
    """Replaces the engine's resident history with an unrelated prompt, so the next request starts cold
    (ling-serve keeps one history and the checkpoints that lie inside it)."""
    greedy_completion(server, "Unrelated: list three prime numbers.", max_tokens=4)


def code_text(lines):
    return "".join(f"def f{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(lines))


def prompt_of_length(server, n):
    """A completions prompt of exactly n tokens: code text padded with ' the' (one token each),
    calibrated by the server's own usage count. Skips the test if calibration fails."""
    base = code_text(max(1, n // 24))
    got = greedy_completion(server, base, max_tokens=1)["usage"]["prompt_tokens"]
    while got > n:
        base = base[: int(len(base) * 0.9)]
        got = greedy_completion(server, base, max_tokens=1)["usage"]["prompt_tokens"]
    prompt = base + " the" * (n - got)
    if greedy_completion(server, prompt, max_tokens=1)["usage"]["prompt_tokens"] != n:
        pytest.skip(f"could not build a prompt of exactly {n} tokens")
    return prompt


# --- Correctness: prefix reuse, checkpoints and hybrid state ------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("n", [2048 + 1, 2048 + 31, 2048 + 32, 2048 + 33, 4096 + 16])
def test_prompt_length_near_chunk_boundary(server, n):
    """vllm#55766, vllm#53305, sglang#22935, sglang#38815.

    Their bug: prompts whose length falls just past a prefill-chunk or cache-block boundary produced NaN
    logits after a prefix-cache hit (output: '!' to max_tokens), and an identical resend lost its cache
    hit. Ours: checkpoints sit at multiples of 2048 and the last prefill chunk can be <= 32 rows (the
    rows path). Check: cold output, identical-resend output and the output of a prompt extending it all
    equal their cold references, none degenerate, and the resend reuses part of the prompt."""
    prompt = prompt_of_length(server, n)
    evict(server)
    cold = greedy_completion(server, prompt)
    warm = greedy_completion(server, prompt)
    assert not degenerate(cold["choices"][0]["text"])
    assert warm["choices"][0]["text"] == cold["choices"][0]["text"]
    assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, "identical resend reused nothing"
    longer = prompt + cold["choices"][0]["text"] + "\n\ndef g(y):\n"
    hit = greedy_completion(server, longer)
    evict(server)
    ref = greedy_completion(server, longer)
    assert hit["choices"][0]["text"] == ref["choices"][0]["text"]
    assert not degenerate(hit["choices"][0]["text"])


@pytest.mark.slow
def test_growing_conversation_warm_equals_cold(server):
    """vllm#60174, vllm#43559, vllm#53912, sglang#41351, vllm#58894.

    Their bug: in an agent session that grows request by request (each prompt extends the previous one,
    with speculation on), outputs after a prefix-cache hit diverged from a cold run, degenerated into '!'
    or lost drafter acceptance. Check: every turn's greedy answer equals the same request sent cold, and
    warm requests report cached tokens."""
    listing = "\n".join(f"-rw-r--r-- 1 dev dev {1000 + 37 * i} Oct  9 12:{i % 60:02d} module_{i}.py" for i in range(220))
    history = [
        {"role": "system", "content": "You are a coding agent. Use the shell tool to inspect the repository. "
                                      "Answer briefly. " + code_text(60)},
        {"role": "user", "content": "Find the largest Python module in the repository."},
    ]
    turns = []
    for k in range(3):
        call_id = f"call_{k}"
        history = history + [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "shell", "arguments": json.dumps({"command": ["ls", "-l", f"pkg{k}"]})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": listing.replace("module_", f"pkg{k}_mod_")},
        ]
        turns.append(list(history))
    evict(server)
    warm = [greedy_chat(server, msgs, tools=[SHELL_TOOL], max_tokens=40) for msgs in turns]
    for k, msgs in enumerate(turns):
        evict(server)
        cold = greedy_chat(server, msgs, tools=[SHELL_TOOL], max_tokens=40)
        w, c = warm[k]["choices"][0]["message"], cold["choices"][0]["message"]
        assert (w.get("content"), [(t["function"]["name"], t["function"]["arguments"]) for t in w.get("tool_calls") or []]) == \
               (c.get("content"), [(t["function"]["name"], t["function"]["arguments"]) for t in c.get("tool_calls") or []]), \
            f"turn {k}: warm and cold answers differ"
        assert not degenerate(w.get("content") or "")
        if k > 0:
            assert warm[k]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, f"turn {k} reused nothing"


def test_greedy_is_deterministic_across_repeats(server):
    """sglang#38009, vllm#54928, vllm#54521, sglang#39597.

    Their bug: greedy output with thinking on changed between identical requests (top-k tie order,
    autotuned kernels, verify numerics). Check: three identical greedy thinking-on requests, separated
    by an unrelated one, produce the same reasoning and content."""
    msgs = [{"role": "user", "content": "Write a Python function that merges two sorted lists, then explain it in one sentence."}]
    outs = []
    for _ in range(3):
        r = server.post("/v1/chat/completions", {"messages": msgs, "temperature": 0, "max_tokens": 400})
        m = r["choices"][0]["message"]
        outs.append((m.get("reasoning_content"), m.get("content")))
        evict(server)
    assert outs[0] == outs[1] == outs[2]


@pytest.mark.slow
def test_many_short_requests_do_not_degrade_later_ones(server):
    """sglang#37326, sglang#28679.

    Their bug: after many very short requests (health checks) the drafter's acceptance decayed to zero
    and outputs degenerated over uptime. Check: a fixed greedy request answers the same before and after
    60 short ones."""
    msgs = [{"role": "user", "content": "List the planets of the solar system in order, comma separated."}]
    before = chat_text(greedy_chat(server, msgs, max_tokens=60))
    for i in range(60):
        greedy_chat(server, [{"role": "user", "content": f"Say ok {i}."}], max_tokens=3)
    after = chat_text(greedy_chat(server, msgs, max_tokens=60))
    assert after == before


def test_long_non_english_prompt(server):
    """sglang#22087, vllm#54739.

    Their bug: a DeltaNet kernel produced garbled, mixed-language text for longer non-English prompts.
    Check: a ~3K-token Japanese prompt gives a non-degenerate greedy answer, the same cold and warm."""
    para = "東京は日本の首都であり、多くの人々が住んでいます。電車は時間通りに走り、街はとても清潔です。"
    msgs = [{"role": "user", "content": para * 60 + "\n上の文章を一文で要約してください。"}]
    evict(server)
    cold = chat_text(greedy_chat(server, msgs, max_tokens=64))
    warm = chat_text(greedy_chat(server, msgs, max_tokens=64))
    assert cold and not degenerate(cold)
    assert warm == cold


# --- Correctness: sampling ----------------------------------------------------------------------------


@pytest.mark.parametrize("truncation", [{"top_k": 1}, {"min_p": 1.0}, {"top_p": 1e-6}])
def test_truncation_to_one_token_equals_greedy(server, truncation):
    """vllm#42744.

    Their bug: under speculative decoding the rejection sampler skipped min_p (and logit_bias) on
    verified tokens. Check: at temperature 1, any truncation that leaves only the top token must give
    exactly the greedy output, which exercises the speculative accept path with a truncated target."""
    msgs = [{"role": "user", "content": "Describe a binary search in three sentences."}]
    greedy = chat_text(greedy_chat(server, msgs, max_tokens=96))
    body = {"messages": msgs, "temperature": 1.0, "max_tokens": 96, "chat_template_kwargs": NO_THINK, "seed": 7}
    body.update(truncation)
    assert chat_text(server.post("/v1/chat/completions", body)) == greedy


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_seed_reproduces(server, path):
    """sglang#15481.

    Their bug: seed had no effect on the completions endpoint. Check: on both endpoints the same seed
    reproduces a temperature-1 sample, and some other seed changes it."""
    body = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 40}
    if path.endswith("chat/completions"):
        body.update(messages=[{"role": "user", "content": "Invent a name for a cat."}], chat_template_kwargs=NO_THINK)
        text = chat_text
    else:
        body.update(prompt="A list of invented cat names:\n1.")
        text = lambda r: r["choices"][0]["text"]  # noqa: E731
    a = text(server.post(path, dict(body, seed=1234)))
    b = text(server.post(path, dict(body, seed=1234)))
    assert a == b
    others = {text(server.post(path, dict(body, seed=s))) for s in (1, 2, 3)}
    assert others != {a}


@pytest.mark.parametrize("penalty", [{"presence_penalty": 2.0}, {"repetition_penalty": 1.5}])
def test_penalties_ignore_prompt_tokens(server, penalty):
    """sglang#41124 (penalty history), with SGLang's semantics: presence, frequency and repetition
    penalties count generated tokens only.

    Trigger: a greedy answer that repeats a word from the prompt. Check: at the first generated position
    nothing has been generated yet, so a penalty must not change the first token."""
    msgs = [{"role": "user", "content": "The secret word is Zebra. Reply with only the secret word."}]
    plain = chat_text(greedy_chat(server, msgs, max_tokens=1))
    penalized = chat_text(greedy_chat(server, msgs, max_tokens=1, **penalty))
    assert penalized == plain


@pytest.mark.parametrize("top_k", [-1, 0])
def test_untruncated_sampling_is_not_degenerate(server, top_k):
    """sglang#36537 and production's own FlashInfer bug (specs/DREAMFERENCE_LING_ENGINE_INTEGRATION.md §12: the completions canary).

    Their bug: sampling with top_p 1 and no top_k (the completions default) returned token 0, '!', for
    every step; elsewhere a kernel picked on SM12x looped on token 0 with thinking + tools. Check: an
    untruncated sample on completions, and a thinking + tools chat, contain no '!' runs."""
    r = server.post("/v1/completions", {"prompt": "The capital of France is", "temperature": 1.0, "top_p": 1.0,
                                        "top_k": top_k, "max_tokens": 48, "seed": 11})
    assert not degenerate(r["choices"][0]["text"])
    r = server.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Check the disk usage of /tmp with the shell tool."}],
        "tools": [SHELL_TOOL], "temperature": 1.0, "top_p": 1.0, "top_k": top_k, "max_tokens": 400, "seed": 11})
    m = r["choices"][0]["message"]
    assert not degenerate((m.get("reasoning_content") or "") + (m.get("content") or ""))


def test_nonfinite_logits_are_counted(server):
    """vllm#53305, vllm#55291, sglang#33187.

    Their bug: a NaN or inf logits row was sampled (token 0, '!', to max_tokens, or a uniform draw
    streamed as a healthy answer). ling-serve ends such a request with an error and counts it. A NaN
    cannot be caused over HTTP (tests/sampling_tests.cpp injects one on the host). Check: /metrics
    exports the counter."""
    status, text = server.request("/metrics")
    assert status == 200 and "ling:nonfinite_logits_total" in text


# --- Correctness: reasoning and tool-call parsing, streaming ------------------------------------------


def test_no_template_tokens_leak(server):
    """vllm#51679, sglang#35083, vllm#49955, vllm#36435.

    Their bugs: </think> swallowed or leaked into content, reasoning streamed as content, an end-of-turn
    token at the end of every answer, tool-call XML streamed as text. Check: with thinking on and tools,
    neither content, reasoning nor arguments contain template tokens, and content holds no tool markup
    when a call was parsed."""
    r = server.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Show the current directory with the shell tool."}],
        "tools": [SHELL_TOOL], "temperature": 0, "max_tokens": 800})
    m = r["choices"][0]["message"]
    parts = [m.get("content") or "", m.get("reasoning_content") or ""]
    parts += [t["function"]["arguments"] for t in m.get("tool_calls") or []]
    for p in parts:
        for tok in TEMPLATE_TOKENS:
            assert tok not in p, f"{tok!r} leaked into {p[:200]!r}"
    if m.get("tool_calls"):
        assert "<tool_call>" not in (m.get("content") or "")
        for t in m["tool_calls"]:
            json.loads(t["function"]["arguments"])


def test_stream_matches_non_stream(server):
    """vllm#56263, sglang#34214, vllm#31501, vllm#55284.

    Their bugs: streaming and non-streaming disagreed (text after a call dropped in one mode, text before
    it truncated in the other, arguments lost when several tokens arrive per step). A speculative step
    emits up to 16 tokens at once. Check: greedy, thinking on, tools: identical reasoning, content, tool
    names and arguments, finish_reason and usage in both modes."""
    body = {"messages": [{"role": "user", "content": "Say one sentence about what you will do, then list /tmp with the shell tool."}],
            "tools": [SHELL_TOOL], "temperature": 0, "max_tokens": 800}
    full = server.post("/v1/chat/completions", body)
    status, events = server.stream("/v1/chat/completions", dict(body, stream=True, stream_options={"include_usage": True}))
    assert status == 200
    content, reasoning, calls, finishes, usage, done = stream_chat_parts(events)
    m = full["choices"][0]["message"]
    assert reasoning == (m.get("reasoning_content") or "")
    assert content == (m.get("content") or "")
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == \
           [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m.get("tool_calls") or []]
    assert finishes == [full["choices"][0]["finish_reason"]]
    assert usage and usage[-1]["completion_tokens"] == full["usage"]["completion_tokens"]
    assert usage[-1]["prompt_tokens"] == full["usage"]["prompt_tokens"]


def test_stream_chunk_shape(server):
    """sglang#29441, vllm#27572.

    Their bugs: an empty-content chunk before the tool-call chunks made AI-SDK clients end the turn; a
    stream ended with finish_reason null. Check: no chunk is an empty content string alone; tool-call
    deltas carry index, id, type and name with JSON arguments; exactly one finish_reason, then the usage
    chunk, then [DONE] last."""
    status, events = server.stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "List /tmp with the shell tool."}], "tools": [SHELL_TOOL],
        "temperature": 0, "max_tokens": 800, "stream": True, "stream_options": {"include_usage": True}})
    assert status == 200
    assert events and events[-1]["data"] == "[DONE]"
    datas = [e["data"] for e in events[:-1]]
    finish_at = [i for i, d in enumerate(datas) for ch in d.get("choices") or [] if ch.get("finish_reason")]
    assert len(finish_at) == 1
    usage_at = [i for i, d in enumerate(datas) if d.get("usage")]
    assert usage_at and usage_at[-1] > finish_at[0]
    for d in datas:
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            assert not (delta.get("content") == "" and not delta.get("tool_calls") and not delta.get("reasoning_content")
                        and not ch.get("finish_reason")), f"empty content chunk: {d}"
            for tc in delta.get("tool_calls") or []:
                assert isinstance(tc.get("index"), int) and tc.get("id") and tc.get("type") == "function"
                assert tc["function"].get("name")
                json.loads(tc["function"]["arguments"])


@pytest.mark.parametrize("stop", ["END", "完毕"])
def test_streaming_with_stop_and_multibyte_text(server, stop):
    """Bug class: streaming detokenization and stop strings with multi-byte text (sglang#40529 family;
    vLLM's and SGLang's incremental detokenizers both had partial-UTF-8 fixes). Continue's autocomplete
    sends stop lists on code with non-ASCII comments.

    Trigger: streamed chat, thinking off, Chinese output, a stop string. Check: no error event, every
    chunk is valid UTF-8 JSON, the stream's text equals the non-stream text, and the stop string is not
    in the output."""
    body = {"messages": [{"role": "user", "content": f"用中文写三句关于长城的话，然后写 {stop}，再写一句话。"}],
            "temperature": 0, "max_tokens": 160, "stop": [stop], "chat_template_kwargs": NO_THINK}
    full = chat_text(server.post("/v1/chat/completions", body))
    status, events = server.stream("/v1/chat/completions", dict(body, stream=True))
    assert status == 200
    content, _, _, finishes, _, done = stream_chat_parts(events)
    assert done and finishes
    assert content == full
    assert stop not in content


def test_tool_choice_none_returns_text(server):
    """vllm#55080.

    Their bug: tool_choice 'none' still ran the tool parser and returned content null. Check: tools
    offered with tool_choice 'none' give text content and no tool_calls."""
    r = server.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Call the shell tool to list /tmp."}], "tools": [SHELL_TOOL],
        "tool_choice": "none", "temperature": 0, "max_tokens": 200, "chat_template_kwargs": NO_THINK})
    m = r["choices"][0]["message"]
    assert not m.get("tool_calls")
    assert m.get("content")
    assert r["choices"][0]["finish_reason"] in ("stop", "length")


@pytest.mark.xfail(reason="render_chat refuses tool-call arguments that are not valid JSON (400)", strict=False)
def test_history_with_invalid_tool_arguments_is_accepted(server):
    """vllm#47761 (Continue), vllm#55495 (a Codex conversation).

    Their bug: once one assistant tool call in the history had arguments that were not valid JSON (cut
    off, or with a leaked tag), every later request of the conversation was a 400. Check: such a history
    is served."""
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Read src/main.py."},
                     {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                         "name": "shell", "arguments": "{\"command\": [\"cat\", \"src/ma"}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": "error: bad arguments"},
                     {"role": "user", "content": "Try again."}],
        "tools": [SHELL_TOOL], "temperature": 0, "max_tokens": 8, "chat_template_kwargs": NO_THINK})
    assert status == 200, out


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
def test_reasoning_efforts_chat(server, effort):
    """sglang#40789, vllm#52738.

    Their bug: an accepted effort value aborted the stream or was refused by the template. Production
    patches the template so high/max map to xhigh and minimal to low. Check: each value completes."""
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}], "reasoning_effort": effort,
        "temperature": 0, "max_tokens": 64})
    assert status == 200, out


@pytest.mark.xfail(reason="chat completions passes reasoning_effort 'none' to the template, which refuses it; "
                          "the Responses path maps it to thinking off", strict=False)
def test_reasoning_effort_none_chat(server):
    """vllm#53284, vllm#52738.

    Their bug: when 'none' closed the think block in the template but the parser was not told, the whole
    answer came back as reasoning. Check: reasoning_effort 'none' on chat completions answers in content
    with no reasoning."""
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number."}],
        "reasoning_effort": "none", "temperature": 0, "max_tokens": 32})
    assert status == 200, out
    m = out["choices"][0]["message"]
    assert "4" in (m.get("content") or "") and not m.get("reasoning_content")


def test_system_message_mid_conversation(server):
    """vllm#41114: the checkpoint's template raises on a system message that is not first; production
    patches it into a user <system-reminder>. Check: the request is served."""
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."},
                     {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Count to three."},
                     {"role": "system", "content": "Answer in French."}],
        "temperature": 0, "max_tokens": 32, "chat_template_kwargs": NO_THINK})
    assert status == 200, out


# --- The Responses API (the agent's endpoint) ---------------------------------------------------------


def responses_events(server, body):
    status, events = server.stream("/v1/responses", dict(body, stream=True))
    assert status == 200, events
    return [e for e in events if e["data"] != "[DONE]"]


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh"])
def test_responses_stream_event_order(server, effort):
    """sglang#40789, vllm#55284, vllm#36435.

    Their bugs: an effort value aborted the stream inside response.created; back-to-back calls in one
    burst crashed the stream; tool-call XML was streamed as output_text. Check: response.created first
    and response.completed last; sequence numbers count up by one; every item is added before its deltas
    and done after them; function_call items carry a call_id, a name and JSON arguments; output_text
    holds no tool markup."""
    evs = responses_events(server, {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text":
                   "Run `ls` and then `pwd` with the shell tool."}]}],
        "tools": [SHELL_TOOL_RESPONSES], "reasoning": {"effort": effort}, "temperature": 0, "max_output_tokens": 600})
    names = [e["event"] for e in evs]
    assert names[0] == "response.created" and names[-1] == "response.completed", names
    seqs = [e["data"]["sequence_number"] for e in evs]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))
    open_items, done_items = set(), set()
    for e in evs:
        d = e["data"]
        if e["event"] == "response.output_item.added":
            open_items.add(d["output_index"])
        elif e["event"].endswith(".delta"):
            assert d["output_index"] in open_items and d["output_index"] not in done_items
            if e["event"] == "response.output_text.delta":
                assert "<tool_call>" not in d["delta"] and "<function=" not in d["delta"]
        elif e["event"] == "response.output_item.done":
            assert d["output_index"] in open_items
            done_items.add(d["output_index"])
            item = d["item"]
            if item["type"] == "function_call":
                assert item.get("call_id") and item.get("name") == "shell"
                assert isinstance(json.loads(item["arguments"]), dict)
    assert open_items == done_items
    usage = evs[-1]["data"]["response"]["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


def test_responses_replayed_turn_renders_like_merged_chat(server):
    """sglang#42110, vllm#37167.

    Their bug: one assistant turn replayed as reasoning + commentary message + function calls was
    rendered as several assistant blocks, and the agent ended turns early. Check: the Responses request
    has exactly as many prompt tokens as the chat request with that turn merged into one assistant
    message (render_test.py checks the same against production on recorded sessions)."""
    calls = [("c1", ["ls"]), ("c2", ["pwd"])]
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Look around."}]},
             {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I should list files first."}]},
             {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking the tree."}]}]
    items += [{"type": "function_call", "call_id": c, "name": "shell", "arguments": json.dumps({"command": a})} for c, a in calls]
    items += [{"type": "function_call_output", "call_id": c, "output": out} for (c, _), out in zip(calls, ["a.txt", "/w"])]
    resp = server.post("/v1/responses", {"input": items, "max_output_tokens": 1, "temperature": 0})
    chat = server.post("/v1/chat/completions", {"messages": [
        {"role": "user", "content": "Look around."},
        {"role": "assistant", "content": "Checking the tree.", "reasoning_content": "I should list files first.",
         "tool_calls": [{"id": c, "type": "function", "function": {"name": "shell", "arguments": json.dumps({"command": a})}}
                        for c, a in calls]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
        {"role": "tool", "tool_call_id": "c2", "content": "/w"}], "max_tokens": 1, "temperature": 0})
    assert resp["usage"]["input_tokens"] == chat["usage"]["prompt_tokens"]


# Codex's freeform apply_patch, as its pinned source offers it: the description and the grammar from
# codex-rs/core/assets/tools/apply_patch.lark (rendered to the model with the description, API §4).
APPLY_PATCH_LARK = "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n\nhunk: add_hunk | delete_hunk | update_hunk\nadd_hunk: \"*** Add File: \" filename LF add_line+\ndelete_hunk: \"*** Delete File: \" filename LF\nupdate_hunk: \"*** Update File: \" filename LF change_move? change?\n\nfilename: /(.+)/\nadd_line: \"+\" /(.*)/ LF -> line\n\nchange_move: \"*** Move to: \" filename LF\nchange: (change_context | change_line)+ eof_line?\nchange_context: (\"@@\" | \"@@ \" /(.+)/) LF\nchange_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\neof_line: \"*** End of File\" LF\n\n%import common.LF\n"
APPLY_PATCH_CUSTOM = {"type": "custom", "name": "apply_patch",
                      "description": "The `apply_patch` tool can be used to edit files. This is a FREEFORM tool, "
                                     "so do not wrap the patch in JSON.",
                      "format": {"type": "grammar", "syntax": "lark", "definition": APPLY_PATCH_LARK}}
PATCH_PROMPT = ("Create the file hello.txt containing the single line `hello` by calling the apply_patch tool. "
                "Call the tool now; do not explain.")


def test_responses_custom_tool_call_stream(server):
    """BACKLOG §19.13: a `type: custom` tool was dropped and no custom_tool_call item ever emitted.

    Check, against what Codex's parser requires: the tool's call arrives as a custom_tool_call item,
    added with `input` already present (Codex drops an item without it), then exactly one
    custom_tool_call_input.delta carrying item_id and call_id, then .done, then output_item.done with the
    full input, the same item in response.completed; no tool markup in output_text; the input is the
    patch text itself, not JSON."""
    evs = responses_events(server, {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": PATCH_PROMPT}]}],
        "tools": [APPLY_PATCH_CUSTOM], "reasoning": {"effort": "none"}, "temperature": 0, "max_output_tokens": 400})
    names = [e["event"] for e in evs]
    assert names[0] == "response.created" and names[-1] == "response.completed", names
    for e in evs:
        if e["event"] == "response.output_text.delta":
            assert "<tool_call>" not in e["data"]["delta"] and "<function=" not in e["data"]["delta"]
    added = [e["data"] for e in evs if e["event"] == "response.output_item.added" and e["data"]["item"]["type"] == "custom_tool_call"]
    assert len(added) == 1, f"expected one custom_tool_call item, got {names}"
    item, index = added[0]["item"], added[0]["output_index"]
    assert item["name"] == "apply_patch" and item["call_id"] and item["id"] and item["input"] == ""
    deltas = [e["data"] for e in evs if e["event"] == "response.custom_tool_call_input.delta"]
    dones = [e["data"] for e in evs if e["event"] == "response.custom_tool_call_input.done"]
    assert len(deltas) == 1 and len(dones) == 1
    assert deltas[0]["item_id"] == item["id"] and deltas[0]["call_id"] == item["call_id"] and deltas[0]["output_index"] == index
    finished = [e["data"]["item"] for e in evs if e["event"] == "response.output_item.done" and e["data"]["output_index"] == index]
    assert len(finished) == 1 and finished[0]["type"] == "custom_tool_call" and finished[0]["status"] == "completed"
    text = finished[0]["input"]
    assert text == deltas[0]["delta"] == dones[0]["input"] and text.strip()
    # Free text, not JSON. The dialect is the model's to choose: the grammar is rendered, not enforced
    # (API §4); which one it wrote is recorded in reports/api-compat-checklist.md, not asserted here.
    assert "hello" in text and not text.lstrip().startswith("{"), text
    assert names.index("response.custom_tool_call_input.delta") < names.index("response.custom_tool_call_input.done")
    completed = evs[-1]["data"]["response"]["output"]
    assert any(o["type"] == "custom_tool_call" and o["input"] == text and o["call_id"] == item["call_id"] for o in completed)


def test_responses_custom_tool_replay_matches_what_the_model_wrote(server):
    """BACKLOG §19.13: a replayed custom_tool_call must render exactly as the model emitted it, or the
    next turn misses the prefix cache and (with speculation) can diverge from a cold run.

    Check: after the turn above, the next turn with the call and its output replayed as custom_tool_call /
    custom_tool_call_output answers warm (cached tokens reported) exactly as it answers cold; and the
    replay has as many prompt tokens as the same turn as chat messages with a one-parameter function
    call, which is how it is offered to the model. (The cached-token count itself is a checkpoint
    position, not the first divergent token, so it is not compared with the first prompt's length.)"""
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": PATCH_PROMPT}]}
    evict(server)
    first = server.post("/v1/responses", {"input": [user], "tools": [APPLY_PATCH_CUSTOM], "reasoning": {"effort": "none"},
                                          "temperature": 0, "max_output_tokens": 400})
    calls = [o for o in first["output"] if o["type"] == "custom_tool_call"]
    assert len(calls) == 1, [o["type"] for o in first["output"]]
    call = calls[0]
    replay = [user, {"type": "custom_tool_call", "call_id": call["call_id"], "name": "apply_patch", "input": call["input"]},
              {"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "Done!"}]
    body = {"input": replay, "tools": [APPLY_PATCH_CUSTOM], "reasoning": {"effort": "none"}, "temperature": 0,
            "max_output_tokens": 40}
    warm = server.post("/v1/responses", body)
    assert warm["usage"]["input_tokens_details"]["cached_tokens"] > 0, "the replayed turn reused nothing"
    evict(server)
    cold = server.post("/v1/responses", body)

    def answer(r):
        return [(o["type"], o.get("input") or "".join(c.get("text", "") for c in o.get("content", []))) for o in r["output"]]
    assert answer(warm) == answer(cold), "warm and cold answers differ after the replayed custom call"
    second = server.post("/v1/responses", dict(body, max_output_tokens=1))
    chat = server.post("/v1/chat/completions", {"messages": [
        {"role": "user", "content": PATCH_PROMPT},
        {"role": "assistant", "content": "", "tool_calls": [{"id": call["call_id"], "type": "function",
                                                              "function": {"name": "apply_patch", "arguments": json.dumps({"input": call["input"]})}}]},
        {"role": "tool", "tool_call_id": call["call_id"], "content": "Done!"}],
        "tools": [{"type": "function", "function": {
            "name": "apply_patch",
            "description": APPLY_PATCH_CUSTOM["description"] + "\n\nThe input must follow this lark grammar:\n" + APPLY_PATCH_LARK,
            "parameters": {"type": "object", "properties": {"input": {"type": "string", "description": "The tool's input, as free text"}},
                           "required": ["input"]}}}],
        "chat_template_kwargs": NO_THINK, "max_tokens": 1, "temperature": 0})
    assert second["usage"]["input_tokens"] == chat["usage"]["prompt_tokens"]


def test_responses_reasoning_tokens_counted(server):
    """vllm#49711, sglang#39826.

    Their bugs: reasoning_tokens reported 0 when the prompt opened the thinking span; under speculation
    it exceeded output_tokens when an accepted block crossed EOS. Check: with thinking on, 0 <
    reasoning_tokens < output_tokens on both routes (the answer after the span is at least one token);
    with thinking off, 0."""
    q = "Is 91 prime? Think, then answer yes or no."
    r = server.post("/v1/responses", {"input": q, "reasoning": {"effort": "low"}, "temperature": 0, "max_output_tokens": 600})
    u = r["usage"]
    assert any(item["type"] == "reasoning" for item in r["output"])
    assert 0 < u["output_tokens_details"]["reasoning_tokens"] < u["output_tokens"], u
    c = server.post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "reasoning_effort": "low",
                                             "temperature": 0, "max_tokens": 600})
    cu = c["usage"]
    assert c["choices"][0]["message"].get("reasoning_content")
    assert 0 < cu["completion_tokens_details"]["reasoning_tokens"] < cu["completion_tokens"], cu
    off = server.post("/v1/responses", {"input": "Say yes.", "reasoning": {"effort": "none"}, "temperature": 0, "max_output_tokens": 8})
    assert off["usage"]["output_tokens_details"]["reasoning_tokens"] == 0


def test_metrics_has_the_harness_gauges(server):
    """The benchmark harness's admission reads sglang:num_running_reqs and sglang:num_queue_reqs from
    /metrics and refuses to start while another client's request runs; its parallelism divides
    sglang:max_total_num_tokens by its per-task budget (API §6). Check: the three gauges are there,
    running is 1 while a request streams and 0 after, and the pool equals max_model_len."""
    def gauges():
        status, text = server.request("/metrics")
        assert status == 200
        out = {}
        for line in text.splitlines():
            if line.startswith("sglang:num_running_reqs") or line.startswith("sglang:num_queue_reqs") or \
                    line.startswith("sglang:max_total_num_tokens"):
                name, value = line.split(" ")[0].split("{")[0], float(line.rsplit(" ", 1)[1])
                out[name] = value
        return out
    idle = gauges()
    assert set(idle) == {"sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:max_total_num_tokens"}, idle
    assert idle["sglang:num_running_reqs"] == 0 and idle["sglang:num_queue_reqs"] == 0
    assert idle["sglang:max_total_num_tokens"] == server.max_model_len
    r = server.open_stream("/v1/completions", {"prompt": "Count from one to two hundred, separated by commas: 1, 2, 3,",
                                               "max_tokens": 300, "temperature": 0, "stream": True})
    try:
        r.readline()  # the first chunk: the request is running
        busy = gauges()
        assert busy["sglang:num_running_reqs"] == 1, busy
    finally:
        r.close()
    time.sleep(1)
    assert gauges()["sglang:num_running_reqs"] == 0


# --- Errors, limits, cancellation ---------------------------------------------------------------------


def test_context_overflow_is_recognisable(server):
    """The agents decide whether to compact the conversation from this error, and each recognises a
    different shape (reports/api-compat-checklist.md, section B):
    - Codex compacts only on an in-stream response.failed whose error.code is context_length_exceeded;
      an HTTP 400 is terminal for it and a server_error is retried five times with the same prompt.
    - Cline wants 400/413/422 with code context_length_exceeded or a message such as "maximum context".
    - OpenHands (through LiteLLM) matches message substrings such as "maximum context length is".
    Check: completions and chat answer HTTP 400 with code context_length_exceeded and a message LiteLLM
    recognises, before any stream starts; the streamed Responses API ends with response.failed carrying
    that code, and the non-streamed one answers HTTP 400 with it."""
    n = server.max_model_len + 64
    status, out = server.request("/v1/completions", {"prompt": [785] * n, "max_tokens": 4})
    assert status == 400 and out["error"].get("code") == "context_length_exceeded", out
    assert "maximum context length is" in out["error"]["message"]
    text = "word " * n
    status, events = server.stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": text}], "max_tokens": 4, "stream": True})
    assert status == 400, events[-1:]
    assert "context_length_exceeded" in str(events[0]["data"])
    status, events = server.stream("/v1/responses", {"input": text, "max_output_tokens": 4, "stream": True})
    assert status == 200
    failed = [e["data"] for e in events if e["event"] == "response.failed"]
    assert failed and failed[0]["response"]["error"]["code"] == "context_length_exceeded", events[-1:]
    assert "maximum context length is" in failed[0]["response"]["error"]["message"]
    status, out = server.request("/v1/responses", {"input": text, "max_output_tokens": 4})
    assert status == 400 and out["error"].get("code") == "context_length_exceeded", out


def test_absurd_values_leave_the_server_healthy(server):
    """sglang#41482: huge top_k, logprobs or n passed validation and killed the server. Check: such
    requests get a 400 or a bounded answer, a wrong type gets a 400, and /health still answers."""
    for extra in ({"top_k": 2147483648}, {"n": 2147483648}, {"top_k": -2147483648}, {"seed": 2 ** 64 - 1}):
        status, _ = server.request("/v1/completions", dict({"prompt": "Hello", "max_tokens": 4}, **extra))
        assert status in (200, 400), extra
    status, _ = server.request("/v1/completions", {"prompt": "Hello", "max_tokens": 4, "temperature": "hot"})
    assert status == 400
    status, _ = server.request("/v1/chat/completions", {"messages": "not a list", "max_tokens": 4})
    assert status == 400
    assert server.request("/health")[0] == 200


@pytest.mark.slow
def test_disconnect_frees_the_engine(server):
    """sglang#36333, sglang#34113.

    Their bug: a streaming client that disconnected left a zombie request decoding to max_tokens,
    delaying everything after it. ling-serve runs one request at a time, so a zombie blocks the next
    request outright. Check: after dropping a long stream at its first event, a short request finishes
    quickly."""
    body = {"messages": [{"role": "user", "content": "Write a 2000-word essay on the history of the printing press."}],
            "max_tokens": 3000, "temperature": 0, "stream": True}
    r = server.open_stream("/v1/chat/completions", body)
    r.readline()
    # Close the socket itself, not just the response object, so the server sees the disconnect.
    try:
        r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    except (AttributeError, OSError):
        pass
    r.close()
    t0 = time.monotonic()
    greedy_chat(server, [{"role": "user", "content": "Say ok."}], max_tokens=4)
    assert time.monotonic() - t0 < 20, "the abandoned request kept the engine busy"


# --- Request features the clients send that ling-serve ignores (reports/api-compat-checklist.md) -----


@pytest.mark.xfail(reason="logprobs are not implemented and the field is ignored", strict=False)
def test_logprobs_are_returned_or_refused(server):
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hi."}], "logprobs": True, "top_logprobs": 2, "max_tokens": 4,
        "chat_template_kwargs": NO_THINK})
    assert status == 400 or (out["choices"][0].get("logprobs") or {}).get("content")


@pytest.mark.xfail(reason="n is ignored: one choice comes back", strict=False)
def test_n_is_honoured_or_refused(server):
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hi."}], "n": 2, "max_tokens": 4, "chat_template_kwargs": NO_THINK})
    assert status == 400 or len(out["choices"]) == 2


@pytest.mark.xfail(reason="response_format is ignored (no structured output)", strict=False)
def test_response_format_is_honoured_or_refused(server):
    """vllm#38696, vllm#39929: when built, a json_schema must still allow tool calls and must not allow
    unbounded whitespace."""
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Give me a city and its country."}], "max_tokens": 200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
            "type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"], "additionalProperties": False}}},
        "chat_template_kwargs": NO_THINK})
    if status == 200:
        obj = json.loads(chat_text(out))
        assert set(obj) == {"city", "country"}
    else:
        assert status == 400


@pytest.mark.xfail(reason="tool_choice 'required' is not enforced", strict=False)
def test_tool_choice_required_yields_a_call(server):
    """vllm#38106, sglang#27336: with tool_choice 'required' the answer must be a tool call (and, when a
    grammar enforces it, string values must never contain a stray </parameter)."""
    out = server.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Hello! How are you today?"}], "tools": [SHELL_TOOL],
        "tool_choice": "required", "temperature": 0, "max_tokens": 300, "chat_template_kwargs": NO_THINK})
    calls = out["choices"][0]["message"].get("tool_calls") or []
    assert calls and all("</parameter" not in c["function"]["arguments"] for c in calls)


@pytest.mark.xfail(reason="an image part is refused with HTTP 400 ('does not accept images or video')", strict=False)
def test_image_input_does_not_fail_the_turn(server):
    """reports/api-compat-checklist.md: Codex's model catalog advertises image input, and Cline's default
    model info claims it, so a pasted screenshot reaches the server. Codex treats an HTTP 400 as terminal
    for the turn. Check: until the vision encoder exists, an image part is answered (for example with a
    placeholder in its place), not refused."""
    png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    status, out = server.request("/v1/responses", {"input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "What is in this image?"}, {"type": "input_image", "image_url": png}]}],
        "reasoning": {"effort": "none"}, "max_output_tokens": 16})
    assert status == 200, out
