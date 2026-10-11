"""Resident sessions and the interleaved scheduler (M4 stages 1 and 2:
specs/drafts/M4-sessions-and-batches-2026-10-10.md §1, §3, §6, §7).

ling-serve keeps --sessions (default 2) resident sessions, each with its own KV cache and state, and runs
each request in the session it continues, else in a new session that takes a free slot or evicts the least
recently used one. Sessions bound with a prompt_cache_key whose last request ran less than
--session-protect-seconds ago are never evicted: a new session's request waits instead. Up to --batch
(default 2) requests run at once, each in its own session, taking turns one prefill chunk or decode step at
a time.

The queueing test runs last and leaves its two keyed sessions protected for the protection window, so a
new conversation sent right after this file (a re-run of the suite) waits up to that long.

    LING_SERVE_URL=http://127.0.0.1:8300 python3 -m pytest tests/server -q
"""
import json
import socket
import threading
import time

import pytest

from test_regressions import NO_THINK, code_text, evict, greedy_chat


def metrics(server):
    status, text = server.request("/metrics")
    assert status == 200
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            out[line.split(" ")[0].split("{")[0]] = float(line.rsplit(" ", 1)[1])
    return out


def conversation(tag, lines=90):
    """A conversation of about 2,500 tokens whose first tokens are its own (no prefix shared with another
    tag's beyond the template's first few tokens)."""
    return [{"role": "system", "content": f"Session {tag}. You review Python modules. Answer in one short line.\n"
                                          + code_text(lines).replace("def f", f"def {tag}_f")},
            {"role": "user", "content": f"Which {tag} function returns the largest value for x = 3?"}]


def next_turn(messages, reply, question):
    return messages + [{"role": "assistant", "content": reply["choices"][0]["message"].get("content") or ""},
                       {"role": "user", "content": question}]


def cached(resp):
    return resp["usage"]["prompt_tokens_details"]["cached_tokens"]


def test_pool_gauge_counts_every_session(server):
    """The harness divides sglang:max_total_num_tokens by its per-task budget (API §6) to decide how many
    tasks it runs at once. Check: the pool is the resident sessions times the context, 131,072 with the
    production flags (two sessions at 65,536), so the harness admits two tasks."""
    m = metrics(server)
    assert m["ling:sessions"] >= 2, m
    assert m["sglang:max_total_num_tokens"] == m["ling:sessions"] * server.max_model_len, m
    if server.max_model_len == 65536 and m["ling:sessions"] == 2:
        assert m["sglang:max_total_num_tokens"] == 131072
        assert int(m["sglang:max_total_num_tokens"]) // 49152 == 2  # the harness's per-task budget


@pytest.mark.slow
def test_two_alternating_conversations_stay_warm(server):
    """BACKLOG §19.8: with one resident history, two agent tasks that alternate evicted each other's
    history, so every turn re-prefilled from the shared system prompt. Check: conversations A and B
    alternate (A, B, A, B); each second turn reuses at least 90% of its first turn's prompt, and its
    answer equals the same request sent cold."""
    a = conversation("alpha")
    b = conversation("beta")
    evict(server)
    a1 = greedy_chat(server, a, max_tokens=24)
    b1 = greedy_chat(server, b, max_tokens=24)
    a2_msgs = next_turn(a, a1, "And which one returns the smallest value?")
    b2_msgs = next_turn(b, b1, "And which one returns the smallest value?")
    a2 = greedy_chat(server, a2_msgs, max_tokens=24)
    b2 = greedy_chat(server, b2_msgs, max_tokens=24)
    for first, second, name in ((a1, a2, "A"), (b1, b2, "B")):
        assert cached(second) >= 0.9 * first["usage"]["prompt_tokens"], \
            f"{name}'s second turn reused {cached(second)} of {first['usage']['prompt_tokens']} tokens: evicted"
    evict(server)
    a2_cold = greedy_chat(server, a2_msgs, max_tokens=24)
    evict(server)
    b2_cold = greedy_chat(server, b2_msgs, max_tokens=24)
    assert cached(a2_cold) < 0.5 * a2_cold["usage"]["prompt_tokens"], "the cold reference was not cold"
    assert a2["choices"][0]["message"] == a2_cold["choices"][0]["message"], "A's warm turn differs from cold"
    assert b2["choices"][0]["message"] == b2_cold["choices"][0]["message"], "B's warm turn differs from cold"


@pytest.mark.slow
def test_two_requests_run_at_once_and_each_equals_alone(server):
    """M4 §3 (stage 2): up to --batch requests run at once, each in its own resident session, the
    scheduler interleaving their prefill chunks and decode steps. Check: two streamed greedy requests with
    unrelated prompts both run (num_running_reqs 2) and both stream text before either finishes; each
    answer equals the same request sent alone afterwards (nothing numeric changes when they interleave)."""
    m = metrics(server)
    if m.get("ling:batch", 1) < 2:
        pytest.skip("this server runs one request at a time")
    bodies = [{"messages": [{"role": "system", "content": f"Module {tag}.\n" + code_text(60).replace("def f", f"def {tag}_f")},
                            {"role": "user", "content": f"Write out every {tag} function's body, one line each, in order."}],
               "max_tokens": 160, "temperature": 0, "stream": True, "chat_template_kwargs": NO_THINK}
              for tag in ("left", "right")]
    evict(server)
    out = [{"text": "", "first": None, "done": None, "error": None} for _ in bodies]

    def stream(i):
        try:
            r = server.open_stream("/v1/chat/completions", bodies[i])
            with r:
                for raw in r:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    for ch in json.loads(line[6:]).get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if piece and out[i]["first"] is None:
                            out[i]["first"] = time.monotonic()
                        out[i]["text"] += piece
            out[i]["done"] = time.monotonic()
        except Exception as e:  # reported below, on the test's thread
            out[i]["error"] = repr(e)

    threads = [threading.Thread(target=stream, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    running = 0
    while any(t.is_alive() for t in threads):
        running = max(running, metrics(server)["sglang:num_running_reqs"])
        time.sleep(0.1)
    for t in threads:
        t.join()
    assert not out[0]["error"] and not out[1]["error"], out
    assert running == 2, f"never two requests running (max {running})"
    assert max(o["first"] for o in out) < min(o["done"] for o in out), "the two requests did not overlap"
    for i, body in enumerate(bodies):
        alone = greedy_chat(server, body["messages"], max_tokens=body["max_tokens"])
        assert (alone["choices"][0]["message"].get("content") or "") == out[i]["text"], \
            f"request {i}'s answer alone differs from its answer beside the other"
    assert metrics(server)["sglang:num_running_reqs"] == 0


@pytest.mark.slow
def test_third_conversation_queues_while_both_sessions_are_busy(server):
    """M4 §7 question 6: a session with a request in the last few minutes is never evicted, so a third
    client queues instead of evicting a task's history while its tool runs. Check: with every slot held
    by a keyed conversation (prompt_cache_key, as Codex sends its thread id) used just now, a third keyed
    conversation waits (num_queue_reqs 1, nothing running, no eviction) while a next turn of the first
    conversation still runs, warm; when the third client disconnects its request leaves the queue."""
    m = metrics(server)
    if m.get("ling:session_protect_seconds", 0) < 30:
        pytest.skip("session protection is off or too short on this server")
    n = int(m["ling:sessions"])
    convs = [conversation(f"busy{i}", lines=40) for i in range(n)]
    first = [greedy_chat(server, c, max_tokens=8, prompt_cache_key=f"m4-busy-{i}") for i, c in enumerate(convs)]
    evictions = metrics(server)["ling:session_evictions_total"]
    third = server.open_stream("/v1/chat/completions", {
        "messages": conversation("third", lines=40), "max_tokens": 8, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}, "prompt_cache_key": "m4-third"})
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            g = metrics(server)
            if g["sglang:num_queue_reqs"] == 1 and g["sglang:num_running_reqs"] == 0:
                break
            time.sleep(0.2)
        time.sleep(2)
        g = metrics(server)
        assert g["sglang:num_queue_reqs"] == 1 and g["sglang:num_running_reqs"] == 0, g
        # The first conversation's next turn runs past the waiting request, in its own session.
        t0 = time.monotonic()
        nxt = greedy_chat(server, next_turn(convs[0], first[0], "And for x = 4?"), max_tokens=8,
                          prompt_cache_key="m4-busy-0")
        assert time.monotonic() - t0 < 60
        assert cached(nxt) >= 0.9 * first[0]["usage"]["prompt_tokens"], "the busy session was not kept"
        g = metrics(server)
        assert g["sglang:num_queue_reqs"] == 1, g
        assert g["ling:session_evictions_total"] == evictions, "a busy session was evicted"
    finally:
        try:
            third.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except (AttributeError, OSError):
            pass
        third.close()
    # The server learns of the disconnect when it next writes to the stream: the keep-alive, every 15 s by
    # default (--keepalive-seconds), so allow two intervals.
    deadline = time.monotonic() + 45
    while metrics(server)["sglang:num_queue_reqs"] != 0 and time.monotonic() < deadline:
        time.sleep(0.2)
    assert metrics(server)["sglang:num_queue_reqs"] == 0, "the abandoned request stayed in the queue"
