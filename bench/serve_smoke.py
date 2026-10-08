#!/usr/bin/env python3
"""Smoke-tests a running ling-serve. It covers:
- health and models;
- chat, both non-streaming and streaming;
- reasoning split;
- a tool call with typed arguments;
- completions from token ids.

    serve_smoke.py [--url http://127.0.0.1:8200]
"""
import argparse
import json
import sys
import urllib.request

SHELL_TOOL = {"type": "function", "function": {
    "name": "shell", "description": "Runs a shell command and returns its output.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "array", "items": {"type": "string"}, "description": "The command and its arguments"},
        "timeout_ms": {"type": "integer"}}, "required": ["command"]}}}


def call(url, path, body=None, stream=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=900)
    if not stream:
        raw = r.read().decode()
        return json.loads(raw) if raw.strip().startswith(("{", "[")) else raw
    events = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    url = ap.parse_args().url
    ok = True

    def check(cond, what):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok &= bool(cond)

    check(call(url, "/health") == "ok", "/health")
    models = call(url, "/v1/models")
    check(models["data"][0].get("max_model_len", 0) > 0, f"/v1/models: {models['data'][0]['id']}")

    r = call(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17 * 3? Answer with the number."}],
                                           "temperature": 0, "max_tokens": 400})
    msg = r["choices"][0]["message"]
    check("51" in (msg.get("content") or ""), f"chat answer: {msg.get('content')!r}")
    check(bool(msg.get("reasoning_content")), f"reasoning split ({len(msg.get('reasoning_content') or '')} chars)")
    check(r["usage"]["completion_tokens"] > 0, f"usage {r['usage']}")

    ev = call(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello in French."}],
                                            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0,
                                            "max_tokens": 30, "stream": True, "stream_options": {"include_usage": True}},
              stream=True)
    text = "".join((e["choices"][0]["delta"].get("content") or "") for e in ev if e.get("choices"))
    check("onjour" in text, f"streaming content: {text!r} in {len(ev)} events")
    check(any(e.get("usage") for e in ev), "streamed usage")

    r = call(url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "List the files in /tmp using the shell tool."}],
        "tools": [SHELL_TOOL], "temperature": 0, "max_tokens": 800})
    ch = r["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    check(ch["finish_reason"] == "tool_calls" and calls, f"tool call: finish={ch['finish_reason']}")
    if calls:
        args = json.loads(calls[0]["function"]["arguments"])
        check(calls[0]["function"]["name"] == "shell" and isinstance(args.get("command"), list),
              f"typed arguments: {args}")

    ev = call(url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Show the current directory with the shell tool."}],
        "tools": [SHELL_TOOL], "temperature": 0, "max_tokens": 800, "stream": True}, stream=True)
    streamed_calls = [tc for e in ev if e.get("choices") for tc in (e["choices"][0]["delta"].get("tool_calls") or [])]
    finish = [e["choices"][0]["finish_reason"] for e in ev if e.get("choices") and e["choices"][0]["finish_reason"]]
    check(streamed_calls and finish == ["tool_calls"], f"streamed tool call {[c['function'] for c in streamed_calls]}")

    r = call(url, "/v1/completions", {"prompt": [785, 6722, 315, 9625, 374], "max_tokens": 8, "temperature": 0})
    check(len(r["choices"][0]["text"]) > 0, f"completions from ids: {r['choices'][0]['text']!r}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
