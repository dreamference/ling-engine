"""Fixtures for the server regression tests (tests/server).

The tests talk to a running ling-serve over HTTP, the way bench/serve_smoke.py does: urllib and
server-sent events parsed by hand, no client library. They never start a server.

Which server: LING_SERVE_URL (for example http://127.0.0.1:8300). There is no default port on purpose:
production SGLang answers the same endpoints on this machine, and these tests must not run against it.
Every test skips unless LING_SERVE_URL is set, /health answers, and /v1/models says the model is owned
by ling-engine.
"""
import json
import os
import urllib.error
import urllib.request

import pytest

TIMEOUT = float(os.environ.get("LING_SERVE_TIMEOUT", "900"))


class Server:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.model = None
        self.max_model_len = 0

    def request(self, path, body=None, method=None):
        """Returns (status, parsed JSON or text). HTTP errors are returned, not raised."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        text = raw.decode("utf-8")  # raises on invalid UTF-8, which is itself a failure
        try:
            return status, json.loads(text)
        except ValueError:
            return status, text

    def post(self, path, body):
        status, out = self.request(path, body)
        assert status == 200, f"{path} answered {status}: {out}"
        return out

    def stream(self, path, body):
        """Posts a streaming request. Returns (status, events): each event is a dict with the SSE `event`
        name (None for chat completions) and `data` (parsed JSON, or the string "[DONE]"). Every data
        line must be valid UTF-8 and valid JSON."""
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=TIMEOUT)
        except urllib.error.HTTPError as e:
            return e.code, [{"event": None, "data": e.read().decode("utf-8", "replace")}]
        events, name = [], None
        with r:
            for raw in r:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    payload = line[6:]
                    events.append({"event": name, "data": payload if payload == "[DONE]" else json.loads(payload)})
                    name = None
            return r.status, events

    def open_stream(self, path, body):
        """Starts a streaming request and returns the open response, for tests that disconnect early."""
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=TIMEOUT)


def _probe(url):
    s = Server(url)
    try:
        status, health = s.request("/health")
        if status != 200:
            return None, f"{url}/health answered {status}"
        status, models = s.request("/v1/models")
        if status != 200 or not isinstance(models, dict) or not models.get("data"):
            return None, f"{url}/v1/models answered {status}"
    except (OSError, urllib.error.URLError) as e:
        return None, f"no server at {url}: {e}"
    m = models["data"][0]
    if m.get("owned_by") != "ling-engine":
        return None, f"{url} is not ling-serve (owned_by={m.get('owned_by')!r}); refusing to test it"
    s.model = m["id"]
    s.max_model_len = int(m.get("max_model_len") or 0)
    return s, None


@pytest.fixture(scope="session")
def server():
    url = os.environ.get("LING_SERVE_URL")
    if not url:
        pytest.skip("LING_SERVE_URL is not set: no ling-serve to test")
    s, why = _probe(url)
    if s is None:
        pytest.skip(why)
    return s


def chat_text(resp):
    """The assistant content of a non-streaming chat completion ('' for None)."""
    return resp["choices"][0]["message"].get("content") or ""


def stream_chat_parts(events):
    """Concatenates a streamed chat completion: (content, reasoning, tool call deltas, finish reasons,
    usage chunks, saw [DONE])."""
    content, reasoning, calls, finishes, usage, done = "", "", [], [], [], False
    for e in events:
        d = e["data"]
        if d == "[DONE]":
            done = True
            continue
        assert "error" not in d, f"error event in the stream: {d}"
        if d.get("usage"):
            usage.append(d["usage"])
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            calls.extend(delta.get("tool_calls") or [])
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])
    return content, reasoning, calls, finishes, usage, done
