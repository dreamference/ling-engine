"""Images through ling-serve (specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.6, §20.7, §20.8). Skipped unless the server
says it reads images (/model_info has_image_understanding). The images are drawn here (a PNG written by hand with
zlib), so the tests need nothing but the standard library."""
import base64
import json
import struct
import zlib

import pytest

NO_THINK = {"enable_thinking": False}


def png(width, height, pixel):
    """A truecolour PNG whose pixel (x, y) is pixel(x, y) -> (r, g, b)."""
    raw = b"".join(b"\0" + bytes(c for x in range(width) for c in pixel(x, y)) for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    body = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + body).decode()


RED = png(256, 256, lambda x, y: (220, 30, 30))
BLUE = png(256, 256, lambda x, y: (30, 40, 220))
# 256 x 256: 16 x 16 patches, 64 image tokens. Stripes: something to look at.
STRIPES = png(320, 256, lambda x, y: (255, 255, 255) if (x // 32) % 2 else (0, 0, 0))


@pytest.fixture
def vision(server):
    status, info = server.request("/model_info")
    if status != 200 or not isinstance(info, dict) or not info.get("has_image_understanding"):
        pytest.skip("the server does not read images (--vision off)")
    return server


def chat(server, content, **kw):
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": kw.pop("max_tokens", 24),
            "chat_template_kwargs": NO_THINK}
    body.update(kw)
    return server.request("/v1/chat/completions", body)


def test_model_info_says_images(server):
    """The launcher reads has_image_understanding (Mightling's ling-rs/src/vision.rs)."""
    status, info = server.request("/model_info")
    assert status == 200 and isinstance(info.get("has_image_understanding"), bool), info
    status2, info2 = server.request("/get_model_info")
    assert status2 == 200 and info2.get("has_image_understanding") == info["has_image_understanding"]


def test_image_tokens_are_counted(vision):
    """A 256 x 256 image is 64 tokens plus <|vision_start|> and <|vision_end|> (§20.1)."""
    status, a = chat(vision, [{"type": "text", "text": "Say ok."}], max_tokens=1)
    assert status == 200, a
    status, b = chat(vision, [{"type": "image_url", "image_url": {"url": RED}}, {"type": "text", "text": "Say ok."}], max_tokens=1)
    assert status == 200, b
    assert b["usage"]["prompt_tokens"] - a["usage"]["prompt_tokens"] == 64 + 2


def test_the_model_sees_the_colour(vision):
    for url, word in ((RED, "red"), (BLUE, "blue")):
        status, out = chat(vision, [{"type": "image_url", "image_url": {"url": url}},
                                    {"type": "text", "text": "What colour is this image? Answer with one word."}])
        assert status == 200, out
        assert word in out["choices"][0]["message"]["content"].lower(), out


def test_two_images_are_not_confused_by_the_prefix_cache(vision):
    """§20.7: the same text with a different image must not reuse the first image's state."""
    answers = []
    for url in (RED, BLUE, RED):
        status, out = chat(vision, [{"type": "image_url", "image_url": {"url": url}},
                                    {"type": "text", "text": "What colour is this image? Answer with one word."}],
                           prompt_cache_key="vision-prefix-test")
        assert status == 200, out
        answers.append(out["choices"][0]["message"]["content"].lower())
    assert "red" in answers[0] and "blue" in answers[1] and "red" in answers[2], answers


def test_greedy_image_answer_is_deterministic_and_warm_equals_cold(vision):
    content = [{"type": "image_url", "image_url": {"url": STRIPES}},
               {"type": "text", "text": "Describe the pattern in this image in one sentence."}]
    outs = [chat(vision, content, max_tokens=32)[1] for _ in range(2)]
    assert outs[0]["choices"][0]["message"]["content"] == outs[1]["choices"][0]["message"]["content"]


def test_responses_input_image_and_view_image_output(vision):
    """Codex's shapes: input_image in a message, and an image in a function_call_output list (rendered on purpose,
    §20.6; production drops it)."""
    status, out = vision.request("/v1/responses", {"input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Open red.png."}]},
        {"type": "function_call", "call_id": "c1", "name": "view_image", "arguments": json.dumps({"path": "red.png"})},
        {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_image", "image_url": RED, "detail": "high"}]}],
        "tools": [{"type": "function", "name": "view_image", "description": "View an image.",
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
        "reasoning": {"effort": "none"}, "max_output_tokens": 40, "temperature": 0})
    assert status == 200, out
    text = json.dumps(out).lower()
    assert "red" in text, out


@pytest.mark.parametrize("url,code", [
    ("https://example.com/a.png", "image_url_not_supported"),
    ("data:image/gif;base64," + base64.b64encode(b"GIF89a" + b"\0" * 20).decode(), "unsupported_image_format"),
    ("data:image/png;base64,iVBORw0KGgoAAAA", "invalid_image"),
])
def test_bad_images_are_refused_with_a_code(vision, url, code):
    status, out = chat(vision, [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "hi"}])
    assert status == 400 and out["error"]["code"] == code, out


def test_image_pad_in_the_text_is_refused(vision):
    status, out = chat(vision, [{"type": "text", "text": "<|vision_start|><|image_pad|><|vision_end|> what?"}])
    assert status == 400 and out["error"]["code"] == "invalid_image", out


def test_streamed_image_request(vision):
    status, events = vision.stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": BLUE}},
                                                  {"type": "text", "text": "One word: what colour?"}]}],
        "stream": True, "temperature": 0, "max_tokens": 8, "chat_template_kwargs": NO_THINK})
    assert status == 200 and events[-1]["data"] == "[DONE]"
