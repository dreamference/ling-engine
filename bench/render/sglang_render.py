"""Renders Responses API request bodies to prompt token ids exactly as production SGLang does.

Runs inside production's SGLang image (it imports SGLang's own request conversion and chat-template
code, so nothing is re-implemented here):

  docker exec -i m0-prod8000 python3 - MODEL_DIR TEMPLATE < sglang_render.py  (bodies on fd 3: see render_test.py)

or, simpler, with the files mounted: `python3 sglang_render.py MODEL_DIR TEMPLATE BODIES OUT`.

Each input line is {"body": <Responses request>, ...}; each output line is {"ids": [...], "text": "..."}:
the ids production's scheduler receives and the template's rendered text. The model is multimodal, so SGLang renders the template, encodes
it, decodes the ids back to text and lets the multimodal processor tokenize that text again; the ids
written are the result of that round trip.
"""
import asyncio
import json
import sys
from types import SimpleNamespace

from transformers import AutoTokenizer

import importlib
import pkgutil

import sglang.srt.entrypoints as entrypoints
from sglang.srt.parser.template_manager import TemplateManager


def _sglang_api():
    """SGLang's HTTP API modules, found by what they define: the request models and the chat and
    Responses serving classes (the ones the server itself runs)."""
    found = {}
    for info in pkgutil.walk_packages(entrypoints.__path__, entrypoints.__name__ + "."):
        leaf = info.name.rsplit(".", 1)[-1]
        if leaf not in ("protocol", "serving_chat", "serving_responses"):
            continue
        mod = importlib.import_module(info.name)
        if leaf == "protocol" and hasattr(mod, "ResponsesRequest"):
            found["ChatCompletionRequest"] = mod.ChatCompletionRequest
            found["ResponsesRequest"] = mod.ResponsesRequest
        for name, obj in vars(mod).items():
            if isinstance(obj, type) and obj.__module__ == mod.__name__:
                if name.endswith("ServingChat"):
                    found["chat"] = obj
                elif name.endswith("ServingResponses"):
                    found["responses"] = obj
    return found


API = _sglang_api()
ResponsesRequest = API["ResponsesRequest"]


class StubModelConfig:
    is_multimodal = True
    context_len = 262144

    def __init__(self, hf_config):
        self.hf_config = hf_config

    def get_default_sampling_params(self):
        return {}


def main():
    model_dir, template, bodies, out_path = sys.argv[1:5]
    tok = AutoTokenizer.from_pretrained(model_dir)
    # Keep the rendered template text too (before encoding): the "text" field below is decoded from the
    # ids, which carries the tokenizer's NFC normalization, so it cannot show a difference NFC hides.
    rendered = []
    apply = tok.apply_chat_template

    def apply_and_keep(*args, **kwargs):
        out = apply(*args, **kwargs)
        rendered.append(out)
        return out

    tok.apply_chat_template = apply_and_keep
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    server_args = SimpleNamespace(default_chat_template_kwargs=None, revision=None,
                                  tokenizer_metrics_allowed_custom_labels=None, tool_call_parser="qwen3_coder",
                                  reasoning_parser="qwen3", enable_custom_logit_processor=False)
    tm = SimpleNamespace(tokenizer=tok, server_args=server_args, model_config=StubModelConfig(hf_config),
                         model_path=model_dir, served_model_name="RadixArk/Qwen3.8-27B-NVFP4",
                         num_reserved_tokens=0,
                         config_value=lambda k: getattr(server_args, k, None))
    templates = TemplateManager()
    templates.load_chat_template(tm, template, model_dir)
    chat = API["chat"](tm, templates)
    resp = API["responses"].__new__(API["responses"])
    resp.tokenizer_manager = tm
    resp.serving_chat = chat
    resp.msg_store = {}
    for name in ("_process_messages", "_validate_media_content"):
        if not hasattr(resp, name) or name == "_process_messages":
            setattr(resp, name, getattr(chat, name))
    n = 0
    with open(bodies) as f, open(out_path, "w") as out:
        for line in f:
            rec = json.loads(line)
            req = ResponsesRequest(**rec["body"])
            messages, _, prompts, processed = asyncio.run(resp._make_request(req, None, tok))
            ids = processed.prompt_ids
            text = tok.decode(ids)
            ids2 = tok.encode(text, add_special_tokens=False)
            out.write(json.dumps({"ids": ids2, "text": rendered[-1] if rendered else text,
                                  "roundtrip_changed": ids2 != ids}) + "\n")
            rendered.clear()
            n += 1
    print(f"rendered {n} requests", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
