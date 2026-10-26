"""Production's side of the image agreement study (VISION §20.5 item 5, §20.8 window 1). Talks to a reference SGLang
server (LING_REF_URL, default http://127.0.0.1:8000) running production's image and flags; standard library only.

  prod_vision.py render PROMPTS OUT          production's prompt ids for each request (chat completions,
                                             return_prompt_token_ids; the image tokens expanded) and its usage
  prod_vision.py gen PROMPTS RENDER OUT N    greedy continuation of N tokens with production's top-20 at each step
                                             (/generate on its own prompt ids and the images)
  prod_vision.py force PROMPTS RENDER GEN OUT  teacher-forced top-20 over the prompt and production's continuation
                                             (its prefill on the same positions: its own decode-vs-prefill noise)
PROMPTS: one chat-completions body per line (bench/vision/prompts.jsonl); image paths relative to its directory.
"""
import base64
import json
import os
import sys
import urllib.request

URL = os.environ.get("LING_REF_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("LING_REF_MODEL", "RadixArk/Qwen3.8-27B-NVFP4")
PAD = 248056
K = 20


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read() or b"{}")


def load(prompts_path):
    base = os.path.dirname(os.path.abspath(prompts_path))
    out = []
    for line in open(prompts_path):
        if not line.strip():
            continue
        body = json.loads(line)
        urls = []
        for m in body["messages"]:
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if "image_url" in part:
                        u = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
                        if not u.startswith("data:"):
                            data = open(os.path.join(base, u), "rb").read()
                            u = "data:image;base64," + base64.b64encode(data).decode()
                        part["image_url"] = {"url": u}
                        urls.append(u)
        out.append((body, urls))
    return out


def collapse(ids):
    """Production's expanded prompt ids with each image's run of <|image_pad|> back to one."""
    out = []
    for t in ids:
        if t == PAD and out and out[-1] == PAD:
            continue
        out.append(t)
    return out


def render(prompts_path, out_path):
    with open(out_path, "w") as f:
        for body, urls in load(prompts_path):
            b = dict(body, model=MODEL, max_tokens=1, temperature=0, return_prompt_token_ids=True, stream=False)
            r = post("/v1/chat/completions", b)
            ids = r.get("prompt_token_ids") or r["choices"][0].get("prompt_token_ids")
            f.write(json.dumps({"prompt_ids": ids, "short_ids": collapse(ids), "usage": r.get("usage"), "images": len(urls)}) + "\n")


def gen(prompts_path, render_path, out_path, n):
    items = load(prompts_path)
    rend = [json.loads(l) for l in open(render_path)]
    with open(out_path, "w") as f:
        for (body, urls), rd in zip(items, rend):
            r = post("/generate", {"input_ids": rd["short_ids"], "image_data": urls,
                                   "sampling_params": {"temperature": 0, "max_new_tokens": n, "stop_token_ids": [248046]},
                                   "return_logprob": True, "top_logprobs_num": K})
            m = r["meta_info"]
            cont = [t[1] for t in m["output_token_logprobs"]]
            top = [[[c[1], c[0]] for c in pos] for pos in m["output_top_logprobs"]]
            f.write(json.dumps({"cont": cont, "top": top, "text": r["text"], "prompt_tokens": m.get("prompt_tokens")}) + "\n")
            print(f"gen: {len(cont)} tokens, prompt {m.get('prompt_tokens')} (render {len(rd['prompt_ids'])})", flush=True)


def force(prompts_path, render_path, gen_path, out_path):
    items = load(prompts_path)
    rend = [json.loads(l) for l in open(render_path)]
    gens = [json.loads(l) for l in open(gen_path)]
    with open(out_path, "w") as f:
        for (body, urls), rd, g in zip(items, rend, gens):
            L = len(rd["prompt_ids"])
            r = post("/generate", {"input_ids": rd["short_ids"] + g["cont"], "image_data": urls,
                                   "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                                   "return_logprob": True, "logprob_start_len": L - 1, "top_logprobs_num": K})
            m = r["meta_info"]
            tops = m["input_top_logprobs"]
            top = [[[c[1], c[0]] for c in tops[j + 1]] for j in range(len(g["cont"]))]
            f.write(json.dumps({"top": top, "prompt_tokens": m.get("prompt_tokens")}) + "\n")
            print(f"force: prompt {m.get('prompt_tokens')} (want {L + len(g['cont'])})", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        render(sys.argv[2], sys.argv[3])
    elif cmd == "gen":
        gen(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == "force":
        force(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
