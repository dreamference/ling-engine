#!/usr/bin/env python3
"""Checks ling-engine v0's greedy output against the production SGLang server on the same machine.

For each prompt, ling-run generates N tokens greedily. The prompt plus those tokens are then sent to
SGLang's native /generate endpoint, teacher-forced, asking for the top-5 candidates at every position.
For each generated position the script reports:
- top1: our token is SGLang's argmax at that position;
- top5: our token is among SGLang's top five;
- our token's log-probability under SGLang.

The two servers quantize activations differently (SGLang runs NVFP4 and FP8 GEMMs on quantized
activations, v0 dequantizes weights and keeps activations in FP32), so near-ties can flip. A correct
engine shows top1 well above 90% and top5 near 100%. A broken layer shows near-zero agreement within a
few tokens.

    validate_v0.py --ling-run build/ling-run --model DIR [--sglang http://127.0.0.1:8000] [--tokens 48]
"""
import argparse
import json
import subprocess
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>user\nWrite a C++ function that reverses a singly linked list.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "Die Hauptstadt von Deutschland ist Berlin. Die Hauptstadt von Italien ist",
    "import numpy as np\n\n# Compute the moving average of an array\ndef moving_average(a, w):\n",
]


def sglang_logprobs(url, ids, prompt_len):
    body = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": prompt_len - 1,
        "top_logprobs_num": 5,
    }
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        meta = json.load(r)["meta_info"]
    return meta["input_token_logprobs"], meta["input_top_logprobs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ling-run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sglang", default="http://127.0.0.1:8000")
    ap.add_argument("--tokens", type=int, default=48)
    args = ap.parse_args()

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for prompt in PROMPTS:
            f.write(json.dumps(prompt) + "\n")
    run = subprocess.run([args.ling_run, "--model", args.model, "--prompts-file", f.name, "--max-tokens",
                          str(args.tokens), "--ids-out", "--prompt-ids-out"],
                         capture_output=True, text=True, check=True)
    lines = run.stdout.strip().splitlines()
    total = top1 = top5 = 0
    lp_sum = 0.0
    for n, prompt in enumerate(PROMPTS):
        prompt_ids = [int(x) for x in lines[2 * n].split(",")]
        gen = [int(x) for x in lines[2 * n + 1].split(",") if x]
        token_lps, tops = sglang_logprobs(args.sglang, prompt_ids + gen, len(prompt_ids))
        # Entries are aligned with ids[prompt_len - 1:]; the entry for position i scores ids[i].
        p_top1 = p_top5 = 0
        for j, tok in enumerate(gen):
            idx = j + 1
            cands = [c[1] for c in tops[idx]] if tops[idx] else []
            p_top1 += bool(cands) and cands[0] == tok
            p_top5 += tok in cands
            lp_sum += token_lps[idx][0] or 0.0
        total += len(gen)
        top1 += p_top1
        top5 += p_top5
        print(f"{len(gen):3d} tokens  top1 {p_top1 / max(len(gen), 1):6.1%}  top5 {p_top5 / max(len(gen), 1):6.1%}  "
              f"{prompt[:50]!r}")
    print(f"ALL {total} tokens: top1 {top1 / total:.1%}  top5 {top5 / total:.1%}  "
          f"mean logprob under SGLang {lp_sum / total:.3f}")
    return 0 if top5 / total > 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
