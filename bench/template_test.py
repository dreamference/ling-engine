#!/usr/bin/env python3
"""Compares ling-template (the chat template in C++) with the checkpoint's Jinja template rendered by
transformers, byte for byte, on conversations shaped like an agent's.

    template_test.py --ling-template build/ling-template --model DIR
"""
import argparse
import json
import subprocess
import sys

from transformers import AutoTokenizer

TOOLS = [
    {"type": "function", "function": {"name": "shell", "description": "Run a command",
                                      "parameters": {"type": "object", "properties": {
                                          "command": {"type": "array", "items": {"type": "string"}},
                                          "workdir": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "apply_patch", "description": "Edit files — UTF-8 ✓",
                                      "parameters": {"type": "object", "properties": {"input": {"type": "string"}}}}},
]
CONVS = [
    [{"role": "user", "content": "Hello"}],
    [{"role": "system", "content": "  You are a coding agent.  "}, {"role": "user", "content": "List files"}],
    [{"role": "system", "content": "sys"}, {"role": "user", "content": "run ls"},
     {"role": "assistant", "content": "", "reasoning_content": "I will run ls.",
      "tool_calls": [{"id": "1", "type": "function", "function": {"name": "shell", "arguments": {"command": ["ls", "-la"], "workdir": "/tmp"}}}]},
     {"role": "tool", "tool_call_id": "1", "content": "a.txt\nb.txt"},
     {"role": "assistant", "content": "Two files.", "reasoning_content": "Done."},
     {"role": "user", "content": "thanks"}],
    [{"role": "user", "content": "do two things"},
     {"role": "assistant", "content": "Doing both.", "tool_calls": [
         {"id": "1", "type": "function", "function": {"name": "shell", "arguments": {"command": ["pwd"]}}},
         {"id": "2", "type": "function", "function": {"name": "apply_patch", "arguments": {"input": "*** Begin Patch\n*** End Patch"}}}]},
     {"role": "tool", "tool_call_id": "1", "content": "/w"}, {"role": "tool", "tool_call_id": "2", "content": "ok"}],
    [{"role": "user", "content": [{"type": "text", "text": "part one, "}, {"type": "text", "text": "part two"}]}],
]
KWARGS = [{}, {"enable_thinking": False}, {"reasoning_effort": "low"}, {"reasoning_effort": "medium"},
          {"preserve_thinking": False}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ling-template", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    cases = []
    for conv in CONVS:
        for tools in (None, TOOLS):
            for kw in KWARGS:
                cases.append({"messages": conv, "tools": tools, "kwargs": kw})
    inp = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
    out = subprocess.run([args.ling_template], input=inp, capture_output=True, text=True, check=True)
    ours = [json.loads(line) for line in out.stdout.splitlines()]
    bad = 0
    for c, got in zip(cases, ours):
        want = tok.apply_chat_template(c["messages"], tools=c["tools"], tokenize=False, add_generation_prompt=True,
                                       **c["kwargs"])
        if got != want:
            bad += 1
            if bad <= 3:
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
                print(f"MISMATCH kwargs={c['kwargs']} tools={bool(c['tools'])} at char {i}:\n"
                      f"  jinja {want[max(0, i - 40):i + 60]!r}\n  ours  {str(got)[max(0, i - 40):i + 60]!r}")
    print(f"{len(cases) - bad}/{len(cases)} renderings identical")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
