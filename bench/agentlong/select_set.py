"""The fixed long-context replay set (agentlong): sixteen `ling` agent-coding turns at 25K-64K tokens of context.

Every engine and SGLang number on long agent turns is measured on this set, so that two runs never
compare different prompts. The turns come from M0's recorded SWE-bench sessions on second-puffin
(`~/m0/traces/swe-bench`, real `ling` sessions: shell commands, file edits through the shell, code-index
calls). Prompt contents never leave that machine: the repository holds only `manifest.json`, which names
each turn (its rollout, relative to the traces root, and its request index), its recorded sizes and
the SHA-256 of the request body; `run.py` refuses a body whose hash differs.

Selection (deterministic, seed 20261019): one turn per SWE-bench instance, sixteen turns in five bins
of recorded input tokens, four each at 25-33K, 33-41K and 41-49K, two at 49-52K and two at 52-64K;
recorded output 400-2,000 tokens (250-2,000 in the last bin, where long outputs are rare); in each
bin all turns but one are edits (the output writes a file: a heredoc, a patch, `sed -i`, a Python
write) and the last is not, where the bin has them. The body is replay.py's (bench/replay): the
rollout's own items as `input`, its `base_instructions`, the captured tools (with the code-index tools
unless the session ran without them), reasoning effort none, `max_output_tokens` the recorded output,
and the capture's custom tool (`exec`) dropped, which is the prompt SGLang renders, so both servers
prefill the same tokens. `run.py` adds `temperature: 0` (greedy) or a per-turn seed (sampled at the
model's defaults, as `ling` sends it).

Usage (on second-puffin):
  select_set.py [--traces ~/m0/traces/swe-bench] [--tools ~/m0/tools-capture.json]
                [--tools-index ~/m0/tools-index.json] [--bodies ~/bench-sets/agentlong/bodies.jsonl]
                [--manifest bench/agentlong/manifest.json] [--check]
  --check rebuilds the bodies of an existing manifest and verifies their hashes (no selection).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay"))
from replay import clean_item, custom_tools, parse_rollout  # noqa: E402

HOME = os.path.expanduser("~")
# (lowest recorded input, highest, turns, shortest recorded output): contexts past 52K are rare in the
# recordings (the sessions compact near there), so the last bin takes shorter outputs.
BINS = [(25000, 33000, 4, 400), (33000, 41000, 4, 400), (41000, 49000, 4, 400), (49000, 52000, 2, 400),
        (52000, 64000, 2, 250)]
MAX_OUT = 2000
SEED = 20261019
MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
EDIT = re.compile(r"apply_patch|\*\*\* Begin Patch|cat > |cat >> |<<'?EOF|\.write\(|\.write_text\(|sed -i")


def body_of(path, instr, rq, tools_plain, tools_index):
    """replay.py's body for one request (custom tools dropped), without temperature or seed."""
    return {
        "model": MODEL, "instructions": instr,
        "input": [clean_item(x) for x in rq["input"]],
        "tools": tools_plain if "index-off" in path else tools_index, "tool_choice": "auto",
        "parallel_tool_calls": True, "reasoning": {"effort": "none", "summary": "auto"},
        "store": False, "stream": True,
        "max_output_tokens": max(int(rq["usage"].get("output_tokens") or 0), 1),
    }


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def kind_of(rq):
    names, text = [], ""
    for it in rq["out_items"]:
        t = it.get("type")
        names.append(it.get("name", "") if t in ("function_call", "custom_tool_call") else "message")
        a = it.get("arguments") or it.get("input") or ""
        if isinstance(a, str):
            text += a
    return ",".join(names), bool(EDIT.search(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=f"{HOME}/m0/traces/swe-bench")
    ap.add_argument("--tools", default=f"{HOME}/m0/tools-capture.json")
    ap.add_argument("--tools-index", default=f"{HOME}/m0/tools-index.json")
    ap.add_argument("--bodies", default=f"{HOME}/bench-sets/agentlong/bodies.jsonl")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tools_plain = custom_tools(json.load(open(a.tools)), "drop")
    tools_index = custom_tools(json.load(open(a.tools_index)), "drop")

    if a.check:
        man = json.load(open(a.manifest))
        bad = 0
        for t in man["turns"]:
            p = os.path.join(a.traces, t["rollout"])
            instr, reqs = parse_rollout(p)
            h = hashlib.sha256(canonical(body_of(p, instr, reqs[t["req"]], tools_plain, tools_index)).encode()).hexdigest()
            ok = h == t["body_sha256"]
            bad += not ok
            print(t["id"], "ok" if ok else "HASH DIFFERS")
        return 1 if bad else 0

    cands = []
    for p in sorted(glob.glob(os.path.join(a.traces, "**", "rollout-*.jsonl"), recursive=True)):
        rel = os.path.relpath(p, a.traces)
        if rel.split(os.sep)[0].startswith("smoke"):
            continue
        m = re.search(r"/scratch/([^/]+)/", p)
        if not m:
            continue
        try:
            instr, reqs = parse_rollout(p)
        except (ValueError, KeyError):
            continue
        for i, rq in enumerate(reqs):
            inp = int(rq["usage"].get("input_tokens") or 0)
            out = int(rq["usage"].get("output_tokens") or 0)
            if any(lo <= inp < hi and mo <= out <= MAX_OUT for lo, hi, _, mo in BINS):
                kinds, edit = kind_of(rq)
                cands.append({"rollout": rel, "req": i, "instance": m.group(1), "rec_in": inp, "rec_out": out,
                              "kind": kinds, "edit": edit})
    rng = random.Random(SEED)
    used, chosen = set(), []
    for lo, hi, n, mo in reversed(BINS):  # the rare long bins choose first
        b = [c for c in cands if lo <= c["rec_in"] < hi and c["rec_out"] >= mo]
        rng.shuffle(b)
        picked = []
        for want_edit in [True] * max(n - 1, 1) + [False] * (n > 1):
            pool = [c for c in b if c["instance"] not in used and c not in picked and (c["edit"] or not want_edit)]
            if want_edit is False:
                pool = [c for c in pool if not c["edit"]] or pool
            if not pool:
                pool = [c for c in b if c["instance"] not in used and c not in picked]
            if not pool:
                continue
            c = pool[0]
            picked.append(c)
            used.add(c["instance"])
        chosen += picked
    chosen.sort(key=lambda c: c["rec_in"])
    os.makedirs(os.path.dirname(a.bodies), exist_ok=True)
    turns = []
    with open(a.bodies, "w") as f:
        for k, c in enumerate(chosen):
            p = os.path.join(a.traces, c["rollout"])
            instr, reqs = parse_rollout(p)
            s = canonical(body_of(p, instr, reqs[c["req"]], tools_plain, tools_index))
            tid = f"t{k:02d}"
            f.write(json.dumps({"id": tid, "body": json.loads(s)}, ensure_ascii=False) + "\n")
            turns.append({"id": tid, **c, "body_sha256": hashlib.sha256(s.encode()).hexdigest()})
            print(tid, c["rec_in"], c["rec_out"], c["edit"], c["kind"], c["instance"])
    json.dump({"set": "agentlong", "seed": SEED, "bins": BINS, "traces_root": "~/m0/traces/swe-bench",
               "custom_tools": "drop", "turns": turns}, open(a.manifest, "w"), indent=1)
    print(f"{len(turns)} turns; recorded output {sum(t['rec_out'] for t in turns)} tokens; bodies {a.bodies}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
