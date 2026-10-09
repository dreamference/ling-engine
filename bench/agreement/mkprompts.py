"""The agreement study's prompt set: validate_v0.py's six prompts, a file of short prompts and a file of long ones
(one JSON string per line, or {"text": ...} as bench/render/sglang_render.py writes them).

    mkprompts.py OUTDIR SHORT.jsonl LONG.jsonl    -> OUTDIR/prompts.jsonl and OUTDIR/sets.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import validate_v0
D, SHORT, LONG = sys.argv[1], sys.argv[2], sys.argv[3]
out = open(os.path.join(D, "prompts.jsonl"), "w")
sets = []
for p in validate_v0.PROMPTS:
    out.write(json.dumps(p) + "\n"); sets.append("m1six")
def text(line):
    v = json.loads(line)
    return v["text"] if isinstance(v, dict) else v
for l in open(SHORT):
    out.write(json.dumps(text(l)) + "\n"); sets.append("short24")
for l in open(LONG):
    out.write(json.dumps(text(l)) + "\n"); sets.append("long24")
json.dump(sets, open(os.path.join(D, "sets.json"), "w"))
print(len(sets))
