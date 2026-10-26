#!/bin/bash
# big_window.sh OUTDIR: the BIG_DRAFTER spec's GPU measurements, run inside a hold of the GPU lock (the caller takes
# it, stops m0-prod8000 and restarts it). Two parts:
#  1. The drafter's cost per step at each size: ling-serve (this branch's build, production's flags: 12 drafts, step
#     graphs, PDL, the tree of 16 rows, the sparse head) with each drafter from big_dummy.py, NVFP4 and BF16, one
#     warm-up request and one measured greedy request of 320 tokens over a ~10K-token prompt; the draft, verify and
#     commit seconds per step come from /spec_stats (the difference across the measured request).
#  2. One training step of a drafter of each depth with dummy data (train_step.py in the SGLang image's torch).
# Arms are listed in ARMS below as NAME:DRAFTER_DIR:EXTRA (EXTRA ling-serve flags, commas for spaces).
set -u
O=$1; mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
B=${BUILD:-$HOME/le-bd}
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
RTN=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
INC=$(ls -d ~/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/*)
D=${DUMMIES:-$HOME/bd/dummies}
# In order of priority (the window may end before the last ones).
ARMS=${ARMS:-"rtn:$RTN: f4-L5:$D/f4-L5-I17408: f4-L10:$D/f4-L10-I17408: f4-L20:$D/f4-L20-I17408:
 bf-L5:$INC:--draft-bf16 bf-L10:$D/bf-L10-I17408:--draft-bf16 f4-W2:$D/f4-L5-I38912: f4-W4:$D/f4-L5-I79872:
 bf-W2:$D/bf-L5-I38912:--draft-bf16 f4-L8:$D/f4-L8-I17408: f4-W1.5:$D/f4-L5-I27648: f4-L10W2:$D/f4-L10-I38912:
 bf-L20:$D/bf-L20-I17408:--draft-bf16 bf-L8:$D/bf-L8-I17408:--draft-bf16 bf-W4:$D/bf-L5-I79872:--draft-bf16"}
# END_EPOCH (from the caller): no arm starts with less than 3 minutes left, the training step needs 6.
END_EPOCH=${END_EPOCH:-$(( $(date -u +%s) + 2400 ))}
left() { echo $(( END_EPOCH - $(date -u +%s) )); }
PROMPT=$O/prompt.json
python3 - "$HERE/../../src/core/speculate.cpp" "$PROMPT" <<'EOF'
import json, sys
src = open(sys.argv[1]).read()[:36000]
msg = "Explain this C++ file in detail, one paragraph per function, in the order they appear.\n\n```cpp\n" + src + "\n```"
json.dump({"model": "m", "messages": [{"role": "user", "content": msg}], "max_tokens": 320, "temperature": 0},
          open(sys.argv[2], "w"))
EOF
WARM='{"model":"m","messages":[{"role":"user","content":"Write a haiku about memory bandwidth."}],"max_tokens":64,"temperature":0}'
stats() { curl -s -m 10 localhost:8300/spec_stats; }
for arm in $ARMS; do
  IFS=: read -r name dir extra <<< "$arm"
  [ -d "$dir" ] || { echo "$name: no drafter at $dir, skipped"; continue; }
  [ "$(left)" -lt 180 ] && { echo "$name: not started, $(left) s left"; continue; }
  T0=$(date -u +%s)
  setsid "$B"/build/ling-serve --model "$M" --draft "$dir" --port 8300 --served-model-name m --max-context 65536 \
    --draft-block 12 --graphs --pdl ${extra//,/ } > "$O/$name.serve.log" 2>&1 < /dev/null &
  SRV=$!
  up=no
  for i in $(seq 1 120); do
    curl -sf -m 2 localhost:8300/health > /dev/null && { up=yes; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 2
  done
  if [ $up = yes ]; then
    curl -s -m 120 localhost:8300/v1/chat/completions -H 'content-type: application/json' -d "$WARM" > /dev/null
    stats > "$O/$name.s0.json"
    curl -s -m 300 localhost:8300/v1/chat/completions -H 'content-type: application/json' -d @"$PROMPT" > "$O/$name.out.json"
    stats > "$O/$name.s1.json"
  fi
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  python3 - "$O" "$name" "$T0" <<'EOF'
import json, re, sys, time
o, name, t0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
log = open(f"{o}/{name}.serve.log").read()
gb = re.search(r"drafter: ([0-9.]+) GB", log)
rec = {"arm": name, "drafter_gb": float(gb.group(1)) if gb else None, "wall_s": int(time.time()) - t0}
try:
    a, b = json.load(open(f"{o}/{name}.s0.json")), json.load(open(f"{o}/{name}.s1.json"))
    n = b["steps"] - a["steps"]
    for k in ("draft", "verify", "commit"):
        rec[k + "_ms"] = round(1000 * (b[k + "_seconds"] - a[k + "_seconds"]) / n, 3)
    rec["steps"] = n
    rec["tokens_per_step"] = round((b["accepted"] - a["accepted"] + n) / n, 3)
    rec["sparse_steps"] = b.get("sparse_steps", 0) - a.get("sparse_steps", 0)
    rec["tree_steps"] = b.get("tree_steps", 0) - a.get("tree_steps", 0)
except Exception as e:
    rec["error"] = repr(e)[:200] + " | " + log[-300:].replace("\n", " / ")
print(json.dumps(rec))
open(f"{o}/results.jsonl", "a").write(json.dumps(rec) + "\n")
EOF
done
# Part 2: the training step (torch from the SGLang image; the harness's directory mounted read-only).
if [ "${TRAIN:-1}" = 1 ] && [ "$(left)" -ge 360 ]; then
  echo "train_step $(date -u +%T)"
  timeout $(( $(left) - 30 )) docker run --rm --gpus all --name bd-train --ipc=host -v "$HERE":/w:ro lmsysorg/sglang:v0.5.21 \
    python3 /w/train_step.py --layers 5,10 > "$O/train.jsonl" 2> "$O/train.err"
  echo "train_step rc $? $(date -u +%T)"; cat "$O/train.jsonl"; docker rm -f bd-train > /dev/null 2>&1
fi
