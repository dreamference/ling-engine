#!/bin/bash
# batch_profile.sh OUTDIR [BUILD]: where a step's time goes at one and at two sequences, and what a session costs in
# memory (BATCHING §3). Run inside a GPU window (streams_window.sh run:...), never on its own.
#   1. The step in place, timed (no profiler): ling-run --spec at ~31K (the tree on, then the chain alone), and
#      --batch-bench on a pair of ~24K and ~31K prompts (every step batched while both run), 1,000 tokens each; the
#      same at ~124K (four copies of each prompt), where the KV read dominates.
#   2. The same three at ~31K under Nsight Systems 2025.3.2 (CUDA graph nodes traced; 2026.3.2 records no graph
#      kernels on this driver), reduced by bench/profile/step_breakdown.py: GPU time per step by kernel family and by
#      draft / verify / commit.
#   3. Memory: ling-serve started at --sessions 1, 2 and 4 (--max-context 65536) and at --sessions 2 with
#      --max-context 131072 and 262144; the host's MemAvailable before and once it serves (this machine's GPU memory
#      is the host's), the prefix KV store off (--prefix-kv-mb 0) so only the sessions differ.
set -u
O=$1; BUILD=${2:-$HOME/le-batch/build}
mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
D=$HOME/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
NSYS=/opt/nvidia/nsight-systems/2025.3.2/bin/nsys
R="$BUILD/ling-run --model $M --draft $D"
P24=$HOME/m3/p24k.txt; P31=$HOME/m3/p31k.txt
cat "$P24" "$P31" > "$O/pair-agent.jsonl"
python3 - "$P24" "$P31" "$O" <<'EOF'
import json, sys
p24, p31, o = sys.argv[1:4]
a = [json.loads(open(p).read()) for p in (p24, p31)]
open(f"{o}/pair-long.jsonl", "w").write(json.dumps(a[0] * 5) + "\n" + json.dumps(a[1] * 4) + "\n")
open(f"{o}/p-long.txt", "w").write(json.dumps(a[1] * 4) + "\n")
EOF
mem() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
t() { echo "--- $* ($(date -u +%T))"; }

t "1. in place"
for rep in 1 2; do
  $R --prompts-file "$P31" --max-tokens 400 --max-context 40000 --spec --graphs --pdl 2>&1 >/dev/null | grep -E "^spec:" | sed "s/^/one tree rep $rep: /"
  LING_TREE=0 $R --prompts-file "$P31" --max-tokens 400 --max-context 40000 --spec --graphs --pdl 2>&1 >/dev/null | grep -E "^spec:" | sed "s/^/one chain rep $rep: /"
  $R --prompts-file "$O/pair-agent.jsonl" --batch-bench --sessions 2 --max-tokens 1000 --max-context 40000 --graphs --pdl 2>&1 | grep -E "^batch bench" | sed "s/^/two rep $rep: /"
done
$R --prompts-file "$O/p-long.txt" --max-tokens 400 --max-context 131072 --spec --graphs --pdl 2>&1 >/dev/null | grep -E "^spec:" | sed "s/^/one tree long: /"
LING_TREE=0 $R --prompts-file "$O/p-long.txt" --max-tokens 400 --max-context 131072 --spec --graphs --pdl 2>&1 >/dev/null | grep -E "^spec:" | sed "s/^/one chain long: /"
$R --prompts-file "$O/pair-long.jsonl" --batch-bench --sessions 2 --max-tokens 1000 --max-context 131072 --graphs --pdl 2>&1 | grep -E "^batch bench" | sed "s/^/two long: /"

t "2. nsys"
prof() {  # label env cmd...
  local L=$1 E=$2; shift 2
  env $E $NSYS profile -t cuda,nvtx --cuda-graph-trace=node -o "$O/$L" --force-overwrite=true "$@" > "$O/$L.out" 2>&1
  $NSYS export --type sqlite -o "$O/$L.sqlite" --force-overwrite=true "$O/$L.nsys-rep" > /dev/null 2>&1
  python3 "$HERE/../profile/step_breakdown.py" "$O/$L.sqlite" > "$O/$L.breakdown.txt" 2>&1
  echo "$L: $(grep -cE . "$O/$L.breakdown.txt") lines"
}
prof one-tree "LING_X=0" $R --prompts-file "$P31" --max-tokens 400 --max-context 40000 --spec --graphs --pdl
prof one-chain "LING_TREE=0" $R --prompts-file "$P31" --max-tokens 400 --max-context 40000 --spec --graphs --pdl
prof two "LING_X=0" $R --prompts-file "$O/pair-agent.jsonl" --batch-bench --sessions 2 --max-tokens 1000 --max-context 40000 --graphs --pdl
rm -f "$O"/*.nsys-rep

t "3. memory"
srv() {  # label flags...
  local L=$1; shift
  sync; sleep 5; local m0; m0=$(mem)
  setsid "$BUILD"/ling-serve --model "$M" --draft "$D" --port 8301 --served-model-name RadixArk/Qwen3.8-27B-NVFP4 \
    --draft-block 12 --graphs --pdl "$@" > "$O/mem-$L.log" 2>&1 < /dev/null &
  local p=$!
  "$HERE/mem_guard.sh" "pid:$p" "$O/mem_guard.log" < /dev/null &
  for _ in $(seq 1 120); do curl -sf -m 2 localhost:8301/health > /dev/null && break; sleep 3; done
  sleep 10; local m1; m1=$(mem)
  echo "mem $L: before $m0 MB, serving $m1 MB, engine $((m0 - m1)) MB ($*)"
  kill $p; wait $p 2>/dev/null; sleep 5
}
srv s1-64k --sessions 1 --batch 1 --max-context 65536 --prefix-kv-mb 0
srv s2-64k --sessions 2 --max-context 65536 --prefix-kv-mb 0
srv s4-64k --sessions 4 --batch 4 --max-context 65536 --prefix-kv-mb 0
srv s2-128k --sessions 2 --max-context 131072 --prefix-kv-mb 0
srv s2-256k --sessions 2 --max-context 262144 --prefix-kv-mb 0
srv s2-64k-kv --sessions 2 --max-context 65536
t "done"
