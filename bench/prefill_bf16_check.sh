#!/bin/bash
# B34: measures --prefill-bf16 (BF16 DeltaNet intermediates in the prefill) against the exact engine, on the
# development GB10 with nothing else on the GPU (the reference SGLang stopped, no soak running). About 30 minutes.
#
#   bench/prefill_bf16_check.sh OUT_DIR [BUILD_DIR] [PROMPT_35K] [STUDY_DIR]
#
#   BUILD_DIR   a build of this tree (default: the tree's build/); ling-force is built there if missing
#   PROMPT_35K  a prompts file (one JSON string) of about 35K tokens; default ~/m2/p32000.txt (34,778 tokens, M2 §5)
#   STUDY_DIR   the agreement study of M2 §6 (default ~/m2/agreement): cases.jsonl, prompts.jsonl, prod-force-cold-c1.jsonl
#
# Steps, and what passing means (specs/DREAMFERENCE_LING_ENGINE_MILESTONES.md §14, the exactness policy):
#   1. Timing: the 35K prefill with the switch off, on, on, off (ling-run's `prefill:` line). Expected: on faster
#      than off by 0.6-1 s (the backlog's B34 estimate); any slowdown is a finding.
#   2. Bit-exactness within the mode: `prefix check passed` and (with the drafter) `spec check passed` with the switch on.
#   3. Agreement with the exact engine on the study's 54 prompts (6 + 24 short + 24 long, 48 forced positions each):
#      mean KL(exact || mode) <= 0.0008 nats, top-5 agreement 100%, no growth in the longest prompt bucket
#      (mode_kl.py prints PASS/FAIL per line and MODE CHECK PASSED/FAILED).
#   4. The same at the 35K prompt itself (48 positions after the exact engine's greedy continuation).
#   5. Optional (B34_GEN400=1, about 25 minutes more): 400-token greedy continuations of the 24 long agent prompts in
#      both modes, and how many tokens each pair shares before diverging: a proxy for "tool calls no worse", which
#      only the replay harness (bench/replay) judges properly.
# Not covered here: KL at 64K tokens (ling_force.cpp allocates a 40,000-token context; cases reach 37K), the 24
# replayed tool-call prompts under the harness, and a SWE-bench night. Those stay to run before the switch could
# become the default.
set -eu
H=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$H/.." && pwd)
OUT=${1:?OUT_DIR}
BUILD=${2:-$SRC/build}
P35=${3:-$HOME/m2/p32000.txt}
STUDY=${4:-$HOME/m2/agreement}
M=$(ls -d "$HOME"/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/* | head -1)
D=$(ls -d "$HOME"/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/* 2>/dev/null | head -1 || true)
RUN=$BUILD/ling-run
FORCE=$BUILD/ling-force
mkdir -p "$OUT"
cd "$OUT"
echo "build $BUILD, model $M, study $STUDY, prompt $P35" | tee summary.txt
if [ ! -x "$FORCE" ]; then  # the agreement study's engine tool, built against this build's libraries (no GPU needed)
  NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
  "$NVCC" -std=c++20 -O2 -gencode arch=compute_121a,code=sm_121a -I"$SRC/src" "$H/agreement/ling_force.cpp" \
    "$BUILD/libling_core.a" "$BUILD/libling_tokenizer.a" -lcublas -lpcre2-8 -licuuc -o "$FORCE"
fi

# ---- 1. Timing: off, on, on, off (alternated so clocks and temperature affect both alike) ----
echo "== 1. 35K prefill timing" | tee -a summary.txt
i=0
for mode in off on on off; do
  i=$((i + 1))
  flag=""; [ "$mode" = on ] && flag="--prefill-bf16"
  "$RUN" --model "$M" --prompts-file "$P35" --max-tokens 2 --max-context 40000 $flag > "time-$mode-$i.out" 2> "time-$mode-$i.err"
  echo "$mode $i: $(grep '^prefill:' "time-$mode-$i.err")" | tee -a summary.txt
done

# ---- 2. Exactness within the mode ----
echo "== 2. prefix check and spec check with the switch on" | tee -a summary.txt
{ head -n 30 "$STUDY/prompts.jsonl"; sed -n '31,33p' "$STUDY/prompts.jsonl"; } > prefix-prompts.jsonl
"$RUN" --model "$M" --prompts-file prefix-prompts.jsonl --prefix-check --max-context 40000 --prefill-bf16 \
  > prefix-check-on.txt 2> prefix-check-on.err || true
grep -E 'prefix check passed|PREFIX CHECK FAILED' prefix-check-on.txt | tee -a summary.txt
if [ -n "$D" ] && [ -f "$HOME/m2/spec.jsonl" ]; then
  "$RUN" --model "$M" --draft "$D" --prompts-file "$HOME/m2/spec.jsonl" --spec-check --max-tokens 200 \
    --max-context 40000 --prefill-bf16 > spec-check-on.txt 2> spec-check-on.err || true
  grep -E 'spec check passed|SPEC CHECK FAILED' spec-check-on.txt | tee -a summary.txt
else
  echo "spec check skipped: no drafter or no ~/m2/spec.jsonl" | tee -a summary.txt
fi

# ---- 3. Agreement with the exact engine on the study's prompts ----
echo "== 3. KL against the exact engine (study prompts)" | tee -a summary.txt
"$FORCE" score "$M" 20 < "$STUDY/cases.jsonl" > eng-off.jsonl 2> eng-off.err
python3 "$H/agreement/mode_kl.py" want eng-off.jsonl "$STUDY/cases.jsonl" > cases-on.jsonl
LING_PREFILL_BF16=1 "$FORCE" score "$M" 20 < cases-on.jsonl > eng-on.jsonl 2> eng-on.err
python3 "$H/agreement/mode_kl.py" compare eng-off.jsonl eng-on.jsonl "$STUDY/cases.jsonl" \
  --prod "$STUDY/prod-force-cold-c1.jsonl" | tee kl-study.txt | tee -a summary.txt || true

# ---- 4. The same at the 35K prompt ----
echo "== 4. KL against the exact engine at 35K" | tee -a summary.txt
"$FORCE" tokenize "$M" < "$P35" | head -n 1 | python3 -c 'import json,sys; print(json.dumps({"prompt": json.loads(sys.stdin.readline())["ids"]}))' > case-35k-gen.jsonl
"$FORCE" gen "$M" 48 < case-35k-gen.jsonl > gen-35k-off.jsonl 2> gen-35k-off.err
python3 - <<'PY'
import json
p = json.loads(open("case-35k-gen.jsonl").readline())["prompt"]
g = json.loads(open("gen-35k-off.jsonl").readline())["ids"]
open("cases-35k.jsonl", "w").write(json.dumps({"prompt": p, "cont": g}) + "\n")
PY
"$FORCE" score "$M" 20 < cases-35k.jsonl > eng-35k-off.jsonl 2> eng-35k-off.err
python3 "$H/agreement/mode_kl.py" want eng-35k-off.jsonl cases-35k.jsonl > cases-35k-on.jsonl
LING_PREFILL_BF16=1 "$FORCE" score "$M" 20 < cases-35k-on.jsonl > eng-35k-on.jsonl 2> eng-35k-on.err
python3 "$H/agreement/mode_kl.py" compare eng-35k-off.jsonl eng-35k-on.jsonl cases-35k.jsonl | tee kl-35k.txt | tee -a summary.txt || true

# ---- 5. Optional: long greedy continuations of the agent prompts in both modes ----
if [ "${B34_GEN400:-0}" = 1 ]; then
  echo "== 5. 400-token continuations of the 24 long prompts, off against on" | tee -a summary.txt
  "$FORCE" gen "$M" 400 < "$STUDY/cases-long.jsonl" > gen400-off.jsonl 2> gen400-off.err
  LING_PREFILL_BF16=1 "$FORCE" gen "$M" 400 < "$STUDY/cases-long.jsonl" > gen400-on.jsonl 2> gen400-on.err
  python3 - <<'PY' | tee -a summary.txt
import json
a = [json.loads(l)["ids"] for l in open("gen400-off.jsonl")]
b = [json.loads(l)["ids"] for l in open("gen400-on.jsonl")]
same = 0
for x, y in zip(a, b):
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]: n += 1
    same += x == y
    print(f"shared prefix {n:4d} of {len(x):4d}/{len(y):4d} tokens{'  (identical)' if x == y else ''}")
print(f"{same} of {len(a)} continuations identical")
PY
fi
echo "== done: $OUT/summary.txt" | tee -a summary.txt
