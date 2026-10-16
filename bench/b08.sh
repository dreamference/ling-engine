#!/bin/bash
# B08's trace (BACKLOG §19.24): one decode step's bubbles and the bandwidth a subset of SMs reaches. A measurement,
# no engine code. One model on the machine: run with production stopped (the caller holds ~/gpu.lock).
#
#   bench/b08.sh BUILD OUT [PROMPT_FILE]
#
# 1. nsys 2025.3.2 (the release that records graph nodes on this driver) around `ling-run --spec --graphs --pdl` on a
#    ~24K-token prompt, 400 tokens, greedy; exported to sqlite; bubbles.py and step_breakdown.py on it.
# 2. `ling-stream-tests --bench-groups` with the step's weights as B26's coded blobs, on the whole GPU and on a green
#    context of 8, 16, 24 and 48 SMs (LING_BENCH_SMS).
set -u
B=${1:?build dir}; O=${2:?out dir}; P=${3:-$HOME/m3/p24k.txt}
H=$(cd "$(dirname "$0")" && pwd)
NSYS=/opt/nvidia/nsight-systems/2025.3.2/bin/nsys
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
D=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
mkdir -p "$O"
echo "== untraced step"
$B/ling-run --model $M --draft $D --prompts-file $P --max-tokens 400 --max-context 40000 --spec --graphs --pdl 2>&1 >/dev/null | grep -E "^spec:"
echo "== traced step"
$NSYS profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite=true -o "$O/step" \
  $B/ling-run --model $M --draft $D --prompts-file $P --max-tokens 400 --max-context 40000 --spec --graphs --pdl 2>&1 \
  | grep -E "^spec:|Generated|rror"
$NSYS export --type sqlite --force-overwrite=true -o "$O/step.sqlite" "$O/step.nsys-rep" > /dev/null
echo "== bench groups"
for sms in 0 8 16 24 48; do
  echo "-- LING_BENCH_SMS=$sms"
  LING_BENCH_CODED=1 LING_BENCH_SMS=$sms $B/ling-stream-tests --bench-groups 2>&1 | tee "$O/groups-$sms.txt" | grep -E "green|step GEMMs|rror"
done
echo "== step breakdown"
python3 "$H/profile/step_breakdown.py" "$O/step.sqlite" | tee "$O/breakdown.txt"
GB=$(grep -oE "step GEMMs \(([0-9.]+) GB" "$O/groups-0.txt" | grep -oE "[0-9.]+")
echo "== bubbles (GEMM bytes per step $GB GB from bench-groups; sustained 240 GB/s, B16's best single-kernel rate)"
python3 "$H/profile/bubbles.py" "$O/step.sqlite" "$GB" 240 | tee "$O/bubbles.txt"
echo b08-done
