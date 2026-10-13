#!/bin/bash
# Moonshot P1: the copy proposer's exactness checks on the GPU (ling-run). Not run when P1's CPU half landed
# (2026-10-13); run when the GPU is free, one model on the machine (m0-prod8000 stopped first, restarted after).
#
#   bench/copy_exact.sh [BUILD] [which...]     which: regress spec graph prefix chunk interleave batch (default all)
#
# Each check runs with --copy 1 (replace) and --copy 2 (extend) on copy-heavy prompts (bench/copy_prompts.py)
# and on the M3 set; "regress" runs today's checks without --copy (the default path must be unchanged). A check
# passes when its output has no FAIL/mismatch line and ling-run exits 0. The copy counters printed after each run
# ("copy: N steps held copied tokens ...") show that copies really happened.
set -u
B=${1:-$HOME/ling-engine-p1/build}
shift || true
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
D=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
R=$B/ling-run
OUT=~/p1-exact
mkdir -p $OUT && cd $OUT
python3 "$(dirname "$0")/copy_prompts.py" > copy.jsonl
cp ~/m3/spec.jsonl ~/m3/prefix.jsonl .
run() {
  n=$1
  shift
  echo "== $n: $*"
  flock ~/gpu.lock $R --model $M --draft $D "$@" > $n.txt 2>&1
  echo "exit $?"
  grep -E "FAIL|mismatch|differ|copy:|passed|tokens/step" $n.txt | head -20
}
for w in "${@:-all}"; do
  case $w in
  regress | all)
    run regress-spec --prompts-file spec.jsonl --max-tokens 200 --spec-check --graphs --pdl
    run regress-batch --prompts-file spec.jsonl --max-tokens 200 --batch-check --sessions 2 --graphs --pdl
    ;;&
  spec | all)
    for c in 1 2; do
      run spec-copy$c --prompts-file copy.jsonl --max-tokens 3000 --spec-check --copy $c
      run spec-copy$c-graphs --prompts-file copy.jsonl --max-tokens 3000 --spec-check --copy $c --graphs --pdl
      run spec-copy$c-m3 --prompts-file spec.jsonl --max-tokens 200 --spec-check --copy $c --graphs --pdl
    done
    ;;&
  graph | all)
    for c in 1 2; do run graph-copy$c --prompts-file copy.jsonl --max-tokens 3000 --graph-check --seed 7 --copy $c --pdl; done
    ;;&
  prefix | all)
    run prefix-copy2 --prompts-file prefix.jsonl --prefix-check --sessions 2 --copy 2 --pdl --max-context 40000
    ;;&
  chunk | all)
    run chunk-copy2 --prompts-file copy.jsonl --chunk-check 256 --copy 2 --max-tokens 512
    ;;&
  interleave | all)
    run interleave-copy2 --prompts-file copy.jsonl --interleave-check --sessions 2 --copy 2 --max-tokens 512
    ;;&
  batch | all)
    # A batch never copies, so --batch-check (batch against runs alone) is run without --copy: the batch path
    # must be what it was. (With --copy the runs alone would copy and the steps would differ by design.)
    run batch --prompts-file copy.jsonl --batch-check --sessions 2 --max-tokens 512 --graphs --pdl
    ;;
  esac
done
echo copy-exact-done
