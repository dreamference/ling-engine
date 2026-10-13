#!/bin/bash
# Moonshot P1: the copy proposer's and the 64-row verify's exactness checks on the GPU. Written 2026-10-13 with no GPU
# run; run when P1's GPU slot comes, one model on the machine (m0-prod8000 stopped first and restarted after, with a
# line in the chat each time).
#
#   bench/copy_exact.sh [BUILD] [which...]
#   which: kernels rows regress spec graph prefix chunk interleave batch (default: all, in that order)
#
# kernels    ling-stream-tests (stream_gemm, bf16_rows and attention to 64 rows, bit for bit across M; also under
#            LING_KSPLIT=2, where the split-K choice must not follow M), ling-kernel-tests, ling-prefill-tests
# rows       ling-run --rows-check 64: one verify of n <= 64 rows against n one-row steps, every row's logits bit for bit
# regress    today's --spec-check and --batch-check without --copy (the default path unchanged)
# spec       --spec-check --copy 1|2 (greedy tokens and recurrent state equal plain decoding), history only and with
#            every source (--copy-views 3 --copy-edit), eager and with --graphs --pdl, on copy-heavy prompts
#            (bench/copy_prompts.py) and the M3 set
# graph      --graph-check --copy 1|2 with every source (graphs against eager, greedy and sampled; the verify graphs
#            keyed by row count up to 64)
# prefix, chunk, interleave
#            the index, the views and the edit anchors give the same proposals after resumes, chunked prefills and
#            interleaving (same steps, same accepted drafts)
# batch      --batch-check without --copy (a batch never copies; with --copy the runs alone would copy and their
#            steps would differ by design)
#
# A check passes when ling-run exits 0 and prints no FAIL/DIFFER line. The spec lines print the copy steps per source.
set -u
B=${1:-$HOME/ling-engine-p1/build}
shift || true
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
D=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
R=$B/ling-run
OUT=~/p1-exact
ALL="--copy-views 3 --copy-edit"
mkdir -p $OUT && cd $OUT
python3 "$(dirname "$(readlink -f "$0")")/copy_prompts.py" > copy.jsonl
cp ~/m3/spec.jsonl ~/m3/prefix.jsonl .
run() {
  n=$1
  shift
  echo "== $n: $*"
  flock ~/gpu.lock $R --model $M --draft $D "$@" > $n.txt 2>&1
  echo "exit $?"
  grep -E "FAIL|DIFFER|passed|copy steps|rows:" $n.txt | head -20
}
bin() {
  n=$1
  shift
  echo "== $n"
  flock ~/gpu.lock env "$@" > $n.txt 2>&1
  echo "exit $?"
  grep -E "FAIL|NO |passed" $n.txt | head -20
}
for w in "${@:-all}"; do
  case $w in
  kernels | all)
    bin stream $B/ling-stream-tests
    bin stream-ksplit LING_KSPLIT=2 $B/ling-stream-tests
    bin kernel $B/ling-kernel-tests
    bin prefill $B/ling-prefill-tests
    ;;&
  rows | all)
    run rows --prompts-file spec.jsonl --rows-check 64
    run rows-pdl --prompts-file spec.jsonl --rows-check 64 --pdl --attn-bulk
    ;;&
  regress | all)
    run regress-spec --prompts-file spec.jsonl --max-tokens 200 --spec-check --graphs --pdl
    run regress-batch --prompts-file spec.jsonl --max-tokens 200 --batch-check --sessions 2 --graphs --pdl
    ;;&
  spec | all)
    for c in 1 2; do
      run spec-copy$c --prompts-file copy.jsonl --max-tokens 3000 --spec-check --copy $c
      run spec-copy$c-all --prompts-file copy.jsonl --max-tokens 3000 --spec-check --copy $c $ALL
      run spec-copy$c-all-graphs --prompts-file copy.jsonl --max-tokens 3000 --spec-check --copy $c $ALL --graphs --pdl
      run spec-copy$c-m3 --prompts-file spec.jsonl --max-tokens 200 --spec-check --copy $c $ALL --graphs --pdl
    done
    ;;&
  graph | all)
    for c in 1 2; do
      run graph-copy$c --prompts-file copy.jsonl --max-tokens 3000 --graph-check --seed 7 --copy $c $ALL --pdl
    done
    ;;&
  prefix | all)
    run prefix-copy2 --prompts-file prefix.jsonl --prefix-check --sessions 2 --copy 2 $ALL --pdl --max-context 40000
    ;;&
  chunk | all)
    run chunk-copy2 --prompts-file copy.jsonl --chunk-check 256 --copy 2 $ALL --max-tokens 512
    ;;&
  interleave | all)
    run interleave-copy2 --prompts-file copy.jsonl --interleave-check --sessions 2 --copy 2 $ALL --max-tokens 512
    ;;&
  batch | all)
    run batch --prompts-file copy.jsonl --batch-check --sessions 2 --max-tokens 512 --graphs --pdl
    ;;
  esac
done
echo copy-exact-done
