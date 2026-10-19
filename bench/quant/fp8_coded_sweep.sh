#!/bin/bash
# BACKLOG §19 item 30, stage 2: build the coded-GEMM microbenchmarks and run the sweep of 2026-10-19 (under the GPU
# lock, m0-prod8000 stopped). q = fp8_coded_gemm.cu (one value a lookup, 10-bit codes), p = fp8_coded_gemm_pair.cu
# (two values a lookup, 6-bit codes, cp.async segments with one tile); the digit = minimum blocks an SM.
# Usage: bench/quant/fp8_coded_sweep.sh [SNAPSHOT_DIR]   (from the repository root, after the CMake build)
set -e
SNAP=${1:-$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/ | head -1)}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
for v in q p; do
  src=bench/quant/fp8_coded_gemm.cu; [ $v = p ] && src=bench/quant/fp8_coded_gemm_pair.cu
  for mb in 1 2; do
    $NVCC -O3 -std=c++20 -gencode arch=compute_121a,code=sm_121a -Isrc -DFP8C_MINB=$mb -o build/fp8_coded_gemm_$v$mb \
      $src build/libling_core.a -lcuda
  done
done
# min-blocks variant, tiles, chunks in flight, two tiles side by side
for cfg in "q1 2 2 0" "q1 2 2 1" "q1 1 2 0" "q2 1 2 0" "p2 1 1 0" "p2 1 2 0" "p1 2 1 1"; do
  set -- $cfg
  echo "== $1 tiles $2 in flight $3 side by side $4"
  FP8C_TILES=$2 FP8C_PF=$3 FP8C_ILP=$4 ./build/fp8_coded_gemm_$1 "$SNAP" --reps 3 | grep -E "^[0-9]+ FP8|^M=|exact on"
done
