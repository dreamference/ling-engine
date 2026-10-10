#!/usr/bin/env bash
# Builds the three device probes of spec section 16.1 (items 1-3) with the toolkit the tree uses.
# The probes live in bench/membw/ (M0 wrote them); this folder holds the runner and the clocks loop.
# Usage: bench/micro/build.sh [OUT_DIR=build/micro]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${1:-$ROOT/build/micro}
NVCC=${NVCC:-$(sed -n "s/^CMAKE_CUDA_COMPILER[^=]*=//p" "$ROOT/build/CMakeCache.txt" 2>/dev/null | head -1)}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
mkdir -p "$OUT"
# -gencode with the arch-specific target: `-arch=sm_121a` alone emits compute_121 PTX, which ptxas
# rejects for the block-scaled and kind::f8f6f4 MMAs (reports/M0.md section 2).
GEN="-gencode arch=compute_121a,code=sm_121a"
"$NVCC" -O3 -std=c++20 $GEN "$ROOT/bench/membw/latency.cu"   -o "$OUT/latency"
"$NVCC" -O3 -std=c++20 $GEN "$ROOT/bench/membw/membw.cu"     -o "$OUT/membw"
"$NVCC" -O3 -std=c++20 $GEN "$ROOT/bench/membw/isa_probe.cu" -o "$OUT/isa_probe" -lcuda
echo "built: $OUT/{latency,membw,isa_probe} with $("$NVCC" --version | tail -1)"
