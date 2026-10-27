#!/usr/bin/env bash
# Builds each variant of sparse_isa_probe.cu; with --run also runs the ones ptxas accepted (GPU window only).
# Usage: bench/lowbit/sparse_isa_probe.sh [--run] [OUT_DIR=build/lowbit]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=0; [ "${1:-}" = --run ] && { RUN=1; shift; }
OUT=${1:-$HERE/../../build/lowbit}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
mkdir -p "$OUT"
for v in 1 2 3 4 5 6; do
  if "$NVCC" -O3 -std=c++20 -gencode arch=compute_121a,code=sm_121a -DVARIANT=$v "$HERE/sparse_isa_probe.cu" \
       -o "$OUT/sparse_isa_probe_$v" 2> "$OUT/sparse_isa_probe_$v.err"; then
    echo "variant $v: accepted by ptxas"
    [ $RUN = 1 ] && "$OUT/sparse_isa_probe_$v"
  else
    echo "variant $v: rejected: $(grep -m1 -i error "$OUT/sparse_isa_probe_$v.err" | cut -c1-200)"
  fi
done
