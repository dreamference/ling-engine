#!/usr/bin/env bash
# Builds paged_attn_bench against the attention kernels of one or more revisions at several KV page sizes. The page
# size is a compile-time constant (kKvPageTokens in src/core/kernels.cuh), so each (revision, size) is its own binary:
# the revision's src/ is exported with git archive, only kKvPageTokens is changed in the copy, and only kernels.cu is
# compiled beside the bench (the engine's tree and build are not touched; the pool and admission code, which assume
# 64-position pages, are not part of the binary).
# Usage: bench/micro/paged_attn_sweep.sh OUT_DIR [REVS=HEAD] [PAGE_TOKENS="64 128 256 512"]
#   OUT_DIR/paged_attn_bench-<rev>-<pt>: run it as paged_attn_bench (see its header), e.g.
#   paged_attn_bench-HEAD-128 262144 16 4 64 flat,page,head seq,rr,shuf 0 2
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${1:?usage: paged_attn_sweep.sh OUT_DIR [REVS] [PAGE_TOKENS]}
REVS=${2:-HEAD}
PTS=${3:-64 128 256 512}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
GEN="-gencode arch=compute_121a,code=sm_121a"
mkdir -p "$OUT"
for r in $REVS; do
  for pt in $PTS; do
    tag=$(echo "$r" | tr '/^~' '_..')-$pt
    src=$OUT/src-$tag
    rm -rf "$src" && mkdir -p "$src"
    git -C "$ROOT" archive "$r" src | tar -x -C "$src"
    sed -i -E "s/^constexpr int kKvPageTokens = [0-9]+;/constexpr int kKvPageTokens = $pt;/" "$src/src/core/kernels.cuh"
    grep -q "^constexpr int kKvPageTokens = $pt;" "$src/src/core/kernels.cuh" || { echo "kKvPageTokens not found in $r" >&2; exit 1; }
    "$NVCC" -O3 -lineinfo -std=c++20 $GEN --expt-relaxed-constexpr -I"$src/src" "$HERE/paged_attn_bench.cu" \
      "$src/src/core/kernels.cu" -lcuda -o "$OUT/paged_attn_bench-$tag"
    echo "built: $OUT/paged_attn_bench-$tag (kernels of $(git -C "$ROOT" rev-parse --short "$r"), $pt-position pages)"
  done
done
