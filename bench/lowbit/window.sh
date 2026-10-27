#!/usr/bin/env bash
# One GPU window for the low-bit study's microbenchmarks (spec LOWBIT section 6), under the hosts' GPU
# lock convention: flock ~/.ling-gpu.lock, at most 45 minutes, one line per event in ~/.ling-gpu-holds.log.
# Stops a production model server if one runs (the gate container or ling-serve) and restarts it after.
# Usage: bench/lowbit/window.sh OUT_DIR   (expects build/lowbit_gemm and build/sparse_isa_probe_* beside it)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?out dir}
WHO=lowbit-study-agent
LOG=~/.ling-gpu-holds.log
PY=${PY:-$HOME/.local/share/dreamference/venv/bin/python}
mkdir -p "$OUT"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(stamp) $WHO waiting for the lock (low-bit study: sparse MMA probe, 2:4 / 3-bit / FP8 streaming GEMMs on dummy data, cuSPARSELt; <= 45 min)" >> "$LOG"
exec 9> ~/.ling-gpu.lock
flock 9
T0=$(date +%s)
echo "$(stamp) $WHO lock taken" >> "$LOG"
PROD=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(dreamference-vllm|dreamference-sglang|dreamference-gate|m0-prod)' || true)
if [ -n "$PROD" ]; then
  echo "production containers running: $PROD; stopping" | tee -a "$OUT/window.txt"
  for c in $PROD; do docker stop "$c" >/dev/null; done
fi
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm --format=csv | tee -a "$OUT/window.txt"
{
  timeout 120 "$HERE/sparse_isa_probe.sh" --run "$HERE/build" > "$OUT/sparse_isa_probe.txt" 2>&1
  timeout 900 "$HERE/build/lowbit_gemm" 20 > "$OUT/lowbit_gemm.csv" 2> "$OUT/lowbit_gemm.err"
  timeout 900 "$HERE/build/lowbit_gemm" 20 > "$OUT/lowbit_gemm_2.csv" 2>> "$OUT/lowbit_gemm.err"
  timeout 600 "$PY" "$HERE/sparse_lib.py" 20 > "$OUT/sparse_lib.csv" 2> "$OUT/sparse_lib.err"
} || true
if [ -n "$PROD" ]; then
  for c in $PROD; do docker start "$c" >/dev/null; done
fi
HELD=$(( ($(date +%s) - T0 + 59) / 60 ))
echo "$(stamp) $WHO lock released, held $HELD min (production server was running: ${PROD:+yes}${PROD:-no}; restarted: ${PROD:+yes}${PROD:-n/a})" >> "$LOG"
