#!/bin/bash
# Moonshot P1's gate replay (the owner's decision D1 of 2026-10-13: run both modes, keep the faster by tokens per
# millisecond). Written 2026-10-13, not yet run: it needs the GPU (P1's slot comes after the step-savings block).
#
#   bench/copy_gate.sh BUILD TOOLS TOOLS_INDEX SESSIONS_FILE [OUT]
#
# One ling-serve per arm on port 18081 (production's settings from bench/serve_bg.sh: block 12, --graphs --pdl), the
# same greedy replay of M0's main set against each (bench/replay/replay.py, as M1 ran it), /spec_stats read around it:
#   copy0   --copy 0                                the reference: DFlash2 alone
#   copy1   --copy 1 --copy-views 3 --copy-edit     replace, every source
#   copy2   --copy 2 --copy-views 3 --copy-edit     extend, every source
# ARMS="copy0 copy1 copy2 copy1h copy2h" adds the history-only arms (copy1h, copy2h) to split the sources' worth.
# Then bench/copy_gate_report.py: decode tokens per ms per arm and per session against copy0, the outputs compared
# request by request (SHA-256 of the streamed output: greedy output must be byte-identical in every arm), the copy
# counters by source, and the verdict: the faster of copy1 and copy2 whose outputs are all identical, and whether it
# clears P1's gate (whole sessions >= +7%).
#
# One model server on the machine: stop m0-prod8000 first and restart it after, with a line in the chat each time.
set -u
B=${1:?usage: copy_gate.sh BUILD TOOLS TOOLS_INDEX SESSIONS_FILE [OUT]}
TOOLS=${2:?tools.json}
TOOLS_INDEX=${3:?tools-index.json}
SESSIONS=${4:?sessions file}
OUT=${5:-$HOME/p1-gate}
PORT=18081
HERE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"
flags_of() {
  case $1 in
  copy0) echo "--copy 0" ;;
  copy1) echo "--copy 1 --copy-views 3 --copy-edit" ;;
  copy2) echo "--copy 2 --copy-views 3 --copy-edit" ;;
  copy1h) echo "--copy 1" ;;
  copy2h) echo "--copy 2" ;;
  *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}
for arm in ${ARMS:-copy0 copy1 copy2}; do
  echo "== $arm: $(flags_of $arm)"
  # shellcheck disable=SC2046
  bash "$HERE/serve_bg.sh" $PORT "$B" $(flags_of $arm) || exit 1
  curl -s 127.0.0.1:$PORT/spec_stats > "$OUT/$arm-before.json"
  python3 "$HERE/replay/replay.py" --url http://127.0.0.1:$PORT --tools "$TOOLS" --tools-index "$TOOLS_INDEX" \
    --sessions-file "$SESSIONS" --greedy --label "$arm" --out "$OUT/$arm.jsonl" > "$OUT/$arm.log" 2>&1
  curl -s 127.0.0.1:$PORT/spec_stats > "$OUT/$arm-after.json"
  python3 "$HERE/replay/spec_report.py" "$OUT/$arm-before.json" "$OUT/$arm-after.json" > "$OUT/$arm-spec.json"
  bash "$HERE/serve_bg.sh" stop $PORT
  sleep 5
done
python3 "$HERE/copy_gate_report.py" "$OUT" ${ARMS:-copy0 copy1 copy2} | tee "$OUT/report.txt"
