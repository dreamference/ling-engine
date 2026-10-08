#!/bin/bash
# One tuning variant of production SGLang, measured on the replay (M0, deliverable 4).
#
# Starts a scratch container on :30000 with production's flags plus the given serve.py edits and
# request logging, waits for it, replays the tuning set single-stream inside a short
# ~/.ling-quiet window (the machine is shared), and removes the container. :8000 must already be
# stopped (two 60 GB servers do not fit); restoring it is the caller's job.
#
# Usage: tune.sh LABEL [--repeat N] -- [serve.py args...]
set -u
LABEL=$1; shift
REPEAT=1
if [ "${1:-}" = "--repeat" ]; then REPEAT=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
cd ~/m0
python3 serve.py "m0-$LABEL" --port 30000 --log-requests "$@" > "runs/tune-$LABEL.serve.txt"
ok=0
for i in $(seq 1 120); do
  curl -sf localhost:30000/v1/models > /dev/null && { ok=1; break; }
  docker ps -a --format '{{.Names}} {{.Status}}' | grep -q "m0-$LABEL Exited" && break
  sleep 10
done
if [ $ok = 0 ]; then
  echo "FAILED to start $LABEL"; docker logs --tail 80 "m0-$LABEL" > "runs/tune-$LABEL.fail.log" 2>&1
  docker rm -f "m0-$LABEL" > /dev/null; exit 1
fi
R="python3 replay.py --url http://127.0.0.1:30000 --tools tools-capture.json --tools-index tools-index.json"
# Warm-up outside the quiet window: first compiles of shapes, graph paths.
$R --sessions-file set-prof-warm.txt --window 2 --label "$LABEL-warm" --out "runs/tune-$LABEL-warm.jsonl" > /dev/null 2>&1
for r in $(seq 1 "$REPEAT"); do
  # Wait for the other worker's compiles to see the flag (it pauses at its next check).
  touch ~/.ling-quiet; sleep "${QUIET_SETTLE:-45}"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv,noheader -lms 2000 \
    > "runs/tune-$LABEL-r$r.clocks.csv" 2>&1 &
  SMI=$!
  $R --sessions-file "${TUNE_SET:-set-tune.txt}" --window "${TUNE_WINDOW:-8}" --label "$LABEL-r$r" \
     --out "runs/tune-$LABEL-r$r.jsonl" ${REPLAY_EXTRA:-} > "runs/tune-$LABEL-r$r.log" 2>&1
  kill $SMI
  rm -f ~/.ling-quiet
  tail -1 "runs/tune-$LABEL-r$r.jsonl"
  [ "$r" -lt "$REPEAT" ] && sleep "${QUIET_GAP:-1200}"
done
docker logs "m0-$LABEL" > "runs/tune-$LABEL.server.log" 2>&1
docker rm -f "m0-$LABEL" > /dev/null
