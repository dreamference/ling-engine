#!/bin/bash
# One tuning variant of production SGLang, measured on the replay (M0, deliverable 4).
#
# Starts a scratch container on :30000 with production's flags plus the given serve.py edits and
# request logging, waits for it, replays the tuning set single-stream inside a short
# ~/.ling-quiet window (the machine is shared), and removes the container. :8000 must already be
# stopped (two 60 GB servers do not fit); restoring it is the caller's job.
#
# Usage: tune.sh LABEL [--repeat N] -- [serve.py args...]
#   REPLAYS="name:concurrency:set:window:extra replay args;..." runs those replays instead, one quiet
#   window each (default: one single-stream replay of the tuning set per repeat).
set -u
LABEL=$1; shift
REPEAT=1
if [ "${1:-}" = "--repeat" ]; then REPEAT=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
cd ~/m0
# SGLang sizes its KV pool from the free memory it sees around the weight load, and on GB10 the
# page cache does not count as free: drop it, or a start beside another resident engine fails.
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
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
if [ -z "${REPLAYS:-}" ]; then
  REPLAYS=""
  for r in $(seq 1 "$REPEAT"); do
    REPLAYS="$REPLAYS;r$r:1:${TUNE_SET:-set-tune.txt}:${TUNE_WINDOW:-8}:${REPLAY_EXTRA:-}"
  done
fi
IFS=';' read -ra ITEMS <<< "$REPLAYS"
for item in "${ITEMS[@]}"; do
  [ -z "$item" ] && continue
  IFS=: read -r name conc set window extra <<< "$item"
  # The machine is shared: at least QUIET_GAP seconds between quiet windows (the last end is
  # kept in ~/m0/last-quiet-end), then time for the other worker's compiles to pause.
  last=$(cat ~/m0/last-quiet-end 2>/dev/null || echo 0)
  while [ $(( $(date +%s) - last )) -lt "${QUIET_GAP:-1200}" ]; do sleep 15; done
  touch ~/.ling-quiet; sleep "${QUIET_SETTLE:-45}"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv,noheader -lms 2000 \
    > "runs/tune-$LABEL-$name.clocks.csv" 2>&1 &
  SMI=$!
  $R --sessions-file "$set" --window "$window" --concurrency "$conc" --label "$LABEL-$name" \
     --out "runs/tune-$LABEL-$name.jsonl" $extra > "runs/tune-$LABEL-$name.log" 2>&1
  kill $SMI
  rm -f ~/.ling-quiet; date +%s > ~/m0/last-quiet-end
  tail -1 "runs/tune-$LABEL-$name.jsonl"
done
docker logs "m0-$LABEL" > "runs/tune-$LABEL.server.log" 2>&1
docker rm -f "m0-$LABEL" > /dev/null
