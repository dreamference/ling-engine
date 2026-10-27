#!/bin/bash
# streams_window.sh: one GPU window of n_streams.py runs (BATCHING §2), ling-serve and SGLang alike, under node-b's
# GPU lock with the rules every window there keeps: m0-prod8000 stopped first and started again at the end (left
# down when another job waits for the lock), the hold logged in ~/.ling-gpu-holds.log, and no item started once the
# window has held the lock 38 minutes (windows <= 45 min).
#
# Usage: streams_window.sh OUTDIR "DESCRIPTION" ITEM...
#   engine:LABEL:STREAMS:MODE[:FLAGS[:ENV]]   a fresh ling-serve from $BUILD (default ~/le-batch/build) at Mightling's
#                                             flags plus FLAGS and ENV (each with commas for spaces), N streams
#   sglang-up:LABEL                           SGLang from SG_CMD (default ~/ab-n17/sglang.cmd: v0.5.21 with the
#                                             phase-merge patch, production's flags, the ScaleSearch drafter), port 30000
#   sglang:LABEL:STREAMS:MODE                 N streams against it, its radix cache flushed first
#   sglang-down                               its log saved, the container removed
#   run:SCRIPT[ ARGS]:LABEL                   any script inside the hold (words split on spaces)
# MODE greedy or seedN, warm-greedy or warm-seedN for n_streams.py --warm (warm follow-ups in rounds). SGLang runs
# with its request log (--log-requests, level 1, JSON), which streams_server.py reads for the decode intervals.
# Every server runs beside mem_guard.sh (Mightling's memory-pressure kill rule and a MemAvailable floor), and an
# engine item asking for more than 4 sessions is refused. AGENT names the holder in the log (default batching-agent).
set -u
O=$1; DESC=$2; shift 2; mkdir -p "$O"
AGENT=${AGENT:-batching-agent}
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HOME/le-batch/build}
SG_CMD=${SG_CMD:-$HOME/ab-n17/sglang.cmd}
H=$HOME/.ling-gpu-holds.log
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
SS=$HOME/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
echo "$(date -u +%FT%TZ) $AGENT waiting for the lock ($DESC)" >> "$H"
exec 8> "$HOME/.ling-gpu.lock"; flock 8
T_LOCK=$(date -u +%s); echo "=== lock taken $(date -u +%T) UTC"
echo "$(date -u +%FT%TZ) $AGENT lock taken: $DESC" >> "$H"
late() { [ $(( $(date -u +%s) - T_LOCK )) -ge 2280 ]; }
if [ -n "$(docker ps -q -f name=^m0-prod8000$)" ]; then
  docker stop m0-prod8000 > /dev/null; echo "m0 stopped $(date -u +%T)"
  echo "$(date -u +%FT%TZ) $AGENT m0-prod8000 stopped" >> "$H"
fi
SG_UP=0
for item in "$@"; do
  IFS=: read -r kind a b c d e <<< "$item"
  if late && [ "$kind" != sglang-down ]; then echo "too late for $item: skipped"; continue; fi
  case $kind in
    run)
      bash $a > "$O/$b.log" 2>&1 8>&- < /dev/null; echo "$b ($(date -u +%T)): exit $?" ;;
    engine)
      label=$a; n=$b; mode=$c; extra=${d:-}; envs=${e:-}
      # At most 4 sessions until ling-serve has admission control (node-b froze at 8 on 2026-10-27).
      if [[ ",$extra," =~ ,--sessions,([0-9]+), ]] && [ "${BASH_REMATCH[1]}" -gt 4 ]; then
        echo "$label: refused, ${BASH_REMATCH[1]} sessions (at most 4)"; continue
      fi
      env ${envs//,/ } setsid "$BUILD"/ling-serve --model "$M" --draft "$SS" --port 8300 \
        --served-model-name RadixArk/Qwen3.8-27B-NVFP4 --max-context 65536 --draft-block 12 --graphs --pdl \
        ${extra//,/ } > "$O/$label.serve.log" 2>&1 8>&- < /dev/null &
      SRV=$!
      "$HERE/mem_guard.sh" "pid:$SRV" "$O/mem_guard.log" 8>&- < /dev/null &
      for _ in $(seq 1 120); do curl -sf -m 2 localhost:8300/health > /dev/null && break; sleep 3; done
      echo "engine $label up $(date -u +%T), MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MB"
      warm=; [ "${mode#warm-}" != "$mode" ] && { warm=--warm; mode=${mode#warm-}; }
      python3 "$HERE/n_streams.py" --url http://127.0.0.1:8300 --streams "$n" --mode "$mode" --label "$label" $warm --cache-key \
        --out "$O/$label.jsonl" > "$O/$label.log" 2>&1 8>&-
      curl -s -m 10 localhost:8300/spec_stats > "$O/$label.spec.json" 2>/dev/null
      curl -s -m 10 localhost:8300/metrics > "$O/$label.metrics.txt" 2>/dev/null
      kill $SRV; wait $SRV 2>/dev/null
      echo "$label ($(date -u +%T)): $(tail -1 "$O/$label.jsonl" | cut -c1-400)" ;;
    sglang-up)
      label=$a
      docker rm -f batch-sg > /dev/null 2>&1
      grep '^docker run' "$SG_CMD" | sed -e 's/--name [^ ]*/--name batch-sg/' \
        -e 's/$/ --log-requests --log-requests-level 1 --log-requests-format json/' > "$O/$label.cmd"
      echo "before sglang: MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MB"
      bash "$O/$label.cmd" > /dev/null 8>&-
      "$HERE/mem_guard.sh" container:batch-sg "$O/mem_guard.log" 8>&- < /dev/null &
      for i in $(seq 1 240); do
        curl -sf -m 3 localhost:30000/v1/models > /dev/null && break
        [ -z "$(docker ps -q -f name=^batch-sg$)" ] && { echo "sglang exited: $(docker logs batch-sg 2>&1 | grep -E 'Error|error' | tail -2)"; break; }
        sleep 5
      done
      SG_UP=1; echo "sglang $label up after $((i*5)) s $(date -u +%T), MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MB" ;;
    sglang)
      label=$a; n=$b; mode=$c
      curl -s -m 30 -X POST localhost:30000/flush_cache > /dev/null; sleep 2
      warm=; [ "${mode#warm-}" != "$mode" ] && { warm=--warm; mode=${mode#warm-}; }
      python3 "$HERE/n_streams.py" --url http://127.0.0.1:30000 --streams "$n" --mode "$mode" --label "$label" $warm \
        --out "$O/$label.jsonl" > "$O/$label.log" 2>&1 8>&-
      echo "$label ($(date -u +%T)): $(tail -1 "$O/$label.jsonl" | cut -c1-400)" ;;
    sglang-down)
      [ $SG_UP = 1 ] && { docker logs batch-sg > "$O/sglang.serverlog" 2>&1; docker rm -f batch-sg > /dev/null; SG_UP=0; } ;;
  esac
done
[ $SG_UP = 1 ] && { docker logs batch-sg > "$O/sglang.serverlog" 2>&1; docker rm -f batch-sg > /dev/null; }
WAITING=$(exec 8>&-; fuser "$HOME/.ling-gpu.lock" 2>/dev/null | tr -s ' ' '\n' | grep -v "^$$\$" | grep -c .)
if [ "$WAITING" = 0 ]; then
  docker start m0-prod8000 > /dev/null
  for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:8000/v1/models > /dev/null && { echo "m0 healthy $(date -u +%T) UTC after $((i*5)) s"; break; }; sleep 5; done
  echo "$(date -u +%FT%TZ) $AGENT m0-prod8000 started again" >> "$H"
else
  echo "$(date -u +%FT%TZ) $AGENT $WAITING job(s) waiting for the lock: m0-prod8000 left stopped for them" >> "$H"
fi
echo "=== lock released $(date -u +%T) UTC, held $(( ($(date -u +%s) - T_LOCK) / 60 )) min"
echo "$(date -u +%FT%TZ) $AGENT lock released, held $(( ($(date -u +%s) - T_LOCK) / 60 )) min ($DESC)" >> "$H"
