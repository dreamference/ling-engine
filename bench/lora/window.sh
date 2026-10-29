#!/bin/bash
# One GPU window for the adapters work (LORA §6), as the images work's (bench/vision/window.sh): takes
# ~/.ling-gpu.lock, stops m0-prod8000, runs STEPS (a bash file of steps, each a line `step NAME COMMAND...` run with
# its output in V/NAME.log), starts m0 again unless another job waits for the lock, logs the hold. At most 45 minutes: a step that would
# start after 40 minutes is skipped, and one still running then is stopped.
#   window.sh LABEL DESCRIPTION STEPS V        (run it with setsid nohup ... < /dev/null &)
set -u
LABEL=$1; DESC=$2; STEPS=$3; V=$4; mkdir -p $V
exec 8> $HOME/.ling-gpu.lock
echo "$(date -u +%FT%TZ) lora-agent waiting for the lock (adapters $LABEL: $DESC; <= 45 min, m0 stopped)" >> ~/.ling-gpu-holds.log
flock 8; T0=$(date -u +%s)
# Nights (20:30 to 08:00 UTC) are not ours: a window that would end in one does not start.
# LORA_END (HHMM UTC, default 2015) is when m0 must be up again: the steps stop 5 minutes before it, and a window
# with less than 15 minutes of steps left does not start.
END=${LORA_END:-2015}
NOW=$((10#$(date -u +%H%M)))
STOP=$(( $(date -u -d "today ${END:0:2}:${END:2:2}" +%s) - 5 * 60 ))
if [ $(( STOP - $(date -u +%s) )) -lt $(( 15 * 60 )) ] || [ $NOW -lt 800 ]; then
  echo "$(date -u +%FT%TZ) lora-agent lock released unused ($LABEL: too close to the night)" >> ~/.ling-gpu-holds.log
  echo "=== not started: $(date -u +%T) is too close to the night"; exit 3
fi
echo "$(date -u +%FT%TZ) lora-agent lock taken ($LABEL)" >> ~/.ling-gpu-holds.log
M0=$(docker inspect -f '{{.State.Running}}' m0-prod8000 2>/dev/null || echo false)
[ "$M0" = true ] && docker stop m0-prod8000 > /dev/null
step() {
  local name=$1; shift
  local end=$(( T0 + 40 * 60 )); [ $STOP -lt $end ] && end=$STOP
  local left=$(( end - $(date -u +%s) ))
  if [ $left -le 0 ]; then echo "$(date -u +%T) $name skipped (window over)"; return; fi
  echo "$(date -u +%T) $name"
  # A subshell, so a step may be a shell function; past the window's end its children and it are stopped.
  ( "$@" ) > $V/$name.log 2>&1 8>&- &
  local p=$! w=0
  while kill -0 $p 2>/dev/null; do
    sleep 5; w=$((w + 5))
    if [ $w -ge $left ]; then pkill -TERM -P $p; sleep 2; kill -TERM $p 2>/dev/null; fi
  done
  wait $p
  echo "$(date -u +%T) $name exit $?"
}
source $STEPS
# m0 comes back only if nobody else waits for the lock (the next holder would stop it again): the other processes
# that have the lock file open, this window's own shell aside.
WAITERS=$(for p in $(fuser $HOME/.ling-gpu.lock 2>/dev/null); do [ "$p" != $$ ] && echo $p; done | wc -l)
RESTART=$M0
[ "$M0" = true ] && [ "$WAITERS" -gt 0 ] && RESTART=false
if [ "$RESTART" = true ]; then
  docker start m0-prod8000 > /dev/null
  for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:8000/v1/models > /dev/null && { echo "m0 healthy $(date -u +%T)"; break; }; sleep 5; done
fi
echo "=== released $(date -u +%T)"
echo "$(date -u -d @$T0 +%F\ %T)-$(date -u +%T) UTC lora-agent adapters $LABEL (m0 was running: $M0; restarted: $RESTART$([ "$M0" = true ] && [ "$RESTART" = false ] && echo ", left stopped for $WAITERS waiters"))" >> ~/.ling-gpu-holds.log
