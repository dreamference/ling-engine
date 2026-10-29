#!/bin/bash
# One GPU window for the adapters work (LORA §6), as the images work's (bench/vision/window.sh): takes
# ~/.ling-gpu.lock, stops m0-prod8000, runs STEPS (a bash file of steps, each a line `step NAME COMMAND...` run with
# its output in V/NAME.log), starts m0 again and waits for it, logs the hold. At most 45 minutes: a step that would
# start after 40 minutes is skipped, and one still running then is stopped.
#   window.sh LABEL DESCRIPTION STEPS V        (run it with setsid nohup ... < /dev/null &)
set -u
LABEL=$1; DESC=$2; STEPS=$3; V=$4; mkdir -p $V
exec 8> $HOME/.ling-gpu.lock
echo "$(date -u +%FT%TZ) lora-agent waiting for the lock (adapters $LABEL: $DESC; <= 45 min, m0 stopped)" >> ~/.ling-gpu-holds.log
flock 8; T0=$(date -u +%s)
# Nights (20:30 to 08:00 UTC) are not ours: a window that would end in one does not start.
NOW=$((10#$(date -u +%H%M)))
if [ $NOW -ge 1945 ] || [ $NOW -lt 800 ]; then
  echo "$(date -u +%FT%TZ) lora-agent lock released unused ($LABEL: too close to the night)" >> ~/.ling-gpu-holds.log
  echo "=== not started: $(date -u +%T) is too close to the night"; exit 3
fi
echo "$(date -u +%FT%TZ) lora-agent lock taken ($LABEL)" >> ~/.ling-gpu-holds.log
M0=$(docker inspect -f '{{.State.Running}}' m0-prod8000 2>/dev/null || echo false)
[ "$M0" = true ] && docker stop m0-prod8000 > /dev/null
step() {
  local name=$1; shift
  local left=$(( T0 + 40 * 60 - $(date -u +%s) ))
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
if [ "$M0" = true ]; then
  docker start m0-prod8000 > /dev/null
  for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:8000/v1/models > /dev/null && { echo "m0 healthy $(date -u +%T)"; break; }; sleep 5; done
fi
echo "=== released $(date -u +%T)"
echo "$(date -u -d @$T0 +%F\ %T)-$(date -u +%T) UTC lora-agent adapters $LABEL (m0 was running: $M0; restarted: $M0)" >> ~/.ling-gpu-holds.log
