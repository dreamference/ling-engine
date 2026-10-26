#!/bin/bash
# One GPU window for the images work (VISION §20.8): takes ~/.ling-gpu.lock, stops m0-prod8000, runs STEPS (a bash file
# of steps, each a line `step NAME COMMAND...` run with its output in V/NAME.log), restores m0, logs the hold.
# At most 45 minutes: a step that would start after 40 minutes is skipped.
#   window.sh LABEL DESCRIPTION STEPS V
set -u
LABEL=$1; DESC=$2; STEPS=$3; V=$4; mkdir -p $V
exec 8> $HOME/.ling-gpu.lock
echo "$(date -u +%FT%TZ) vision-agent waiting for the lock (images $LABEL: $DESC; <= 45 min, m0 stopped)" >> ~/.ling-gpu-holds.log
flock 8; T0=$(date -u +%s)
echo "$(date -u +%FT%TZ) vision-agent lock taken ($LABEL)" >> ~/.ling-gpu-holds.log
M0=$(docker inspect -f '{{.State.Running}}' m0-prod8000 2>/dev/null || echo false)
[ "$M0" = true ] && docker stop m0-prod8000 > /dev/null
step() {
  local name=$1; shift
  local left=$(( T0 + 40 * 60 - $(date -u +%s) ))
  if [ $left -le 0 ]; then echo "$(date -u +%T) $name skipped (window over)"; return; fi
  echo "$(date -u +%T) $name"
  timeout $left "$@" > $V/$name.log 2>&1 8>&-
  echo "$(date -u +%T) $name exit $?"
}
source $STEPS
[ "$M0" = true ] && docker start m0-prod8000 > /dev/null
echo "=== released $(date -u +%T)"
echo "$(date -u -d @$T0 +%F\ %T)-$(date -u +%T) UTC vision-agent images $LABEL (m0 was running: $M0; restarted: $M0)" >> ~/.ling-gpu-holds.log
