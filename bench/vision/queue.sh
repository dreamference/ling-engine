#!/bin/bash
# Waits for the GPU's earlier claims before the images windows (the coordinator's order, 2026-10-26 evening: no night
# container; after the tree agent's window F; after verify-profile only if it already holds the lock), then runs each
# window in turn (each takes and releases the lock itself). Status in ~/vision/queue.status.
#   queue.sh SCRIPT [ARGS...] [-- SCRIPT [ARGS...]]...
H=~/.ling-gpu-holds.log
st() { echo "$(date -u +%FT%TZ) $*" > ~/vision/queue.status; }
while true; do
  night=$(docker ps --format '{{.Names}}' | grep -c '^ab-n17')
  fline=$(grep -n '^[^ ]* tree-rows-agent .*window F' $H | head -1 | cut -d: -f1)
  fdone=0
  if [ -n "$fline" ]; then
    tail -n +$((fline + 1)) $H | grep -qE 'tree-rows-agent lock released|UTC (agentlong|tree)' && fdone=1
  fi
  # verify-profile first only when it already holds the lock: its last line is "lock taken" without a release after it.
  vp_holds=0
  vt=$(grep -n 'verify-profile-agent lock taken' $H | tail -1 | cut -d: -f1)
  if [ -n "$vt" ] && ! tail -n +$((vt + 1)) $H | grep -qE 'verify-profile-agent .*released|UTC .*verify'; then vp_holds=1; fi
  if [ $night -eq 0 ] && [ $fdone -eq 1 ] && [ $vp_holds -eq 0 ]; then break; fi
  st "waiting: night containers $night, window F line ${fline:-none} done $fdone, verify-profile holding $vp_holds"
  sleep 60
done
while [ $# -gt 0 ]; do
  cmd=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do cmd+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift
  st "running ${cmd[*]}"
  "${cmd[@]}"
  rc=$?
  st "ran ${cmd[*]} (exit $rc)"
  [ -e ~/vision/stop ] && { st "stopped before the next window (~/vision/stop)"; exit 0; }
done
st "done all windows"
