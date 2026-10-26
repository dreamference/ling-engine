#!/bin/bash
# Runs the images windows in turn once no night container is running; each window blocks on ~/.ling-gpu.lock itself
# (whoever gets the lock runs: no waiting on other agents' log lines, the coordinator's rule of 2026-10-26 evening).
# Status in ~/vision/queue.status; ~/vision/stop stops it before the next window.
#   queue.sh SCRIPT [ARGS...] [-- SCRIPT [ARGS...]]...
H=~/.ling-gpu-holds.log
st() { echo "$(date -u +%FT%TZ) $*" > ~/vision/queue.status; }
while [ "$(docker ps --format '{{.Names}}' | grep -c '^ab-n17')" -gt 0 ]; do
  st "waiting: a night container is running"
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
