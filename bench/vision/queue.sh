#!/bin/bash
# Waits for the GPU's earlier claims before an images window (the brief of 2026-10-26: after night 17 and the tree
# check's window F), then runs it. Status in ~/vision/queue.status.
#   queue.sh WINDOW_SCRIPT ARGS...
H=~/.ling-gpu-holds.log
st() { echo "$(date -u +%FT%TZ) $*" > ~/vision/queue.status; }
while true; do
  now=$(date -u +%s); not_before=$(date -u -d "2026-10-27 07:30" +%s)
  night=$(docker ps --format '{{.Names}}' | grep -c '^ab-n17')
  serve8300=$(ss -ltn 2>/dev/null | grep -c ':8300 ')
  fline=$(grep -n 'window F' $H | head -1 | cut -d: -f1)
  fdone=0
  if [ -n "$fline" ]; then
    tail -n +$((fline + 1)) $H | grep -qE 'tree-rows-agent lock released|tree-rows-agent.*released|UTC (agentlong|tree)' && fdone=1
  fi
  if [ $now -ge $not_before ] && [ $night -eq 0 ] && [ $serve8300 -eq 0 ] && [ $fdone -eq 1 ]; then break; fi
  st "waiting: night containers $night, 8300 listening $serve8300, window F line ${fline:-none} done $fdone"
  sleep 60
done
st "running $1"
"$@"
st "done $1 (exit $?)"
