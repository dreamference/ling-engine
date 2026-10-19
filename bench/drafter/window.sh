#!/bin/bash
# window.sh NAME BUDGET_MIN STEP...: one GPU window of the drafter/lowp measurements. Takes the shared GPU lock,
# stops m0-prod8000, runs the steps in order while the budget allows (a step is skipped if fewer than its
# estimate's minutes remain, or after 19:15 UTC), starts m0-prod8000 again unless another job waits for the lock
# (then leaves it down for the waiter, as arms.sh does), and logs the hold in ~/.ling-gpu-holds.log.
# Steps:  nsys:LABEL[:ENV[:GRAPHS[:NSYS]]]          ling-run --spec under nsys (one ~3K-token chat prompt), with step
#                                                   graphs unless GRAPHS=0; NSYS the trace options (+ for spaces;
#                                                   default hardware trace of graph nodes, which on this GB10 kept no
#                                                   kernel timestamps; -t+cuda-sw,nvtx with GRAPHS=0)
#         arm:DRAFTER:MODE:LABEL[:ENV]              one agentlong arm (arms.sh), DRAFTER ss, rtn or a path
# Run it with setsid nohup ... < /dev/null &.
set -u
NAME=$1; BUDGET=$2; shift 2
LE=$HOME/le-drafter; OUT=$LE/runs; mkdir -p $OUT $LE/prof
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
SS=$HOME/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
HOLDS=$HOME/.ling-gpu-holds.log
exec 8> /home/user/.ling-gpu.lock
echo "$(date -u +%FT%TZ) drafter-agent waiting for the lock ($NAME)" >> $HOLDS
flock 8
T0=$(date -u +%s)
echo "$(date -u +%FT%TZ) drafter-agent lock taken: $NAME" >> $HOLDS; echo "=== lock taken $(date -u +%T) UTC"
if [ -n "$(docker ps -q -f name=^m0-prod8000$)" ]; then
  docker stop m0-prod8000 > /dev/null; echo "m0 stopped $(date -u +%T) UTC"
  echo "$(date -u +%FT%TZ) drafter-agent m0-prod8000 stopped" >> $HOLDS
fi
PROMPT="Here is part of a C++ inference engine. Explain what draft_launch does step by step, then rewrite it to \
take the LM head through a separate function, keeping the behaviour identical.

$(sed -n 280,345p $LE/src/core/speculate.cpp)
$(sed -n 540,640p $LE/src/core/engine.cpp)"
left() { echo $(( BUDGET - ($(date -u +%s) - T0) / 60 )); }
late() { [ $(date -u +%H%M) -ge 1915 ]; }
for st in "$@"; do
  IFS=: read -r kind a b c d <<< "$st"
  need=$([ "$kind" = nsys ] && echo 4 || echo 10)
  if [ $(left) -lt $need ] || late; then echo "skipped $st ($(left) min left) $(date -u +%T)"; continue; fi
  if [ "$kind" = nsys ]; then
    label=$a; envs=${b:-}; g=$([ "${c:-1}" = 0 ] || echo --graphs); nopt=${d:--t+cuda,nvtx+--cuda-graph-trace=node}
    env ${envs//,/ } nsys profile ${nopt//+/ } -f true -o $LE/prof/$label \
      $LE/build/ling-run --model "$M" --draft "$SS" --block 12 --spec $g --pdl --max-context 16384 \
      --chat "$PROMPT" --max-tokens 384 > $LE/prof/$label.out 2> $LE/prof/$label.err 8>&- < /dev/null
    echo "nsys $label rc $? $(date -u +%T): $(grep -h '^spec:' $LE/prof/$label.err)"
  else
    LOCK_HELD=1 bash $LE/bench/agentlong/arms.sh $OUT "engine:$LE:$a:$b:$c::${d:-}" 8>&- < /dev/null | grep -v "^=== lock\|^m0 "
  fi
done
WAITING=$(exec 8>&-; fuser /home/user/.ling-gpu.lock 2>/dev/null | tr -s ' ' '\n' | grep -v "^$$\$" | grep -c .)
if [ "$WAITING" = 0 ] || [ $(date -u +%H%M) -ge 1930 ]; then
  docker start m0-prod8000 > /dev/null; echo "m0 started $(date -u +%T) UTC"
  echo "$(date -u +%FT%TZ) drafter-agent m0-prod8000 started" >> $HOLDS
  for i in $(seq 1 180); do curl -sf -m 3 http://127.0.0.1:8000/v1/models > /dev/null && { echo "m0 healthy $(date -u +%T) UTC"; break; }; sleep 5; done
else
  echo "m0 left down: $WAITING job(s) waiting for the lock"
  echo "$(date -u +%FT%TZ) drafter-agent m0-prod8000 left down for $WAITING waiting job(s)" >> $HOLDS
fi
echo "$(date -u +%FT%TZ) drafter-agent lock released: $NAME, held $(( ($(date -u +%s) - T0) / 60 )) min" >> $HOLDS
echo "=== lock released $(date -u +%T) UTC, held $(( ($(date -u +%s) - T0) / 60 )) min"
