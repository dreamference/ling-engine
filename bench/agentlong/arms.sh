#!/bin/bash
# arms.sh: the agentlong set (run.py) at one stream, ling-serve and SGLang (production's flags at 12 drafts) alike.
# Usage: arms.sh OUTDIR ITEM...
#   engine:BUILD:DRAFTER:MODE:LABEL[:EXTRA[:ENV]]  BUILD a ling-engine checkout with build/ling-serve; EXTRA ling-serve
#                                             flags and ENV variable assignments, each with commas for spaces
#   sglang:DRAFTER:MODE:LABEL                 a scratch SGLang from bench/sglang/serve.py
#   run:SCRIPT:LABEL                          any script, inside the same hold
# DRAFTER rtn (the RTNcal file production loads) or ss (the ScaleSearch re-quantization); MODE greedy or seedN.
# Holds the GPU lock for the whole list (logging when it was taken and released); m0-prod8000 is stopped first and
# started again at the end, unless KEEP_M0_DOWN=1 or another job is waiting for the lock. Lock hold times are appended
# to ~/.ling-gpu-holds.log. ONLY=t03,t09 replays only those turns (per engine item: ONLY=t03+t09 in its ENV). Run it with setsid nohup ... < /dev/null &.
set -u
O=$1; shift; mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
RTN=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
SS=$HOME/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
SGRTN=/root/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/$(basename "$RTN")
SGSS=/root/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
exec 8> "$HOME"/.ling-gpu.lock; flock 8
T_LOCK=$(date -u +%s); echo "=== lock taken $(date -u +%T) UTC"
if [ -n "$(docker ps -q -f name=^m0-prod8000$)" ]; then docker stop m0-prod8000 > /dev/null; echo "m0 stopped $(date -u +%T) UTC"; fi
for item in "$@"; do
  IFS=: read -r kind a b c d e f <<< "$item"
  if [ "$kind" = run ]; then  # run:SCRIPT:LABEL, a script run inside the hold (output to LABEL.log)
    bash "$a" > "$O/$b.log" 2>&1 8>&- < /dev/null; echo "$b ($(date -u +%T)): exit $?"; continue
  fi
  if [ "$kind" = engine ]; then
    B=$a; D=$([ "$b" = ss ] && echo "$SS" || echo "$RTN"); mode=$c; label=$d; extra=${e:-}; envs=${f:-}
    # ONLY=t00+t08 among an item's ENV: that item replays only those turns (overrides the global ONLY).
    only=${ONLY:-}; [[ ",$envs," =~ ,ONLY=([^,]*), ]] && only=${BASH_REMATCH[1]//+/,} && envs=$(sed -E 's/(^|,)ONLY=[^,]*//' <<< "$envs")
    env ${envs//,/ } setsid "$B"/build/ling-serve --model "$M" --draft "$D" --port 8300 --served-model-name RadixArk/Qwen3.8-27B-NVFP4 \
      --max-context 65536 --draft-block 12 --graphs --pdl ${extra//,/ } > "$O/$label.serve.log" 2>&1 8>&- < /dev/null &
    SRV=$!
    for i in $(seq 1 120); do curl -sf -m 2 localhost:8300/health > /dev/null && break; sleep 3; done
    python3 "$HERE/run.py" --url http://127.0.0.1:8300 --mode "$mode" --label "$label" --out "$O/$label.jsonl" \
      ${only:+--only $only} > "$O/$label.log" 2>&1 8>&-
    curl -s -m 10 localhost:8300/spec_stats > "$O/$label.spec.json" 2>/dev/null
    kill $SRV; wait $SRV 2>/dev/null
  else
    D=$([ "$a" = ss ] && echo "$SGSS" || echo "$SGRTN"); mode=$b; label=$c
    python3 "$HERE/../sglang/serve.py" agentlong-sg --port 30000 --set speculative-num-draft-tokens 12 \
      --set speculative-draft-model-path "$D" --log-requests > "$O/$label.cmd" 8>&-
    for i in $(seq 1 240); do curl -sf -m 3 localhost:30000/v1/models > /dev/null && break; sleep 5; done
    echo "sglang up after $((i*5)) s $(date -u +%T)"
    python3 "$HERE/run.py" --url http://127.0.0.1:30000 --mode "$mode" --label "$label" --out "$O/$label.jsonl" \
      ${ONLY:+--only $ONLY} > "$O/$label.log" 2>&1 8>&-
    docker logs agentlong-sg > "$O/$label.serverlog" 2>&1; docker rm -f agentlong-sg > /dev/null
  fi
  echo "$label ($(date -u +%T)): $(tail -1 "$O/$label.jsonl")"
done
# Waiters: every other process with the lock file open (a flock waiting, or a script that opened it to wait).
WAITING=$(exec 8>&-; fuser "$HOME"/.ling-gpu.lock 2>/dev/null | tr -s ' ' '\n' | grep -v "^$$\$" | grep -c .)
if [ "${KEEP_M0_DOWN:-0}" != 1 ] && [ "$WAITING" = 0 ]; then
  docker start m0-prod8000 > /dev/null
  for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:8000/v1/models > /dev/null && { echo "m0 healthy $(date -u +%T) UTC after $((i*5)) s"; break; }; sleep 5; done
fi
[ "$WAITING" != 0 ] && echo "m0 left down: $WAITING job(s) waiting for the lock"
echo "=== lock released $(date -u +%T) UTC, held $(( ($(date -u +%s) - T_LOCK) / 60 )) min"
echo "$(date -u -d @$T_LOCK +%F\ %T)-$(date -u +%T) UTC agentlong/arms.sh $O $*" >> ~/.ling-gpu-holds.log
