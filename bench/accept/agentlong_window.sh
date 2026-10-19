#!/bin/bash
# win.sh ITEM...: one GPU window of acceptance runs on bench/agentlong at one stream (12 drafts, RTN drafter unless rtn|ss given).
#   sgf:LABEL                 SGLang greedy over the agentlong prompts via /generate (forced.py sglang): ids + counters
#   sgr:LABEL:MODE            SGLang, run.py over /v1/responses, MODE greedy|seed:N
#   enf:LABEL:BUILD:ACT:FORCE ling-serve via /v1/completions, forced to FORCE's ids (- = free greedy); ACT = LING_DRAFT_ACT
#   enr:LABEL:BUILD:ACT:MODE  ling-serve, run.py over /v1/responses
# Takes the GPU lock (logging to ~/acc/holds.log), stops m0 if up; at the end starts m0 unless another process waits.
set -u
O=~/cast/runs/agentlong; mkdir -p $O; LOG=~/acc/holds.log; P=~/acc/al/prompts.jsonl
RUN=~/le-main/bench/agentlong/run.py; FORCED=~/ling-engine-accf/bench/accept/forced.py
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
RTN=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
SGRTN=/root/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/$(basename $RTN)
exec 8> /home/kladko/.ling-gpu.lock; flock 8
T0=$(date -u +%s); echo "acc lock taken $(date -u +%T) [$*]" | tee -a $LOG
[ -n "$(docker ps -q -f name=^m0-prod8000$)" ] && { docker stop m0-prod8000 > /dev/null; echo "acc m0 stopped $(date -u +%T)" | tee -a $LOG; }
SG=0
sg_up() {
  [ $SG = 1 ] && return
  python3 ~/ling-engine-acc/bench/sglang/serve.py acc-sg --port 30000 --set speculative-num-draft-tokens 12 \
    --set speculative-draft-model-path $SGRTN --log-requests > $O/sg.cmd 8>&-
  for i in $(seq 1 240); do curl -sf -m 3 localhost:30000/v1/models > /dev/null && break; sleep 5; done
  echo "sglang up after $((i*5)) s $(date -u +%T)"; SG=1
}
sg_down() { [ $SG = 1 ] && { docker logs acc-sg > $O/sg-$(date -u +%H%M).serverlog 2>&1; docker rm -f acc-sg > /dev/null; SG=0; }; true; }
for item in "$@"; do
  IFS=: read -r kind label a b c d e <<< "$item"; ENVS=${e:-}; ENVS=${ENVS//+/ }
  case $kind in
  sgf) sg_up; python3 $FORCED sglang --prompts $P --url http://127.0.0.1:30000 --out $O/$label.jsonl > $O/$label.log 2>&1 8>&- ;;
  sgr) sg_up; python3 $RUN --url http://127.0.0.1:30000 --mode "$a${b:+:$b}" --label $label --out $O/$label.jsonl > $O/$label.log 2>&1 8>&- ;;
  enf|enr)
    sg_down
    env LING_DRAFT_ACT=$b $ENVS setsid $a/build/ling-serve --model $M --draft $RTN --port 8300 --served-model-name RadixArk/Qwen3.8-27B-NVFP4 \
      --max-context 65536 --draft-block 12 --graphs --pdl > $O/$label.serve.log 2>&1 8>&- < /dev/null &
    SRV=$!
    for i in $(seq 1 120); do curl -sf -m 2 localhost:8300/health > /dev/null && break; sleep 3; done
    if [ $kind = enf ]; then
      FA=$([ "$c" = - ] && echo "" || echo "--force $O/$c.jsonl")
      python3 $FORCED engine --prompts $P --url http://127.0.0.1:8300 --out $O/$label.jsonl $FA > $O/$label.log 2>&1 8>&-
    else
      python3 $RUN --url http://127.0.0.1:8300 --mode "$c${d:+:$d}" --label $label --out $O/$label.jsonl > $O/$label.log 2>&1 8>&-
    fi
    curl -s localhost:8300/metrics > $O/$label.metrics
    kill $SRV; wait $SRV 2>/dev/null ;;
  esac
  echo "acc $label done $(date -u +%T)" | tee -a $LOG
done
sg_down
waiters=$(lsof -t /home/kladko/.ling-gpu.lock 2>/dev/null | grep -v -x $$ | wc -l)
if [ "${KEEP_M0_DOWN:-0}" != 1 ] && [ "$waiters" = 0 ]; then
  docker start m0-prod8000 > /dev/null; echo "acc m0 started $(date -u +%T)" | tee -a $LOG
  for i in $(seq 1 180); do curl -sf -m 3 http://127.0.0.1:8000/v1/models > /dev/null && { echo "acc m0 answering $(date -u +%T)" | tee -a $LOG; break; }; sleep 5; done
else echo "acc m0 left down ($waiters waiting) $(date -u +%T)" | tee -a $LOG; fi
echo "acc lock released $(date -u +%T), held $(( ($(date -u +%s) - T0) / 60 )) min" | tee -a $LOG
