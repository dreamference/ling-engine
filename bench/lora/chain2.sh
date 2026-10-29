#!/bin/bash
# The adapters windows after w3: w4 (SGLang's reference with an adapter) then w5 (the engine on the same cases, the
# adapter path's speed, the per-adapter cost), once window PID (w6) has ended. Each takes the lock on its own.
#   chain2.sh OUT PID   (setsid nohup ... < /dev/null &)
H=$(cd "$(dirname "$0")" && pwd); O=$1
while kill -0 $2 2>/dev/null; do sleep 30; done
$H/window.sh w4 "agreement: SGLang with an adapter" $H/w4.steps $O/w4 || exit
[ -s $HOME/lora-agree/cases.jsonl ] && $H/window.sh w5 "agreement: the engine, adapter cost" $H/w5.steps $O/w5
