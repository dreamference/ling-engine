#!/bin/bash
# The adapters windows of 10-30, each taking the lock on its own: w7 (the pdl_begin() fix, LING_LORA_MK=1's
# spec-check), w8 (speed), w4's agreement half (LORA_W4_AGREE_ONLY), then w5 (the engine on the same cases, cost).
#   chain4.sh OUT   (setsid nohup ... < /dev/null &)
H=$(cd "$(dirname "$0")" && pwd); O=$1
$H/window.sh w7 "the pdl_begin() fix, spec-check" $H/w7.steps $O/w7
$H/window.sh w8 "speed with one adapter" $H/w8.steps $O/w8
LORA_W4_AGREE_ONLY=1 $H/window.sh w4 "agreement: SGLang with an adapter" $H/w4.steps $O/w4b
[ -s $HOME/lora-agree/cases.jsonl ] && $H/window.sh w5 "agreement: the engine, adapter cost" $H/w5.steps $O/w5
