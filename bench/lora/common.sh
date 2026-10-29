# The adapters windows' common names (sourced by each wN.steps): SRC the branch's checkout, MAIN origin/main's,
# LIB the test adapters (make_adapter.py: zero, rand, rand-attn), the model, the production drafter and the prompts.
SRC=~/ling-engine-lora; B=$SRC/build; MAIN=$HOME/ling-engine-lmain/build; LIB=$HOME/lora-lib
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
D=$HOME/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
S=$HOME/integ-exact/spec.jsonl; PB=$HOME/lv/prefix-branch.jsonl
BASE=RadixArk/Qwen3.8-27B-NVFP4
al() {  # al LABEL BUILD [MODEL [ling-serve flags]]: agentlong greedy on ling-serve from BUILD
  local label=$1 build=$2 model=${3:-}; shift 3 2>/dev/null || shift $#
  setsid $build/ling-serve --model $M --draft $D --port 8300 --served-model-name $BASE --max-context 65536 \
    --draft-block 12 --graphs --pdl "$@" > $V/$label.serve.log 2>&1 8>&- < /dev/null &
  local srv=$!
  for i in $(seq 1 120); do curl -sf -m 2 localhost:8300/health > /dev/null && break; sleep 3; done
  python3 $SRC/bench/agentlong/run.py --url http://127.0.0.1:8300 --mode greedy --label $label --out $V/$label.jsonl \
    ${model:+--model $model}
  curl -s -m 10 localhost:8300/spec_stats > $V/$label.spec.json 2>/dev/null
  kill $srv; wait $srv 2>/dev/null
}
