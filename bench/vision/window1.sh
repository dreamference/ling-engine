#!/bin/bash
# Images, GPU window 1 (VISION §20.8): production's references, the tower's first run against them, production's
# answers on the image prompt set. At most 45 minutes under ~/.ling-gpu.lock, m0-prod8000 stopped and restored.
#   window1.sh SRC V      SRC: the vision worktree (built); V: ~/vision (images/, photos/, prompts.jsonl)
set -u
SRC=$1; V=$2; R=$V/ref; mkdir -p $R
HUB=$HOME/.cache/huggingface
CK=$HUB/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/52d1adc5f38aa5ebf099c29ed7025ba34cfbb854
IMG=lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469   # production's pin (v0.5.21)
PORT=18100
exec 8> $HOME/.ling-gpu.lock
echo "$(date -u +%FT%TZ) vision-agent waiting for the lock (images window 1: production's references, the tower's first run, production's answers on 24 image prompts; <= 45 min, m0 stopped)" >> ~/.ling-gpu-holds.log
flock 8; T0=$(date -u +%s)
echo "$(date -u +%FT%TZ) vision-agent lock taken" >> ~/.ling-gpu-holds.log
left() { echo $(( T0 + 45 * 60 - $(date -u +%s) )); }
M0=$(docker inspect -f '{{.State.Running}}' m0-prod8000 2>/dev/null || echo false)
[ "$M0" = true ] && docker stop m0-prod8000 > /dev/null
echo "=== $(date -u +%T) kernel tests"
$SRC/build/ling-vision-tests > $V/w1-vision-tests.log 2>&1; echo "vision tests exit $?"
echo "=== $(date -u +%T) references in production's image"
IMAGES=$(ls $V/images/*.png $V/photos/*)
timeout 900 docker run --rm --gpus all --network none --name vision-refs -v $HUB:$HUB:ro -v $V:$V -v $SRC:$SRC:ro \
  -e HF_HUB_OFFLINE=1 --entrypoint python3 $IMG $SRC/bench/vision/reference.py $CK $R $IMAGES > $V/w1-reference.log 2>&1
echo "reference exit $?"
echo "=== $(date -u +%T) the tower against the references ($(left) s left)"
timeout 900 $SRC/build/ling-run --model $CK --vision-check $R --vision-bench > $V/w1-vision-check.log 2>&1
echo "vision check exit $?"
if [ "$(left)" -gt 1200 ]; then
  echo "=== $(date -u +%T) production's server on $PORT"
  docker rm -f vision-ref > /dev/null 2>&1
  docker run -d --ipc=host --network host --name vision-ref --gpus all --cpus=14.0 --memory=85g --memory-swap=85g \
    -v $HUB:/root/.cache/huggingface -v $HOME/.cache/dreamference:/root/.cache/dreamference -e HF_HUB_OFFLINE=1 \
    -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/dreamference/sglang/inductor $IMG python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/52d1adc5f38aa5ebf099c29ed7025ba34cfbb854 \
    --trust-remote-code --served-model-name RadixArk/Qwen3.8-27B-NVFP4 --host 127.0.0.1 --port $PORT --tp-size 1 \
    --mem-fraction-static 0.5 --context-length 262144 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-algorithm DFLASH --speculative-draft-model-path /root/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch \
    --speculative-num-draft-tokens 12 --speculative-draft-model-quantization modelopt_fp4 \
    --chat-template /root/.cache/dreamference/chat-templates/RadixArk--Qwen3.8-27B-NVFP4-52d1adc5f38a-c84b7d991a0e.jinja \
    --attention-backend flashinfer --sampling-backend pytorch --chunked-prefill-size 8192 --disable-prefill-cuda-graph \
    --disable-flashinfer-autotune --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 96 --max-running-requests 8 --enable-torch-compile --torch-compile-max-bs 4 \
    --num-continuous-decode-steps 2 --enable-metrics --cuda-graph-max-bs-decode 8 > /dev/null
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 10; done
  echo "server healthy after $((i * 10)) s"
  export LING_REF_URL=http://127.0.0.1:$PORT
  P="timeout 600 python3 $SRC/bench/vision/prod_vision.py"
  $P render $V/prompts.jsonl $V/prod-render.jsonl > $V/w1-prod.log 2>&1; echo "render exit $?"
  $P gen $V/prompts.jsonl $V/prod-render.jsonl $V/prod-gen.jsonl 48 >> $V/w1-prod.log 2>&1; echo "gen exit $?"
  curl -s -X POST http://127.0.0.1:$PORT/flush_cache > /dev/null
  $P force $V/prompts.jsonl $V/prod-render.jsonl $V/prod-gen.jsonl $V/prod-force.jsonl >> $V/w1-prod.log 2>&1; echo "force exit $?"
  # Production's time to first token on the image prompts, cold (the prefix cache flushed first).
  curl -s -X POST http://127.0.0.1:$PORT/flush_cache > /dev/null
  python3 - $V/prompts.jsonl > $V/w1-prod-ttft.log 2>&1 <<'PY'
import base64, json, os, sys, time, urllib.request
base = os.path.dirname(sys.argv[1])
for line in open(sys.argv[1]):
    body = json.loads(line)
    for m in body["messages"]:
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if "image_url" in p:
                    u = p["image_url"]["url"]
                    p["image_url"]["url"] = "data:image;base64," + base64.b64encode(open(os.path.join(base, u), "rb").read()).decode()
    body.update(model="RadixArk/Qwen3.8-27B-NVFP4", max_tokens=1, temperature=0)
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(os.environ["LING_REF_URL"] + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
    print(f"{time.time() - t:.3f} s, prompt {r['usage']['prompt_tokens']} tokens")
PY
  docker rm -f vision-ref > /dev/null
fi
[ "$M0" = true ] && docker start m0-prod8000 > /dev/null
echo "=== released $(date -u +%T)"
echo "$(date -u -d @$T0 +%F\ %T)-$(date -u +%T) UTC vision-agent images window 1 (m0 was running: $M0; restarted: $M0)" >> ~/.ling-gpu-holds.log
