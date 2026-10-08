#!/bin/bash
# Nsight Systems profile of production SGLang on replayed agent sessions (M0, deliverable 2).
#
# Stops the production-config server on :8000, starts the same configuration under nsys with NVTX
# ranges around every phase of a DFLASH step (patch_dflash_nvtx.py) and SGLang's scheduler spans
# (nvtx_shim.py), records one nsys session window per replay (single stream, then four streams), and
# restarts production. ~/.ling-quiet is set only while the windows run.
#
# Usage: profile.sh TAG [extra serve.py args...]
set -u
TAG=$1; shift
cd ~/m0
touch ~/m0/SGLANG-DOWN
docker stop dreamference-vllm-8000 > /dev/null
mkdir -p out/nsys-tmp
python3 serve.py m0-nsys --port 30000 --nsys --nvtx-patch dflash_worker_v2_nvtx.py \
  --nvtx-shim nvtx_shim.py --log-requests "$@"
for i in $(seq 1 120); do curl -sf localhost:30000/v1/models > /dev/null && break; sleep 10; done
curl -sf localhost:30000/v1/models > /dev/null || { echo "server did not start"; docker logs --tail 50 m0-nsys; }

R="python3 replay.py --url http://127.0.0.1:30000 --tools tools-capture.json --tools-index tools-index.json"
# Warm-up: compile paths and graphs, outside any capture range.
touch ~/.ling-quiet   # quiet only while capturing; the server start needs none
sleep 45              # the other worker's compiles pause at their next check
# Bus latency and SM scaling (spec 16.1 items 1-2) while the server is loaded but idle.
[ -x membw/latency ] && membw/latency > "runs/$TAG-latency.txt" 2>&1
$R --sessions-file set-prof-warm.txt --window 3 --label "$TAG-warm" --out "runs/$TAG-warm.jsonl" > /dev/null 2>&1

prof() {  # label concurrency sessions window
  docker exec m0-nsys /nsys/bin/nsys start --session=m0 -o "/out/$1" --force-overwrite=true
  $R --sessions-file "$3" --window "$4" --concurrency "$2" --label "$1" --out "runs/$1.jsonl" > "runs/$1.log" 2>&1
  docker exec m0-nsys /nsys/bin/nsys stop --session=m0
  sleep 10
}
prof "$TAG-c1" 1 set-prof-c1.txt 6
prof "$TAG-c4" 4 set-prof-c4.txt 5
rm -f ~/.ling-quiet

docker logs m0-nsys > "runs/$TAG.server.log" 2>&1
docker stop -t 60 m0-nsys > /dev/null
docker rm m0-nsys > /dev/null
docker start dreamference-vllm-8000 > /dev/null
for i in $(seq 1 120); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
rm -f ~/m0/SGLANG-DOWN
ls -la out/ | grep "$TAG"
