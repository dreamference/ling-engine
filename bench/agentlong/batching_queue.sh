#!/bin/bash
# batching_queue.sh OUT: the batching measurement's GPU windows (ROUND 1: cold turns; ROUND 2: warm follow-ups and SGLang's request log; the engine with prompt_cache_key, as Codex sends it) on node-b (BATCHING §2): never while a SWE-bench
# night holds the machine, otherwise each window blocks on the GPU lock (flock, in streams_window.sh) like every other
# job, holds it <= 45 min and releases it between windows, pausing two minutes so a job that waits takes it first. No
# window starts after LAST_START (default 2026-10-27 19:10 UTC, so the last window ends by 20:00, before night 17). Status in
# OUT/queue.status, log in OUT/queue.log.
set -u
O=${1:-$HOME/batching/runs}; mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
W=$HERE/streams_window.sh
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/queue.log" > "$O/queue.status"; }
night() {  # a SWE-bench night's lease or containers on this machine
  pgrep -f "ab-n[0-9]+-gpu-lease" > /dev/null || [ -n "$(docker ps -q -f name=^ab-n)" ]
}
window() {  # name description items...
  local name=$1 desc=$2; shift 2
  while night; do st "before $name: a night holds the machine"; sleep 120; done
  if [ "$(date -u +%s)" -ge "$(date -u -d "${LAST_START:-2026-10-27 19:10}" +%s)" ]; then st "$name not started: past ${LAST_START:-2026-10-27 19:10} UTC"; return 1; fi
  st "running $name"
  bash "$W" "$O/$name" "$desc" "$@" > "$O/$name.log" 2>&1
  st "done $name (exit $?)"
  sleep 120  # a job that queued meanwhile takes the lock first
}
E=engine; S=sglang
if [ "${ROUND:-1}" = 1 ]; then
window w1-engine-greedy "batching W1: ling-serve main, agentlong N streams greedy (1, 2, 4 with 4 sessions, 4 with 2, 3 with 3); <= 45 min, m0 stopped" \
  "$E:e1g:1:greedy" "$E:e2g:2:greedy" "$E:e4g-s4:4:greedy:--sessions,4,--batch,4" "$E:e4g-s2:4:greedy" \
  "$E:e3g-s3:3:greedy:--sessions,3,--batch,3"
window w2-sglang-greedy "batching W2: SGLang v0.5.21 phase merge, agentlong N streams greedy (1, 2, 4, 8); <= 45 min, m0 stopped" \
  "sglang-up:sg" "$S:s1g:1:greedy" "$S:s2g:2:greedy" "$S:s4g:4:greedy" "$S:s8g:8:greedy" "sglang-down"
window w3-engine-sampled "batching W3: ling-serve main, agentlong N streams sampled seed 1013 (1, 2, 4 with 4 sessions, 4 with 2, 3 with 3); <= 45 min, m0 stopped" \
  "$E:e1s:1:seed1013" "$E:e2s:2:seed1013" "$E:e4s-s4:4:seed1013:--sessions,4,--batch,4" "$E:e4s-s2:4:seed1013" \
  "$E:e3s-s3:3:seed1013:--sessions,3,--batch,3"
window w4-sglang-sampled "batching W4: SGLang v0.5.21 phase merge, agentlong N streams sampled seed 1013 (1, 2, 4, 8); <= 45 min, m0 stopped" \
  "sglang-up:sg" "$S:s1s:1:seed1013" "$S:s2s:2:seed1013" "$S:s4s:4:seed1013" "$S:s8s:8:seed1013" "sglang-down"
window w5-profile "batching W5: the step at one and two sequences in place and under nsys (31K, 124K), memory per session; <= 45 min, m0 stopped" \
  "run:$HERE/batch_profile.sh $O/w5-profile/prof:profile"
elif [ "${ROUND}" = 2 ]; then
# Round 2 (2026-10-27, after node-b froze at 08:42 under W6's 8-session item: at most 4 sessions from here on).
window w11-engine-warm-greedy-key "batching W11: ling-serve main, agentlong warm follow-ups with prompt_cache_key, greedy (1, 2, 4 and 3 with as many sessions); <= 45 min, m0 stopped" \
  "$E:k1g:1:warm-greedy" "$E:k2g:2:warm-greedy" "$E:k4g:4:warm-greedy:--sessions,4,--batch,4" \
  "$E:k3g:3:warm-greedy:--sessions,3,--batch,3"
window w7-sglang-warm-greedy "batching W7: SGLang v0.5.21 phase merge, agentlong warm follow-ups, N streams greedy (1, 2, 4, 8), request log; <= 45 min, m0 stopped" \
  "sglang-up:sg" "$S:w1g:1:warm-greedy" "$S:w2g:2:warm-greedy" "$S:w4g:4:warm-greedy" "$S:w8g:8:warm-greedy" "sglang-down"
window w9-engine-warm-sampled-key "batching W9: ling-serve main, agentlong warm follow-ups with prompt_cache_key, sampled seed 1013 (1, 2, 4 with 4 sessions), cold one stream again; <= 45 min, m0 stopped" \
  "$E:k1s:1:warm-seed1013" "$E:k2s:2:warm-seed1013" "$E:k4s:4:warm-seed1013:--sessions,4,--batch,4" "$E:e1s:1:seed1013"
window w8-sglang-cold-greedy "batching W8: SGLang v0.5.21 phase merge, agentlong cold turns, N streams greedy (1, 2, 4, 8), request log; <= 45 min, m0 stopped" \
  "sglang-up:sg" "$S:s1g:1:greedy" "$S:s2g:2:greedy" "$S:s4g:4:greedy" "$S:s8g:8:greedy" "sglang-down"
window w10-sglang-warm-sampled "batching W10: SGLang v0.5.21 phase merge, agentlong warm follow-ups sampled seed 1013 (1, 2, 4, 8), request log; <= 45 min, m0 stopped" \
  "sglang-up:sg" "$S:w1s:1:warm-seed1013" "$S:w2s:2:warm-seed1013" "$S:w4s:4:warm-seed1013" "$S:w8s:8:warm-seed1013" "sglang-down"
fi
st "all done"
