#!/bin/bash
# admission_check.sh OUTDIR [BUILD] [REFERENCE.jsonl]: the checks of ling-serve's admission control on host memory
# (serve/admission.hpp; BATCHING §5.6, phase 0). Run inside a GPU window (streams_window.sh run:...), m0 stopped.
#   1. The start refused before anything is allocated: a reserve no host has (LING_HOST_RESERVE_MB), and 8 sessions
#      at 262,144 tokens; exit 3 and the reason, the GPU untouched.
#   2. A request refused while serving: LING_ADMIT_RESERVE_MB above MemAvailable; /v1/responses, /v1/chat/completions
#      and /v1/completions answer HTTP 503 with code host_memory_low and Retry-After, /health and /v1/models still
#      answer, ling:memory_refusals_total counts them.
#   3. No output change: the default server, agentlong greedy at one stream (n_streams.py), compared turn for turn
#      with REFERENCE (a run of the build before the change).
#   4. Auto-size (BATCHING §5.6, 2026-10-28): with LING_AUTO_SIZE=0 the old defaults (2, 2, 65,536); a start with
#      no --sessions, --batch or --max-context logs all three as auto (on node-b with m0 stopped: 3 sessions, batch
#      3, 262,144) and MemAvailable falls by about the plan (plus the prefix KV store's 4 GB, which the plan leaves
#      out), not more. Then agentlong greedy at the chosen sessions' streams on the auto server and on one started
#      with the same three flags given: the two size lines name the same values and the outputs are identical.
set -u
O=$1; BUILD=${2:-$HOME/le-adm/build}; REF=${3:-}
mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
D=$HOME/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
S="$BUILD/ling-serve --model $M --draft $D --served-model-name RadixArk/Qwen3.8-27B-NVFP4 --draft-block 12 --graphs --pdl"
ok=0; bad=0
pass() { echo "PASS $*"; ok=$((ok + 1)); }
fail() { echo "FAIL $*"; bad=$((bad + 1)); }

echo "--- 1. refused at start ($(date -u +%T))"
LING_HOST_RESERVE_MB=10000000 $S --port 8302 --max-context 65536 > "$O/start1.log" 2>&1; rc=$?
[ $rc = 3 ] && grep -q "refused to start" "$O/start1.log" && ! grep -q "^loading" "$O/start1.log" \
  && pass "reserve no host has: exit 3, nothing loaded" || fail "reserve no host has: exit $rc"
$S --port 8302 --max-context 262144 --sessions 8 > "$O/start2.log" 2>&1; rc=$?
[ $rc = 3 ] && grep -q "8 sessions x 17.7 GB" "$O/start2.log" && pass "8 sessions at 262,144: exit 3" \
  || fail "8 sessions at 262,144: exit $rc"

echo "--- 2. refused while serving ($(date -u +%T))"
LING_ADMIT_RESERVE_MB=10000000 setsid $S --port 8302 --max-context 65536 > "$O/serve2.log" 2>&1 < /dev/null &
P=$!
"$HERE/mem_guard.sh" "pid:$P" "$O/mem_guard.log" < /dev/null &
for _ in $(seq 1 120); do curl -sf -m 2 localhost:8302/health > /dev/null && break; sleep 3; done
for route in responses chat/completions completions; do
  case $route in
    responses) body='{"model":"m","input":"Say hi.","max_output_tokens":8,"stream":true}' ;;
    chat/completions) body='{"model":"m","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}' ;;
    completions) body='{"model":"m","prompt":"Say hi.","max_tokens":8}' ;;
  esac
  code=$(curl -s -o "$O/r503.json" -D "$O/r503.hdr" -w '%{http_code}' -m 30 -H 'Content-Type: application/json' \
    -d "$body" "localhost:8302/v1/$route")
  [ "$code" = 503 ] && grep -q '"host_memory_low"' "$O/r503.json" && grep -qi '^retry-after: 5' "$O/r503.hdr" \
    && pass "/v1/$route: 503 host_memory_low" || fail "/v1/$route: HTTP $code $(head -c 200 "$O/r503.json")"
done
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8302/health)" = 200 ] && pass "/health answers" || fail "/health"
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8302/v1/models)" = 200 ] && pass "/v1/models answers" || fail "/v1/models"
n=$(curl -s localhost:8302/metrics | awk '/^ling:memory_refusals_total/ {print $2}')
[ "${n%.*}" = 3 ] && pass "ling:memory_refusals_total 3" || fail "ling:memory_refusals_total '$n'"
kill $P; wait $P 2> /dev/null

echo "--- 3. no output change ($(date -u +%T))"
setsid $S --port 8302 --max-context 65536 > "$O/serve3.log" 2>&1 < /dev/null &
P=$!
"$HERE/mem_guard.sh" "pid:$P" "$O/mem_guard.log" < /dev/null &
for _ in $(seq 1 120); do curl -sf -m 2 localhost:8302/health > /dev/null && break; sleep 3; done
grep -q "^size: " "$O/serve3.log" && pass "the plan logged: $(grep '^size: ' "$O/serve3.log")" || fail "no plan line"
python3 "$HERE/n_streams.py" --url http://127.0.0.1:8302 --streams 1 --mode greedy --label adm1g --cache-key \
  --out "$O/adm1g.jsonl" > "$O/adm1g.log" 2>&1
kill $P; wait $P 2> /dev/null
if [ -n "$REF" ]; then
  python3 "$HERE/n_streams.py" --compare "$REF" "$O/adm1g.jsonl" > "$O/compare.txt" 2>&1 \
    && pass "agentlong greedy: $(tail -1 "$O/compare.txt")" || fail "agentlong greedy: $(tail -1 "$O/compare.txt")"
fi
echo "--- 4. auto-size ($(date -u +%T))"
avail() { awk '/^MemAvailable:/ {printf "%.1f", $2 * 1024 / 1e9}' /proc/meminfo; }  # GB (10^9)
serve() {  # serve LOG ARGS...: start ling-serve, wait for /health; P is its pid
  local log=$1; shift
  setsid $S --port 8302 "$@" > "$log" 2>&1 < /dev/null &
  P=$!
  "$HERE/mem_guard.sh" "pid:$P" "$O/mem_guard.log" < /dev/null &
  for _ in $(seq 1 120); do curl -sf -m 2 localhost:8302/health > /dev/null && break; sleep 3; done
}
LING_AUTO_SIZE=0 serve "$O/serve4off.log"
grep -q "^size: 2 sessions (default, --auto-size 0), batch 2 (default, --auto-size 0), context 65536 " "$O/serve4off.log" \
  && pass "LING_AUTO_SIZE=0: the old defaults" || fail "LING_AUTO_SIZE=0: $(grep '^size: ' "$O/serve4off.log")"
kill $P; wait $P 2> /dev/null; sleep 5
before=$(avail)
serve "$O/serve4.log"
after=$(avail); line=$(grep '^size: ' "$O/serve4.log")
rss=$(awk '/^VmRSS:/ {printf "%.1f GB", $2 * 1024 / 1e9}' "/proc/$P/status" 2> /dev/null)
echo "$line"; echo "MemAvailable $before -> $after GB; ling-serve RSS $rss"
N=$(echo "$line" | sed -n 's/^size: \([0-9]*\) sessions (auto), batch \([0-9]*\) (auto), context \([0-9]*\) (auto).*/\1/p')
B=$(echo "$line" | sed -n 's/^size: [0-9]* sessions (auto), batch \([0-9]*\) (auto).*/\1/p')
C=$(echo "$line" | sed -n 's/.*, context \([0-9]*\) (auto).*/\1/p')
planned=$(echo "$line" | sed -n 's/.*, planned \([0-9.]*\) GB.*/\1/p')
[ -n "$N" ] && [ -n "$B" ] && [ -n "$C" ] && pass "all three chosen: $N sessions, batch $B, context $C" \
  || fail "not all auto: $line"
python3 -c "import sys; b, a, p = map(float, sys.argv[1:]); sys.exit(0 if b - a <= p + 6 else 1)" \
  "$before" "$after" "${planned:-0}" && pass "MemAvailable fell $(python3 -c "print(round($before - $after, 1))") GB, planned $planned" \
  || fail "MemAvailable fell $before -> $after GB, more than the plan ($planned) + 6"
python3 "$HERE/n_streams.py" --url http://127.0.0.1:8302 --streams "${N:-1}" --mode greedy --label adm4auto --cache-key \
  --out "$O/adm4auto.jsonl" > "$O/adm4auto.log" 2>&1
kill $P; wait $P 2> /dev/null; sleep 5
serve "$O/serve4given.log" --sessions "${N:-1}" --batch "${B:-1}" --max-context "${C:-65536}"
grep -q "^size: ${N:-x} sessions (given), batch ${B:-x} (given), context ${C:-x} (given)" "$O/serve4given.log" \
  && pass "the same values given: the same size line" || fail "given: $(grep '^size: ' "$O/serve4given.log")"
python3 "$HERE/n_streams.py" --url http://127.0.0.1:8302 --streams "${N:-1}" --mode greedy --label adm4given --cache-key \
  --out "$O/adm4given.jsonl" > "$O/adm4given.log" 2>&1
kill $P; wait $P 2> /dev/null
python3 "$HERE/n_streams.py" --compare "$O/adm4given.jsonl" "$O/adm4auto.jsonl" > "$O/compare4.txt" 2>&1 \
  && pass "auto against given, $N streams: $(tail -1 "$O/compare4.txt")" || fail "auto against given: $(tail -1 "$O/compare4.txt")"

echo "admission check: $ok passed, $bad failed ($(date -u +%T))"
[ $bad = 0 ]
