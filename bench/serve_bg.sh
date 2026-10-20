#!/bin/bash
# Start ling-serve in the background on node-b and return once it answers /health, so that a
# caller over ssh gets its prompt back:
#
#   ssh node-b '~/ling-engine/bench/serve_bg.sh 18081'                      # main's build
#   ssh node-b '~/ling-engine/bench/serve_bg.sh 18081 ~/ling-engine-x/build --draft-block 16'
#   ssh node-b '~/ling-engine/bench/serve_bg.sh stop 18081'
#
# Arguments: the port, then optionally a build directory, then extra ling-serve flags (they replace
# the defaults below only where ling-serve lets a later flag win). Log: /tmp/ling-serve-<port>.log.
#
# Why a script: `ssh host 'cd dir && nohup server > log 2>&1 &'` never returns. The `&` applies to
# the whole `cd && nohup` list, so bash forks a subshell for it; that subshell keeps ssh's stdout
# and stderr open (the redirection covers only the server) and lives as long as the server, so sshd
# holds the session open. Closing stdin does not help. On 2026-10-11 a test run waited 1 h 33 behind
# such an ssh. Here the server is started from a plain `;` sequence with every descriptor redirected
# and in its own session, so nothing of the ssh channel outlives this script.
set -u
if [ "${1:-}" = stop ]; then
    port=${2:?usage: serve_bg.sh stop <port>}
    for p in $(pgrep -f "build/ling-serve .*--port $port( |$)"); do kill "$p"; done
    exit 0
fi
port=${1:?usage: serve_bg.sh <port> [build-dir] [ling-serve flags...]}
shift
build=$HOME/ling-engine/build
if [ $# -gt 0 ] && [ -d "$1" ]; then build=$1; shift; fi
if pgrep -f "build/ling-serve .*--port $port( |$)" > /dev/null; then
    echo "❌ a ling-serve already listens on $port"; exit 1
fi
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
D=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
log=/tmp/ling-serve-$port.log
cd "$build/.." || exit 1
setsid nohup ./build/ling-serve --model "$M" --draft "$D" --draft-block 12 --port "$port" \
    --served-model-name RadixArk/Qwen3.8-27B-NVFP4 --max-context 65536 --graphs --pdl "$@" \
    > "$log" 2>&1 < /dev/null &
pid=$!
for _ in $(seq 1 100); do
    if curl -sf -m 2 "http://127.0.0.1:$port/health" > /dev/null; then
        echo "✅ ling-serve up on $port (pid $pid, $build, log $log)"; exit 0
    fi
    kill -0 "$pid" 2> /dev/null || { echo "❌ ling-serve exited; tail of $log:"; tail -5 "$log"; exit 1; }
    sleep 3
done
echo "❌ no /health on $port after 5 min; log $log"; exit 1
