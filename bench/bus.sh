#!/bin/bash
# BACKLOG §19.25: decode beside a CPU neighbour on node-b. The seeded replay (M0's main set, seed 1013) at one
# and two streams on BUILD, alone and beside: a synthetic memory stream at 5, 10 and 20 GB/s of traffic
# (bench/bus_neighbour.c, five threads pinned to the A725 cores 0-4, away from the X925 cores), the
# real ling-code indexer (re-indexing REPO in a loop), and a pytest container (python:3-slim with Mightling's Python
# suite, PYIMG, in a loop). Each neighbour starts 30 s before the replay and stops after it. Uses ~/steps/replay.sh
# (production stopped by the caller, ~/gpu.lock).
#   [ARMS="alone index"] [CONCS="1 2"] bench/bus.sh BUILD OUT REPO PYIMG PYSRC
set -u
B=${1:?build root}; O=${2:?out}; REPO=${3:?repo to index}; PYIMG=${4:?pytest image}; PYSRC=${5:?python source}
H=$(cd "$(dirname "$0")" && pwd); mkdir -p "$O"
gcc -O2 -pthread -o "$O/bus_neighbour" "$H/bus_neighbour.c" || exit 1
NB=""
start() {  # kind arg
  case $1 in
  mem) "$O/bus_neighbour" "$2" 5 0 > "$O/nb-$LABEL.txt" 2>&1 & NB=$! ;;
  index) # a user bus, as a login shell has: without one ling-code only queues the request for a session
    export XDG_RUNTIME_DIR=/run/user/$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus
    mkdir -p ~/.cache/codebase-memory-mcp
    (cd "$REPO" && while true; do ~/.local/bin/ling-code forget > /dev/null 2>&1; /usr/bin/time -f "index %e s %M KB" ~/.local/bin/ling-code index --wait; done) > "$O/nb-$LABEL.txt" 2>&1 & NB=$! ;;
  pytest) (while true; do docker run --rm --name steps-bus-pytest -v "$PYSRC":/src:ro -w /src "$PYIMG" sh -c "cp -r /src /tmp/w && cd /tmp/w && python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -1"; done) > "$O/nb-$LABEL.txt" 2>&1 & NB=$! ;;
  esac
  sleep 30
}
stop() {
  [ -n "$NB" ] && { pkill -P $NB 2>/dev/null; kill $NB 2>/dev/null; }
  docker rm -f steps-bus-pytest > /dev/null 2>&1; pkill -x bus_neighbour 2>/dev/null; pkill -f "ling-code index" 2>/dev/null
  NB=""; sleep 10
}
for c in ${CONCS:-1 2}; do
  for arm in ${ARMS:-alone mem5 mem10 mem20 index pytest}; do
    LABEL=bus-$arm-c$c
    case $arm in alone) ;; mem*) start mem ${arm#mem} ;; index) start index ;; pytest) start pytest ;; esac
    ~/steps/replay.sh "$B" "$LABEL" $c
    stop
  done
  [ -z "${ARMS:-}" ] && { LABEL=bus-alone2-c$c; ~/steps/replay.sh "$B" "$LABEL" $c; }   # the baseline again, for drift
done
echo bus-done
