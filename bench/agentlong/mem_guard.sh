#!/bin/bash
# mem_guard.sh pid:PID|container:NAME LOG: kills a model server before the host's memory runs out, with the rule
# Mightling's server start uses (dreamference/vllm_server/psi_watchdog.py): the kernel's memory pressure `full avg10`
# at or above 60% for 5 s, or `full avg60` at or above 25%; and, before pressure builds, MemAvailable below a floor
# (MEM_FLOOR_MB, default 8,192: above earlyoom's 5% line on this 121 GB machine). On this unified-memory machine a
# GPU allocation is host memory that no cgroup is charged for, so neither docker's --memory nor earlyoom stops a
# server in time: node-b froze on 2026-10-27 at 08:42 UTC thirty seconds into an 8-session ling-serve run.
# Exits when the target is gone. Run it beside every server a bench starts.
set -u
T=$1; LOG=${2:-/dev/stderr}
FLOOR=${MEM_FLOOR_MB:-8192}
alive() { case $T in pid:*) kill -0 "${T#pid:}" 2> /dev/null ;; container:*) [ -n "$(docker ps -q -f "name=^${T#container:}$")" ] ;; esac; }
kill_it() {
  echo "$(date -u +%FT%TZ) mem_guard: $1; killing $T" >> "$LOG"
  case $T in pid:*) kill -9 "${T#pid:}" ;; container:*) docker kill "${T#container:}" > /dev/null ;; esac
}
over=0
while alive; do
  read -r a10 a60 < <(awk '/^full/ { for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] } print v["avg10"], v["avg60"] }' /proc/pressure/memory)
  avail=$(awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo)
  if [ "$avail" -lt "$FLOOR" ]; then kill_it "MemAvailable ${avail} MB below the ${FLOOR} MB floor"; break; fi
  if awk -v x="$a60" 'BEGIN { exit !(x >= 25) }'; then kill_it "memory pressure full avg60 ${a60}%"; break; fi
  if awk -v x="$a10" 'BEGIN { exit !(x >= 60) }'; then over=$((over + 1)); else over=0; fi
  if [ $over -ge 5 ]; then kill_it "memory pressure full avg10 ${a10}% for 5 s"; break; fi
  sleep 1
done
