#!/bin/bash
# Runs tuning variants back to back with :8000 down, then restores production.
# Usage: tunebatch.sh "LABEL|serve.py args[|REPLAYS]" ...   (REPLAYS: see tune.sh)
cd ~/m0
TUNE_SH=${TUNE_SH:-./tune.sh}
touch ~/m0/SGLANG-DOWN
docker stop dreamference-vllm-8000 > /dev/null 2>&1
for v in "$@"; do
  IFS='|' read -r label args replays <<< "$v"
  echo "=== $label: $args  [$replays]  $(date)"
  REPLAYS="$replays" $TUNE_SH "$label" -- $args
done
# Restore the reference server on :8000: production's own container, or RESTORE (a command).
if [ -n "${RESTORE:-}" ]; then sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; eval "$RESTORE"
else docker start dreamference-vllm-8000 > /dev/null; fi
for i in $(seq 1 120); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
rm -f ~/m0/SGLANG-DOWN
echo "=== done $(date)"
