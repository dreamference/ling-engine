#!/bin/bash
# Sequential bench driver; every step stops starting new cells after 19:40 UTC.
cd "$(dirname "$0")"
B="python3 -u fanout_bench.py"
NS=1,4,8,16,24,32,64,256,1000
$B fanout --ctx 32000  --n $NS --out fanout.jsonl --src ctx600k.txt
$B fanout --ctx 100000 --n $NS --out fanout.jsonl --src ctx600k.txt
$B tail   --ctx 32000  --n 4096 --out tail.jsonl --src ctx600k.txt
$B tail   --ctx 100000 --n 4096 --out tail.jsonl --src ctx600k.txt
$B score  --ctx 32000  --n $NS --out score.jsonl --src ctx600k.txt
$B score  --ctx 100000 --n $NS --out score.jsonl --src ctx600k.txt
$B chat   --n 1,4,16,64,256,1000 --out chat.jsonl --src ctx600k.txt
# batch sizes 2-8 step-time sweep (run separately into fanout_bs.jsonl):
# $B fanout --ctx 32000 --n 2,3,5,6,7 --out fanout_bs.jsonl --src ctx600k.txt
$B unshared --ctx 32000 --n 1,4,8 --out unshared.jsonl --src ctx600k.txt
echo ALLDONE
