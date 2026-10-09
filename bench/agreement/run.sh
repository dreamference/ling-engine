#!/bin/bash
# The agreement study (reports/M2-prefill.md section 6; SPEC section 14's standing gate), on the machine that holds the
# prompts. Production first (the reference server, untouched apart from two prefix-cache flushes), then each build.
#
#   run.sh STUDY_DIR MODEL_DIR SHORT.jsonl LONG.jsonl m1=PATH/ling-force b44=PATH/ling-force m2=PATH/ling-force
#
# The labels m1, b44 and m2 are the ones analyze.py and extra.py report on. Then:
#   python3 analyze.py STUDY_DIR; python3 extra.py STUDY_DIR; decode_worst.py inside the reference image.
set -e
H=$(cd "$(dirname "$0")" && pwd)
D=$1; M=$2; SHORT=$3; LONG=$4; shift 4
mkdir -p "$D" && cd "$D"
python3 $H/mkprompts.py "$D" "$SHORT" "$LONG"
first=${1#*=}
$first tokenize $M < prompts.jsonl > ids.jsonl
python3 -c "import json; L=[l for l in open('ids.jsonl')]; open('ids-long.jsonl','w').write(''.join(L[30:]))"
P="python3 $H/prod.py"
$P gen ids.jsonl prod-gen48.jsonl 48 1                                 # production's greedy continuation
$P gen ids.jsonl prod-gen48-c1-run2.jsonl 48 1                         # again (cache now warm)
$P gen ids.jsonl prod-gen48-c4.jsonl 48 4                              # four at a time
$P flush
$P force ids.jsonl prod-gen48.jsonl prod-force-cold-c1.jsonl 1         # the reference
$P force ids.jsonl prod-gen48.jsonl prod-force-warm-c1.jsonl 1
$P flush
$P force ids.jsonl prod-gen48.jsonl prod-force-cold-c4.jsonl 4
$P force ids.jsonl prod-gen48.jsonl prod-force-warm-c4.jsonl 4
$P gen ids-long.jsonl prod-gen400-long.jsonl 400 1                     # first responses of the agent prompts
$P gen ids-long.jsonl prod-gen400-long-run2.jsonl 400 1
python3 - <<'PY'
import json
ids = [json.loads(l)["ids"] for l in open("ids.jsonl")]
gen = [json.loads(l) for l in open("prod-gen48.jsonl")]
ref = [json.loads(l) for l in open("prod-force-cold-c1.jsonl")]
with open("cases.jsonl", "w") as f:
    for p, g, r in zip(ids, gen, ref):
        f.write(json.dumps({"prompt": p, "cont": g["cont"], "want": [[c[0] for c in pos] for pos in r["top"]]}) + "\n")
with open("cases-long.jsonl", "w") as f:
    for p in ids[30:]:
        f.write(json.dumps({"prompt": p}) + "\n")
PY
for b in "$@"; do
  n=${b%%=*}; bin=${b#*=}
  $bin score $M 20 < cases.jsonl > eng-$n.jsonl 2> eng-$n.err
  $bin gen $M 400 < cases-long.jsonl > gen400-$n.jsonl 2>> eng-$n.err
done
