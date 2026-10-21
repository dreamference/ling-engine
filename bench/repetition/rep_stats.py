"""Repetition statistics of generated token streams (API section 9: the repetition stop rule's default thresholds).

rep_stats.py dump DUMP...    the LING_DUMP_STEPS files: each step's emitted tokens (the accepted drafts and the next
                             token), split into requests where the position does not continue
rep_stats.py ids FILE...     one JSON array of token ids per line
For each stream: the largest repeat count of a period-P block ending anywhere (P = 1..PMAX, PMAX=128 or the
environment's), and the longest run of one token; then, over all streams, the worst counts per period.
The rollouts' ids: python3 rollout_texts.py ROLLOUT... | ling-tokenize TOKENIZER_JSON > ids.jsonl
"""
import array, json, os, struct, sys

MAGIC = 0x54524545
PMAX = int(os.environ.get("PMAX", 128))


def dump_streams(path):
    data = open(path, "rb").read()
    off, streams, cur, last_end = 0, [], [], None
    while off + 36 <= len(data):
        magic, pos, anchor, E, K16, K, accepted, nxt, greedy = struct.unpack_from("<9i", data, off)
        if magic != MAGIC:
            raise SystemExit(f"{path}: bad magic at {off}")
        off += 36
        off += 4 * E * K16 + 4 * E * K16 * K16
        rows = array.array("i", data[off:off + 4 * (E + 1)]); off += 4 * (E + 1)
        off += 8 * (E + 1) * K
        if last_end is None or pos != last_end:
            if cur: streams.append(cur)
            cur = []
        toks = list(rows[1:1 + accepted]) + [nxt]
        cur.extend(toks)
        last_end = pos + accepted + 1
    if cur: streams.append(cur)
    return streams


def stats(tokens):
    """worst[P] = the largest number of times a block of P tokens repeats back to back (ending at some position);
    run = the longest run of one token."""
    c = [0] * (PMAX + 1)
    worst = [1] * (PMAX + 1)
    run = best_run = 1 if tokens else 0
    for i, t in enumerate(tokens):
        if i and t == tokens[i - 1]:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 1
        for P in range(1, PMAX + 1):
            if i >= P and t == tokens[i - P]:
                c[P] += 1
                worst[P] = max(worst[P], c[P] // P + 1)
            else:
                c[P] = 0
    return worst, best_run


def main():
    kind, files = sys.argv[1], sys.argv[2:]
    streams = []
    for f in files:
        if kind == "dump":
            streams += dump_streams(f)
        else:
            streams += [json.loads(l) for l in open(f) if l.strip()]
    total = sum(len(s) for s in streams)
    print(f"{len(streams)} streams, {total} tokens")
    agg = [1] * (PMAX + 1)
    runs = []
    for s in streams:
        w, r = stats(s)
        runs.append(r)
        for P in range(1, PMAX + 1):
            agg[P] = max(agg[P], w[P])
    print("longest single-token runs:", sorted(runs)[-8:])
    print("worst repeat count per period P (P: count):")
    print(" ".join(f"{P}:{agg[P]}" for P in range(1, PMAX + 1) if agg[P] > 1))


if __name__ == "__main__":
    main()
