"""Simulates ling-serve's prefix reuse on a replay set: the session pool and the checkpoint store (BACKLOG 34, lever 4).

For each prompt in turn (after one short unrelated warm-up prompt, as run.py sends) it applies the engine's rules
(engine.cpp: match_session, free_session, begin_prefill's best_in, save_checkpoint's LRU with the prompt's own entries
kept, the chunk ends of prefill_chunk_run) and prints the tokens each prompt reuses, the tokens it prefills and the chunk
passes it takes. On agentlong's sixteen turns with the defaults it reproduces the cached column ling-serve reports turn
for turn. Decoded tokens are never shared between the turns, so they are stand-ins.

Usage:
  ling-template --responses TOKENIZER_JSON < bodies.jsonl > ids.jsonl
  ckpt_sim.py ids.jsonl [--chunk 2048] [--sessions 2] [--slots 8] [--limit 32768] [--rules base,branch,branch+trim,branch+near]

Rules: base (main); branch (a checkpoint where the prompt leaves the longest prefix it shares with a resident history
or a stored entry, LING_BRANCH_CKPT); +trim (no chunk-multiple checkpoint past both the branch point and the first two
message ends, LING_CKPT_TRIM); +near (no engine switch: the chunk multiple right after the branch point gives way to it).
"""
import argparse
import json

ALIGN = 32  # kResumeAlign


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def simulate(prompts, rule, chunk, nsess, slots, limit, boundary):
    branch, trim, near = rule.startswith("branch"), rule.endswith("+trim"), rule.endswith("+near")
    sess = [dict(hist=[], snap=-1, used=-1, owned=False, turn_start=0) for _ in range(nsess)]
    store = []  # dict(pos, toks, used)
    clock, turn, out = 0, 0, []
    align = lambda p: p - p % ALIGN
    for k, P in enumerate([[-5] * 60 + [boundary] + [-6] * 40] + prompts):
        L = len(P)
        turn += 1
        best, best_l = -1, 0  # match_session: a session the prompt extends past its last turn's start
        for i, s in enumerate(sess):
            if s["owned"] and s["hist"]:
                l = lcp(s["hist"], P)
                if l >= s["turn_start"] and l > best_l:
                    best, best_l = i, l
        if best < 0:
            free = [i for i, s in enumerate(sess) if not s["owned"]]
            best = free[0] if free else min(range(nsess), key=lambda i: sess[i]["used"])
            sess[best]["owned"] = True
        cur = sess[best]
        cur["used"] = turn

        def best_in(s):
            l = lcp(s["hist"], P)
            r = s["snap"] if 0 < s["snap"] <= l and s["snap"] < L else 0
            for c in store:
                if r < c["pos"] <= l and c["pos"] < L and P[:c["pos"]] == c["toks"]:
                    r = c["pos"]
            return r

        reuse = max(best_in(s) for s in sess)
        branch_at = 0
        if branch:
            branch_at = max([lcp(s["hist"], P) for s in sess if s["hist"] and not (s is cur and k > 0 and lcp(s["hist"], P) >= s["turn_start"])]
                            + [lcp(c["toks"], P) for c in store] + [0])
            branch_at = align(branch_at)
        clock += 1
        epoch = clock
        for c in store:
            if c["pos"] <= L and P[:c["pos"]] == c["toks"]:
                c["used"] = epoch
        bounds, msgs = set(), []
        for i in range(min(L, limit)):
            if len(msgs) >= 2:
                break
            if P[i] == boundary:
                msgs.append(align(i + 1))
        bounds.update(m for m in msgs if m > 0)
        bounds.update(range(chunk, min(L, limit + 1), chunk))
        if branch and reuse < branch_at < L and branch_at <= limit:
            bounds.add(branch_at)
        if near and branch_at in bounds:  # the chunk multiple right after the branch point gives way to it
            bounds = {b for b in bounds if not (branch_at < b < branch_at + chunk and b % chunk == 0)}
        if trim:
            keep_to = max([branch_at] + msgs)
            bounds = {b for b in bounds if not (b > keep_to and b % chunk == 0)}
        bl = sorted(bounds)
        i, passes = reuse, 0
        while i < L:
            end = min(i + chunk, L)
            for b in bl:
                if i < b < end:
                    end = b
                    break
            i, passes = end, passes + 1
        for b in bl:
            if b <= reuse or b >= L:
                continue
            toks = P[:b]
            hit = [c for c in store if c["pos"] == b and c["toks"] == toks]
            if hit:
                clock += 1
                hit[0]["used"] = clock
                continue
            if len(store) >= slots:
                cands = [c for c in store if c["used"] < epoch]
                if not cands:
                    continue
                store.remove(min(cands, key=lambda c: c["used"]))
            clock += 1
            store.append(dict(pos=b, toks=toks, used=clock))
        cur["snap"] = align(L)
        last = max([i + 1 for i in range(L) if P[i] == boundary] + [0])
        cur["turn_start"] = last or L
        cur["hist"] = list(P) + [-9] * 300
        if k > 0:
            out.append((L, reuse, passes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--limit", type=int, default=32768)
    ap.add_argument("--boundary", type=int, default=248046, help="<|im_end|>")
    ap.add_argument("--rules", default="base,branch,branch+trim,branch+near")
    a = ap.parse_args()
    prompts = [json.loads(line)["ids"] for line in open(a.ids)]
    for rule in a.rules.split(","):
        o = simulate(prompts, rule, a.chunk, a.sessions, a.slots, a.limit, a.boundary)
        print(f"{rule:13s} new {sum(L - r for L, r, _ in o):7d} passes {sum(p for *_, p in o):4d} "
              f"reused {[r for _, r, _ in o]}")


if __name__ == "__main__":
    main()
