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


STATE_BYTES = 157 << 20  # one DeltaNet and conv state (48 linear layers)
KV_BYTES_PER_TOKEN = 64 * 1024  # the target's full-attention KV per token: 16 layers x K and V x 4 heads x 256 x BF16


def simulate(prompts, rule, chunk, nsess, slots, limit, boundary):
    parts = set(rule.split("+"))
    branch, trim, near, inpass = "branch" in parts, "trim" in parts, "near" in parts, "inpass" in parts
    kv_mb = next((int(x[2:]) for x in parts if x.startswith("kv")), 0)
    kv_cap = kv_mb * (1 << 20) // KV_BYTES_PER_TOKEN  # the prefix KV store's budget in tokens
    sess = [dict(hist=[], snap=-1, used=-1, owned=False, turn_start=0) for _ in range(nsess)]
    store = []  # dict(pos, toks, used)
    kv = []     # the prefix KV store's paths (the radix tree's leaves): dict(toks, used)
    kvst = []   # the states the prefix KV store keeps at its paths' soft points: dict(pos, toks)
    clock, turn, out = 0, 0, []
    align = lambda p: p - p % ALIGN

    def kv_tokens(paths):  # the radix tree's size: every path's tokens past what earlier paths share with it
        n = 0
        for i, a in enumerate(paths):
            n += len(a["toks"]) - max([lcp(a["toks"], b["toks"]) for b in paths[:i]] + [0])
        return n

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
        window = max(0, L - 2048)  # the drafter's context starts here; the store holds no drafter ring

        def from_kv(l):
            r = 0
            for c in store:
                if r < c["pos"] <= l and c["pos"] < L and c["pos"] <= window and P[:c["pos"]] == c["toks"]:
                    r = c["pos"]
            return r

        def best_in(s):
            l = lcp(s["hist"], P)
            r = s["snap"] if 0 < s["snap"] <= l and s["snap"] < L else 0
            for c in store:
                if r < c["pos"] <= l and c["pos"] < L and P[:c["pos"]] == c["toks"]:
                    r = c["pos"]
            return r

        reuse = max(best_in(s) for s in sess)
        kv_l = max([lcp(a["toks"], P) for a in kv] + [0])
        reuse = max(reuse, from_kv(kv_l))
        for st in kvst:  # the store's own states, at its paths' soft points
            if reuse < st["pos"] <= kv_l and st["pos"] < L and st["pos"] <= window and P[:st["pos"]] == st["toks"]:
                reuse = st["pos"]
        for a in kv:
            if lcp(a["toks"], P) > 0:
                a["used"] = turn if lcp(a["toks"], P) == kv_l else a["used"]
        branch_at = 0
        if branch:
            branch_at = max([lcp(s["hist"], P) for s in sess if s["hist"] and not (s is cur and k > 0 and lcp(s["hist"], P) >= s["turn_start"])]
                            + [lcp(c["toks"], P) for c in store] + [kv_l, 0])
            branch_at = align(branch_at)
        clock += 1
        epoch = clock
        for c in store:
            if c["pos"] <= L and P[:c["pos"]] == c["toks"]:
                c["used"] = epoch
        hard, soft, msgs = set(), set(), []
        for i in range(min(L, limit)):
            if len(msgs) >= 2:
                break
            if P[i] == boundary:
                msgs.append(align(i + 1))
        (soft if inpass else hard).update(m for m in msgs if m > 0)
        hard.update(range(chunk, min(L, limit + 1), chunk))
        if branch and reuse < branch_at < L and branch_at <= limit:
            (soft if inpass else hard).add(branch_at)
        if near and branch_at in hard | soft:
            hard = {b for b in hard if not (branch_at < b < branch_at + chunk and b % chunk == 0)}
        if trim:
            keep_to = max([branch_at] + msgs)
            hard = {b for b in hard if not (b > keep_to and b % chunk == 0)}
        hard -= soft
        snap_at = align(L)
        i, passes, kept = reuse, 0, []
        hl, sl = sorted(hard), sorted(soft)
        while i < L:  # chunk_begin: the next hard bound ends the chunk; one state is captured inside the pass
            end = min(i + chunk, L)
            for b in hl:
                if i < b < end:
                    end = b
                    break
            caught = [b for b in sl if i < b < end]
            if i < snap_at < end:  # the snap takes the pass's capture
                if caught:
                    end = caught[0]
                    caught = []
            if len(caught) > 1:  # a second soft point in the chunk ends it
                end = caught[1]
            kept += [b for b in caught[:1] if b < end] + ([end] if end in hard or end in soft else [])
            i, passes = end, passes + 1
        for b in kept:
            if b <= reuse or b >= L:
                continue
            toks = P[:b]
            hit = [c for c in store if c["pos"] == b and c["toks"] == toks]
            if hit:
                clock += 1
                hit[0]["used"] = clock
                continue
            is_soft = b in soft
            if len(store) >= slots:
                cands = [c for c in store if c["used"] < epoch]
                if cands:
                    store.remove(min(cands, key=lambda c: c["used"]))
                elif inpass and is_soft and [c for c in store if not c["soft"]]:  # replaces its own furthest multiple
                    store.remove(max([c for c in store if not c["soft"]], key=lambda c: c["pos"]))
                else:
                    continue
            clock += 1
            store.append(dict(pos=b, toks=toks, used=clock, soft=is_soft))
        if kv_cap > 0:  # insert the shared region [0, q) and evict least recently used leaves past the budget
            q = align(max([branch_at] + msgs))
            if q > 0 and not any(len(a["toks"]) >= q and a["toks"][:q] == P[:q] for a in kv):
                kv = [a for a in kv if not (len(a["toks"]) < q and P[:len(a["toks"])] == a["toks"])]
                kv.append(dict(toks=P[:q], used=turn))
                for b in sorted(set(m for m in msgs if 0 < m <= q) | ({branch_at} if 0 < branch_at <= q else set())):
                    in_store = any(c["pos"] == b and c["toks"] == P[:b] for c in store)
                    if in_store and not any(st["pos"] == b and st["toks"] == P[:b] for st in kvst):
                        kvst.append(dict(pos=b, toks=P[:b]))
            state_tokens = STATE_BYTES // KV_BYTES_PER_TOKEN
            while kv and kv_tokens(kv) + state_tokens * len(kvst) > kv_cap:
                kv.remove(min(kv, key=lambda a: a["used"]))
                kvst = [st for st in kvst if any(len(a["toks"]) >= st["pos"] and a["toks"][:st["pos"]] == st["toks"] for a in kv)]
        cur["snap"] = snap_at
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
