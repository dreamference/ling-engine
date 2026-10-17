"""Paired by request: tokens per pass of arm B over arm A on the engine's request lines (warm requests, c1, same order)."""
import random, re, sys
rx = re.compile(r"request: session (\d+)( \(new\))?, .*?; (\d+) tokens in [\d.]+ s .*?; (\d+) passes")
def rows(f, skip):
    out = []
    for l in open(f):
        if l.startswith("request:"):
            m = rx.search(l)
            if m: out.append((bool(m.group(2)), int(m.group(3)), int(m.group(4))))
    return out[skip:]
pairs = []
for a, b, skip in zip(sys.argv[1::3], sys.argv[2::3], map(int, sys.argv[3::3])):
    A, B = rows(a, skip), rows(b, skip)
    assert len(A) == len(B), (len(A), len(B))
    pairs += [(x, y) for x, y in zip(A, B) if not x[0] and x[2] > 0 and y[2] > 0]
def ratio(ps):
    ta = sum(x[1] for x, _ in ps) / sum(x[2] for x, _ in ps)
    tb = sum(y[1] for _, y in ps) / sum(y[2] for _, y in ps)
    return tb / ta, ta, tb
r, ta, tb = ratio(pairs)
rng = random.Random(1)
bs = sorted(ratio([pairs[rng.randrange(len(pairs))] for _ in pairs])[0] for _ in range(4000))
print(f"{len(pairs)} warm request pairs: A {ta:.3f} tokens/pass, B {tb:.3f}; B/A {r:.3f}, 95% bootstrap {bs[100]:.3f}-{bs[3899]:.3f}")
