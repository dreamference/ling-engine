"""Follow-up 2b, the two-bucket decision: a classifier on step features (learned_step_sim.py's) predicts whether the
chain alone (12 rows) gives more tokens per ms than the 24-row tree on this step; the step takes the chain when the
predicted probability passes a threshold. Held out by request; greedy steps scored; also the per-node AUC check.
Usage: learned_pair_sim.py GREEDY_DUMP [...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tree_sim import read_steps, with_truth, build, accept  # noqa: E402
from adaptive_sim import BUCKETS, attn_extra, probsum  # noqa: E402
from learned_step_sim import step_features  # noqa: E402

COSTS = {12: 100.0, 16: 105.1, 20: 105.5, 24: 107.0, 32: 118.0}


def main():
    steps = []
    for p in sys.argv[1:]:
        steps += read_steps(p)
    reqs = [[s for s in r if len(s["truth"]) == s["E"] and s["greedy"]] for r in with_truth(steps)]
    reqs = [r for r in reqs if r]
    rows = []
    for i, r in enumerate(reqs):
        for s in r:
            res, ps = {}, []
            for b in (12, 24):
                t = build(s, BUCKETS[b], 1.5)
                res[b] = (accept(s, t) + 1, COSTS[b] + attn_extra(b, s["pos"]))
                ps.append(probsum(s, t, 1.5))
            rows.append((i, step_features(s) + ps, res))
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    lab = np.array([float(r[2][12][0] / r[2][12][1] > r[2][24][0] / r[2][24][1]) for r in rows])
    print(f"{len(rows)} steps; the chain alone better on {lab.mean():.1%} of them")
    for name in ("logreg", "gbc"):
        prob = np.zeros(len(rows))
        for hold in range(len(reqs)):
            tr = [j for j, r in enumerate(rows) if r[0] != hold]
            te = [j for j, r in enumerate(rows) if r[0] == hold]
            X = np.array([rows[j][1] for j in tr])
            sc = StandardScaler().fit(X)
            m = (LogisticRegression(max_iter=3000) if name == "logreg" else
                 GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0))
            m.fit(sc.transform(X), lab[tr])
            prob[te] = m.predict_proba(sc.transform(np.array([rows[j][1] for j in te])))[:, 1]
        auc = roc_auc_score(lab, prob)
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            tk = sum(r[2][12 if p > th else 24][0] for r, p in zip(rows, prob))
            ms = sum(r[2][12 if p > th else 24][1] for r, p in zip(rows, prob))
            print(f"{name} (AUC {auc:.3f}) threshold {th}: {1000 * tk / ms:.2f} tokens/s, chain on {(prob > th).mean():.1%}")
    t = sum(r[2][24][0] for r in rows); m = sum(r[2][24][1] for r in rows)
    print(f"fixed 24: {1000 * t / m:.2f} tokens/s")


if __name__ == "__main__":
    main()
