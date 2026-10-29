"""The merge gate for --fair-pairing (C3): whether fair pairing may be switched on, decided on a set of runs in which
fair-on and fair-off servers alternate (ABAB...) on one or more machines, so that noise cannot decide it.

(a) Fairness, exact, from each fair-on run's step trace (LING_STEP_TRACE; serve/step_trace.hpp, decode lines with
    `cand`). Per trace:
      complete  the decode lines' pass numbers run 1, 2, 3, ... (one truncated last line is allowed: the file is
                flushed when the server idles and at exit)
      replay    the trailing windows rebuilt from the decode lines give every candidate's recent and last
      slowest   every batched pass holds a request with the fewest recent passes among the batchable ones (a request
                skipped because its KV pages did not fit is not batchable for that pass; such passes are counted)
      policy    the partners are the first batchable requests in pair_order (serve/pairing.hpp), and a batchable
                request steps alone only when no other request is batchable
    The shares follow from the policy: this file's port of pair_order, run as tests/pairing_tests.cpp runs it, must
    give 200/200/200/200 passes in 400 turns with four decoders and 200/200/200 with three and one prefilling, and
    the policy check ties every pass of the binary to that port. The pass and token shares observed while three or
    more requests were batchable are printed for information only: the turn order (a finishing side chunk takes a
    turn) moves them, and (b) measures what they are worth.
(b) Speed, from the alternated run set: per run the slowest stream's decode tok/s (the lowest of streams_server.py's
    stream_decode_tps) and the combined decode tok/s (its decode_tps_pooled); per arm the median and a bootstrap 95%
    interval; for the difference (fair on - fair off) the difference of the arms' medians, its interval from
    resampling the on/off pairs within each machine. The noise band is the median absolute difference between
    successive runs of the same arm on the same machine, both arms pooled.

Pass: every run's output is identical to the reference turn for turn (greedy); (a) holds in every fair-on trace;
the slowest stream's difference has its interval above 0; the combined difference has its interval's lower end no
lower than minus the noise band; and the set holds at least --min-pairs pairs. Exit 0 on pass, 1 on fail (a bad
input fails with its reason), after a verdict of exactly 10 lines.

The run set: SET/MACHINE/N-LABEL.jsonl (n_streams.py --out), N-LABEL.serve.log (ling-serve's log, whose
"pairing:" line gives the arm) and N-LABEL.trace (the step trace; needed for fair-on runs), N the run's order on
that machine. Runs pair up in order on each machine (1 with 2, 3 with 4, ...), one on and one off in each pair; a
last run without its partner is left out. A set with no subdirectories is one machine.

Usage:
  fair_gate.py SET --reference REF.jsonl [--min-pairs N] [--boot N] [--seed S] [--json OUT]
  fair_gate.py --trace TRACE [TRACE...] [--legacy]     (a) alone, one line per trace
"""
import argparse
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAIR_WINDOW = 16  # serve/pairing.hpp kFairWindow
MIN_PAIRS = 12
BATCHABLE, SKIPPED_KV = 1, 2
PAIRING = re.compile(r"^pairing: (fair|running order)")


class GateError(Exception):
    """A run set the gate cannot judge (the reason goes into the verdict)."""


# ---- serve/pairing.hpp, ported ----


def window_count(marks, now):
    """PassWindow::count: the passes in `marks` within the trailing window ending at pass `now`."""
    return sum(1 for m in marks if m > now - FAIR_WINDOW)


def pair_order(cands, fair):
    """pair_order: running order, or (fair) the fewest recent passes first, then the oldest last pass, then running
    order. `cands` holds (recent, last) per running request."""
    order = list(range(len(cands)))
    if fair:
        order.sort(key=lambda i: (cands[i][0], cands[i][1]))  # stable, as std::stable_sort
    return order


def simulate(kinds, warm, turns, fair, fit=2):
    """tests/pairing_tests.cpp's Sim: the turn goes round the requests (kinds: B batchable, P a plain decoder that
    steps alone, F prefilling), a batchable one steps with the partners pair_order gives. Returns the passes per
    request over the counted turns, whether every batched pass held a request with the fewest recent passes, and
    the trace lines the run would write (decode lines with `cand`, as serve/step_trace.hpp formats them)."""
    marks = [[] for _ in kinds]
    passes = [0] * len(kinds)
    served, lines, p = True, [], 0
    for t in range(warm + turns):
        k = t % len(kinds)
        c = [(window_count(marks[i], p), marks[i][-1] if marks[i] else 0) for i in range(len(kinds))]
        ok = [kind == "B" for kind in kinds]
        b = []
        if ok[k]:
            b = [k] + [o for o in pair_order(c, fair) if o != k and ok[o]][: fit - 1]
        elif kinds[k] == "P":
            b = [k]
        if not b:
            lines.append({"pass": p, "kind": "chunk", "turn": k, "sessions": [k], "tokens": [0], "left": [100]})
            continue
        if len(b) >= 2 and min(c[i][0] for i in b) != min(c[i][0] for i in range(len(kinds)) if ok[i]):
            served = False
        cand = [[i, i + 1, int(ok[i]), c[i][0], c[i][1]] for i in range(len(kinds))]
        p += 1
        lines.append({"pass": p, "kind": "decode", "turn": k, "sessions": b, "tokens": [3] * len(b),
                      "left": [0] * len(b), "cand": cand})
        for i in b:
            marks[i].append(p)
            if t >= warm:
                passes[i] += 1
    return passes, served, lines


def port_matches_simulation():
    """Whether the port gives tests/pairing_tests.cpp's shares: 200/200/200/200 with four decoders and 200/200/200
    with three and one prefilling (each within one pass), legacy 4/2/1/1 and 3/2/1, the slowest in every pass."""
    four, s4, _ = simulate("BBBB", 40, 400, True)
    three, s3, _ = simulate("BBBF", 40, 400, True)
    even = all(abs(x - 200) <= 1 for x in four) and all(abs(x - 200) <= 1 for x in three[:3]) and three[3] == 0
    legacy = simulate("BBBB", 40, 40, False)[0] == [40, 20, 10, 10] and simulate("BBBF", 40, 40, False)[0] == [
        30, 20, 10, 0]
    return even and legacy and s4 and s3, four, three


# ---- (a): one step trace ----


def read_trace(path):
    """The trace's lines as dicts; one truncated last line is dropped (the file is flushed only at idle and exit)."""
    with open(path, errors="replace") as f:
        raw = [ln for ln in f.read().split("\n") if ln.strip()]
    lines = []
    for i, ln in enumerate(raw):
        try:
            lines.append(json.loads(ln))
        except ValueError:
            if i != len(raw) - 1:
                raise GateError(f"{os.path.basename(path)}: line {i + 1} is not JSON") from None
    return lines


def check_trace(lines, fair=True):
    """Checks one trace pass by pass. Returns a dict: problems (the first few, empty when it passes), passes, batched,
    kv_skips (batched passes with a request skipped for its KV pages), shares (per request id while three or more
    were batchable: passes and tokens)."""
    problems = []
    res = {"passes": 0, "batched": 0, "kv_skips": 0, "problems": problems, "shares": {}}

    def bad(msg):
        if len(problems) < 5:
            problems.append(msg)
        res["failed"] = res.get("failed", 0) + 1

    decode = [ln for ln in lines if ln.get("kind") == "decode"]
    if not decode:
        bad("no decode line")
        return res
    if any("cand" not in ln for ln in decode):
        bad("the trace lacks candidates (a build before the `cand` field)")
        return res
    marks = {}
    shares = res["shares"]
    seen = 0
    for ln in lines:
        p = ln["pass"]
        if ln.get("kind") != "decode":
            if p != seen:
                bad(f"a {ln.get('kind')} line at pass {p} after decode pass {seen}")
            continue
        if p != seen + 1:
            bad(f"decode pass {p} follows pass {seen}: lines are missing")
        seen = p
        now = p - 1
        cand, turn, sess = ln["cand"], ln["turn"], ln["sessions"]
        res["passes"] += 1
        for _session, req, _state, recent, last in cand:
            m = marks.get(req, [])
            want = (window_count(m, now), m[-1] if m else 0)
            if (recent, last) != want:
                bad(f"pass {p}: request {req} has recent/last {recent}/{last}, the replay {want[0]}/{want[1]}")
        if turn >= len(cand) or cand[turn][0] != sess[0]:
            bad(f"pass {p}: the request whose turn it was is not the unit's first")
            continue
        by_session = {c[0]: i for i, c in enumerate(cand)}
        if any(s not in by_session for s in sess):
            bad(f"pass {p}: a session in the unit is not running")
            continue
        members = [by_session[s] for s in sess]
        ok = [i for i, c in enumerate(cand) if c[2] == BATCHABLE]
        order = pair_order([(c[3], c[4]) for c in cand], fair)
        eligible = [i for i in order if i != turn and cand[i][2] == BATCHABLE]
        if len(members) >= 2:
            res["batched"] += 1
            if any(c[2] == SKIPPED_KV for c in cand):
                res["kv_skips"] += 1
            if cand[turn][2] != BATCHABLE:
                bad(f"pass {p}: a batch formed for a request that was not batchable")
            elif members[1:] != eligible[: len(members) - 1]:
                bad(f"pass {p}: partners {sess[1:]}, the policy gives "
                    f"{[cand[i][0] for i in eligible[: len(members) - 1]]}")
            elif fair and min(cand[i][3] for i in members) != min(cand[i][3] for i in ok):
                bad(f"pass {p}: no request with the fewest recent passes ({min(cand[i][3] for i in ok)}) in it")
        elif cand[turn][2] == BATCHABLE and eligible:
            bad(f"pass {p}: a batchable request stepped alone beside a batchable partner")
        if len(ok) >= 3:
            for i, tok in zip(members, ln.get("tokens", [0] * len(members)), strict=False):
                e = shares.setdefault(cand[i][1], [0, 0])
                e[0] += 1
                e[1] += tok
        for i in members:
            marks.setdefault(cand[i][1], []).append(p)
    return res


def trace_line(path, res):
    """One line for a checked trace."""
    sh = sorted(v[0] for v in res["shares"].values())
    spread = f"; passes while >= 3 batchable {min(sh)}..{max(sh)} per request" if sh else ""
    verdict = "pass" if not res["problems"] else f"FAIL ({res.get('failed', 0)}: {res['problems'][0]})"
    return (f"{os.path.basename(path)}: {verdict}; {res['passes']} decode passes, {res['batched']} batched, "
            f"{res['kv_skips']} with a KV skip{spread}")


# ---- (b): the alternated run set ----


def arm_of(serve_log):
    """'on' or 'off' from ling-serve's "pairing:" line."""
    with open(serve_log, errors="replace") as f:
        for ln in f:
            m = PAIRING.match(ln)
            if m:
                return "on" if m.group(1) == "fair" else "off"
    raise GateError(f"{os.path.basename(serve_log)} has no pairing line (a build before it)")


def discover(root):
    """The runs per machine, in order: {machine: [run dict, ...]}."""
    if not os.path.isdir(root):
        raise GateError(f"no run set at {root}")
    subs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    machines = {d: os.path.join(root, d) for d in subs} or {".": root}
    out = {}
    for name, d in machines.items():
        runs = []
        for f in sorted(os.listdir(d)):
            if not f.endswith(".jsonl"):
                continue
            stem = f[: -len(".jsonl")]
            m = re.match(r"(\d+)", stem)
            if not m:
                raise GateError(f"{name}/{f}: the name does not start with the run's order number")
            log, trace = os.path.join(d, stem + ".serve.log"), os.path.join(d, stem + ".trace")
            if not os.path.exists(log):
                raise GateError(f"{name}/{stem}: no serve.log")
            runs.append({"machine": name, "stem": stem, "seq": int(m.group(1)), "jsonl": os.path.join(d, f),
                         "log": log, "trace": trace if os.path.exists(trace) else None})
        runs.sort(key=lambda r: r["seq"])
        if len({r["seq"] for r in runs}) != len(runs):
            raise GateError(f"{name}: two runs with one order number")
        if runs:
            out[name] = runs
    if not out:
        raise GateError(f"no runs under {root}")
    return out


def measure(run, reference):
    """Fills a run's arm, identity against the reference, slowest and combined decode tok/s."""
    import n_streams
    import streams_server

    run["arm"] = arm_of(run["log"])
    reqs, _ = n_streams.load(run["jsonl"])
    got = {(r["id"], r["rep"]): r.get("text_sha256") for r in reqs}
    same = sum(1 for k, h in reference.items() if got.get(k) is not None and got.get(k) == h)
    run["identity"] = (same, len(reference))
    rep = streams_server.report(run["jsonl"], run["log"])
    per = rep.get("stream_decode_tps") or []
    if rep.get("matched") != rep.get("requests") or not per or any(x is None for x in per):
        raise GateError(f"{run['machine']}/{run['stem']}: {rep.get('matched')} of {rep.get('requests')} requests "
                        "matched in the server log, or a stream without decode")
    run["slowest"], run["combined"] = min(per), rep["decode_tps_pooled"]


def pair_up(machines):
    """The on/off pairs per machine, in order; a last unpaired run is left out (returned apart)."""
    pairs, left = {}, []
    for name, runs in machines.items():
        ps = []
        for i in range(0, len(runs) - 1, 2):
            a, b = runs[i], runs[i + 1]
            if {a["arm"], b["arm"]} != {"on", "off"}:
                raise GateError(f"{name}: runs {a['stem']} and {b['stem']} are both fair-{a['arm']} (not alternated)")
            ps.append((a, b) if a["arm"] == "on" else (b, a))
        if len(runs) % 2:
            left.append(runs[-1])
        if ps:
            pairs[name] = ps
    return pairs, left


def median_diff(pairs, key):
    """Median over the on runs minus median over the off runs."""
    return statistics.median(on[key] for on, _ in pairs) - statistics.median(off[key] for _, off in pairs)


def bootstrap(pairs_by_machine, stat, boot, rng):
    """95% percentile interval of stat(pairs), resampling the pairs within each machine."""
    groups = list(pairs_by_machine.values())
    vals = []
    for _ in range(boot):
        sample = []
        for g in groups:
            sample.extend(g[rng.randrange(len(g))] for _ in g)
        vals.append(stat(sample))
    vals.sort()
    return vals[int(0.025 * (boot - 1))], vals[int(0.975 * (boot - 1))]


def noise_band(machines, key):
    """Median absolute difference between successive runs of the same arm on the same machine."""
    diffs = []
    for runs in machines.values():
        for arm in ("on", "off"):
            xs = [r[key] for r in runs if r.get("arm") == arm and key in r]
            diffs.extend(abs(b - a) for a, b in zip(xs, xs[1:], strict=False))
    return statistics.median(diffs) if diffs else None


def unjudged(why, reasons):
    """The 10 verdict lines of a set that could not be measured."""
    lines = [f"fair gate: FAIL ({why})"]
    lines += [f"{what}: not judged" for what in ("identity", "fairness (a)", "policy port", "slowest stream tok/s",
                                                 "slowest on - off", "combined tok/s", "combined on - off",
                                                 "noise bands")]
    return lines + ["reasons: " + "; ".join(reasons)]


def judge(root, reference_path, min_pairs=MIN_PAIRS, boot=10000, seed=1):
    """Runs the whole gate. Returns (passed, the 10 verdict lines, details)."""
    reasons, d = [], {"set": os.path.basename(os.path.normpath(root))}
    port_ok, four, three = port_matches_simulation()
    d["port"] = {"ok": port_ok, "four": four, "three": three}
    if not port_ok:
        reasons.append("the policy port does not reproduce the simulation")
    try:
        import n_streams

        ref = {(r["id"], r["rep"]): r.get("text_sha256") for r in n_streams.load(reference_path)[0]}
        if not ref:
            raise GateError("the reference has no turns")
        machines = discover(root)
        runs = [r for rs in machines.values() for r in rs]
        for r in runs:
            measure(r, ref)
        pairs, left = pair_up(machines)
    except GateError as e:
        reasons.append(str(e))
        return False, unjudged(f"the run set cannot be judged; set {d['set']}", reasons), d
    except Exception as e:  # a malformed file: a verdict with the reason, never a traceback
        reasons.append(f"{type(e).__name__}: {e}")
        return False, unjudged(f"the run set cannot be judged; set {d['set']}", reasons), d
    n_pairs = sum(len(p) for p in pairs.values())
    all_pairs = [p for ps in pairs.values() for p in ps]
    used = [r for p in all_pairs for r in p]
    ident_bad = [f"{r['machine']}/{r['stem']} {r['identity'][0]}/{r['identity'][1]}" for r in runs
                 if r["identity"][0] != r["identity"][1]]
    if ident_bad:
        reasons.append("not identical: " + ", ".join(ident_bad[:4]))
    on_runs = [r for r in used if r["arm"] == "on"]
    checked = passed_a = batched = kv = 0
    for r in on_runs:
        if r["trace"] is None:
            reasons.append(f"{r['machine']}/{r['stem']}: fair on without a trace")
            continue
        try:
            res = check_trace(read_trace(r["trace"]), fair=True)
        except GateError as e:
            reasons.append(str(e))
            continue
        except (KeyError, TypeError, ValueError, IndexError) as e:  # a malformed line
            reasons.append(f"{r['machine']}/{r['stem']}: the trace is malformed ({type(e).__name__}: {e})")
            continue
        checked += 1
        batched += res["batched"]
        kv += res["kv_skips"]
        if res["problems"]:
            reasons.append(f"{r['machine']}/{r['stem']}: {res['problems'][0]}")
        else:
            passed_a += 1
    rng = random.Random(seed)
    lines = [""] * 10
    if n_pairs < min_pairs:
        reasons.append(f"too few pairs ({n_pairs} < {min_pairs})")
    if n_pairs == 0:
        return False, unjudged(f"{len(runs)} runs, no on/off pair; set {d['set']}", reasons), d
    stats = {}
    for key in ("slowest", "combined"):
        arm = {}
        for a, idx in (("on", 0), ("off", 1)):
            xs = [p[idx][key] for p in all_pairs]
            per = {m: [(p[idx],) for p in ps] for m, ps in pairs.items()}
            ci = bootstrap(per, lambda s, k=key: statistics.median(r[0][k] for r in s), boot, rng)
            arm[a] = (statistics.median(xs), ci)
        diff = median_diff(all_pairs, key)
        dci = bootstrap(pairs, lambda s, k=key: median_diff(s, k), boot, rng)
        stats[key] = {"on": arm["on"], "off": arm["off"], "diff": diff, "diff_ci": dci,
                      "noise": noise_band({m: [r for p in ps for r in sorted(p, key=lambda r: r["seq"])]
                                           for m, ps in pairs.items()}, key)}
    d["stats"] = stats
    s, c = stats["slowest"], stats["combined"]
    slow_ok = s["diff_ci"][0] > 0
    band = c["noise"]
    comb_ok = band is not None and c["diff_ci"][0] >= -band
    if not slow_ok:
        reasons.append("the slowest stream's gain is not above noise (interval reaches 0)")
    if band is None:
        reasons.append("no noise band (each arm needs two runs on one machine)")
    elif not comb_ok:
        reasons.append("the combined speed may lose more than the noise band")
    ok = not reasons
    f = lambda v: f"{v[0]:.2f} [{v[1][0]:.2f}, {v[1][1]:.2f}]"  # noqa: E731
    lines[0] = (f"fair gate: {'PASS' if ok else 'FAIL'} ({len(used)} runs in {n_pairs} on/off pairs on "
                f"{len(pairs)} machine(s); set {d['set']}" + (f"; {len(left)} unpaired left out)" if left else ")"))
    lines[1] = (f"identity: {len(runs) - len(ident_bad)} of {len(runs)} runs identical to the reference "
                f"turn for turn ({len(ref)} turns)")
    lines[2] = (f"fairness (a): {passed_a} of {len(on_runs)} fair-on traces pass ({batched} batched passes checked, "
                f"{kv} with a KV skip)")
    lines[3] = (f"policy port: {'/'.join(map(str, four))} passes with four decoders, "
                f"{'/'.join(map(str, three[:3]))} with three and one prefilling ({'as' if port_ok else 'NOT as'} "
                "the simulation)")
    lines[4] = f"slowest stream tok/s, median and 95% interval: on {f(s['on'])}, off {f(s['off'])}"
    lines[5] = (f"slowest on - off: {s['diff']:+.2f} [{s['diff_ci'][0]:+.2f}, {s['diff_ci'][1]:+.2f}] "
                f"(needs the interval above 0): {'pass' if slow_ok else 'FAIL'}")
    lines[6] = f"combined tok/s, median and 95% interval: on {f(c['on'])}, off {f(c['off'])}"
    lines[7] = (f"combined on - off: {c['diff']:+.2f} [{c['diff_ci'][0]:+.2f}, {c['diff_ci'][1]:+.2f}] (needs the "
                f"lower end >= -{band if band is None else round(band, 2)}): {'pass' if comb_ok else 'FAIL'}")
    sn = s["noise"]
    lines[8] = (f"noise bands (median successive same-arm difference): combined "
                f"{'n/a' if band is None else f'{band:.2f}'}, slowest {'n/a' if sn is None else f'{sn:.2f}'} tok/s; "
                f"{n_pairs} pairs (minimum {min_pairs})")
    lines[9] = "reasons: " + ("none" if ok else "; ".join(reasons))
    return ok, lines, d


def main(argv=None):
    ap = argparse.ArgumentParser(description="the --fair-pairing merge gate")
    ap.add_argument("set", nargs="?", help="the run set (SET/MACHINE/N-LABEL.{jsonl,serve.log,trace})")
    ap.add_argument("--reference", help="a greedy n_streams.py run every run must match turn for turn")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", help="write the details here")
    ap.add_argument("--trace", nargs="+", help="check these step traces only, (a)")
    ap.add_argument("--legacy", action="store_true", help="with --trace: the traces are fair-off (running order)")
    a = ap.parse_args(argv)
    if a.trace:
        ok = True
        for t in a.trace:
            try:
                res = check_trace(read_trace(t), fair=not a.legacy)
            except (GateError, OSError, KeyError, TypeError, ValueError, IndexError) as e:
                print(f"{os.path.basename(t)}: FAIL ({e})")
                ok = False
                continue
            print(trace_line(t, res))
            ok = ok and not res["problems"]
        return 0 if ok else 1
    if not a.set or not a.reference:
        ap.error("a run set and --reference are needed (or --trace)")
    ok, lines, d = judge(a.set, a.reference, a.min_pairs, a.boot, a.seed)
    print("\n".join(lines))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"pass": ok, "verdict": lines, **{k: v for k, v in d.items() if k != "runs"}}, f, indent=1,
                      default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
