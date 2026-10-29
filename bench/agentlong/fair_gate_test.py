#!/usr/bin/env python3
"""Tests for fair_gate.py (host only, no GPU, no server): the port of the pairing policy, the trace checks on
traces the port writes and on broken ones, the trace line tests/pairing_tests.cpp formats, and the run-set verdict
on synthetic alternated sets.

    python3 bench/agentlong/fair_gate_test.py
"""
import contextlib
import copy
import io
import json
import os
import random
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fair_gate as g  # noqa: E402

PAIRING_TESTS = os.path.join(HERE, "..", "..", "tests", "pairing_tests.cpp")


def trace_text(lines):
    return "\n".join(json.dumps(ln) for ln in lines) + "\n"


class Port(unittest.TestCase):
    def test_simulation_shares(self):
        ok, four, three = g.port_matches_simulation()
        self.assertTrue(ok)
        self.assertEqual(four, [200, 200, 200, 200])
        self.assertEqual(three, [200, 200, 200, 0])

    def test_legacy_shares(self):
        self.assertEqual(g.simulate("BBBB", 40, 40, False)[0], [40, 20, 10, 10])
        self.assertEqual(g.simulate("BBBF", 40, 40, False)[0], [30, 20, 10, 0])

    def test_pair_order(self):
        # tests/pairing_tests.cpp ties_and_turn: fewest, then oldest, then running order
        self.assertEqual(g.pair_order([(2, 9), (0, 0), (2, 9), (0, 0), (2, 7)], True), [1, 3, 4, 0, 2])
        self.assertEqual(g.pair_order([(3, 0), (1, 0)], False), [0, 1])

    def test_window(self):
        marks = list(range(1, 41, 2))
        self.assertEqual(g.window_count(marks, 40), 8)
        self.assertEqual(g.window_count(marks, 60), 0)


class Trace(unittest.TestCase):
    def fair(self, kinds="BBBB"):
        return copy.deepcopy(g.simulate(kinds, 0, 200, True)[2])

    def test_fair_traces_pass(self):
        for kinds in ("BBBB", "BBBF", "BPBFB", "BB", "BBBBB"):
            res = g.check_trace(self.fair(kinds), fair=True)
            self.assertEqual(res["problems"], [], kinds)
            self.assertGreater(res["batched"], 0)

    def test_legacy_trace_fails_fair_and_passes_legacy(self):
        lines = g.simulate("BBBB", 0, 200, False)[2]
        self.assertTrue(g.check_trace(lines, fair=True)["problems"])
        self.assertEqual(g.check_trace(lines, fair=False)["problems"], [])

    def test_three_decoders_legacy_breaks_the_invariant(self):
        lines = g.simulate("BBBF", 0, 200, False)[2]
        res = g.check_trace(lines, fair=True)
        self.assertTrue(any("partners" in p or "fewest" in p for p in res["problems"]))

    def batched_index(self, lines, at=50):
        return next(i for i, ln in enumerate(lines) if i >= at and ln["kind"] == "decode" and len(ln["sessions"]) == 2)

    def test_swapped_partner_fails(self):
        lines = self.fair()
        i = self.batched_index(lines)
        ln = lines[i]
        others = [c[0] for c in ln["cand"] if c[0] not in ln["sessions"]]
        ln["sessions"][1] = others[0]
        self.assertTrue(g.check_trace(lines, fair=True)["problems"])

    def test_dropped_decode_line_fails(self):
        lines = self.fair()
        del lines[self.batched_index(lines)]
        probs = g.check_trace(lines, fair=True)["problems"]
        self.assertTrue(any("missing" in p for p in probs))

    def test_wrong_recent_fails_the_replay(self):
        lines = self.fair()
        lines[self.batched_index(lines)]["cand"][0][3] += 1
        self.assertTrue(any("replay" in p for p in g.check_trace(lines, fair=True)["problems"]))

    def test_batch_for_an_unbatchable_request_fails(self):
        lines = self.fair()
        ln = lines[self.batched_index(lines)]
        ln["cand"][ln["turn"]][2] = 0
        self.assertTrue(g.check_trace(lines, fair=True)["problems"])

    def test_alone_beside_a_partner_fails(self):
        lines = self.fair()
        ln = lines[self.batched_index(lines)]
        ln["sessions"] = ln["sessions"][:1]
        ln["tokens"] = ln["tokens"][:1]
        self.assertTrue(any("alone" in p for p in g.check_trace(lines, fair=True)["problems"]))

    def test_kv_skip_is_not_a_violation(self):
        # The fewest-passes partner was skipped for its KV pages: the next one joins, and the pass is counted.
        lines = self.fair()
        i = self.batched_index(lines)
        ln = lines[i]
        turn = ln["turn"]
        partner = next(j for j, c in enumerate(ln["cand"]) if c[0] == ln["sessions"][1])
        order = g.pair_order([(c[3], c[4]) for c in ln["cand"]], True)
        nxt = [j for j in order if j not in (turn, partner) and ln["cand"][j][2] == 1]
        ln["cand"][partner][2] = 2
        ln["sessions"][1] = ln["cand"][nxt[0]][0]
        del lines[i + 1:]  # the replay after it would differ
        res = g.check_trace(lines, fair=True)
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["kv_skips"], 1)

    def test_no_candidates_is_an_older_build(self):
        lines = self.fair()
        for ln in lines:
            ln.pop("cand", None)
        self.assertIn("lacks candidates", g.check_trace(lines, fair=True)["problems"][0])

    def test_truncated_last_line_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.trace")
            text = trace_text(self.fair())
            with open(p, "w") as f:
                f.write(text[:-20])
            self.assertEqual(g.check_trace(g.read_trace(p), fair=True)["problems"], [])
            with open(p, "w") as f:
                f.write(text[:100] + "\n" + text[100:])
            with self.assertRaises(g.GateError):
                g.read_trace(p)

    def test_cpp_trace_line_parses(self):
        # The literal tests/pairing_tests.cpp checks format_step against, so the two cannot drift apart.
        with open(PAIRING_TESTS) as f:
            src = f.read()
        m = re.search(r"want_cand =  // trace line with cand\n((?:\s*\".*\"\n?)+);", src)
        self.assertIsNotNone(m)
        line = "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", m.group(1))).replace('\\"', '"')
        rec = json.loads(line)
        self.assertEqual(rec["kind"], "decode")
        self.assertEqual(rec["cand"][0], [3, 11, 2, 1, 40])  # session, request, state (skipped for KV), recent, last
        self.assertEqual(rec["cand"][rec["turn"]][0], rec["sessions"][0])
        first = dict(rec, **{"pass": 1, "cand": [[c[0], c[1], c[2], 0, 0] for c in rec["cand"]]})
        self.assertEqual(g.check_trace([first], fair=True)["problems"], [])


def write_run(d, stem, arm, slow, comb, rng, identical=True, trace=True, pairing=True):
    """One synthetic run: 16 turns over 4 streams, stream 0 at `slow` tok/s, the others set so the pooled rate is
    `comb`; the server log in ling-serve's request-line format, a fair or legacy trace from the port."""
    n, out = 4, 101
    tokens_per_stream = 4 * (out - 1)
    others = (3 * tokens_per_stream) / (4 * tokens_per_stream / comb - tokens_per_stream / slow)
    rate = [slow] + [others] * 3
    recs, logs, t = [], [], 1000.0
    for i in range(16):
        s = i % n
        dec = (out - 1) / rate[s]
        sha = f"turn{i}" if identical or i != 5 else "changed"
        recs.append({"kind": "request", "id": f"t{i:02d}", "rep": 0, "stream": s, "t0": t, "t_first": t + 1.0,
                     "t_end": t + 1.0 + dec, "out": out, "prompt": 30000 + i, "cached": 29000, "text_sha256": sha})
        logs.append(f"request: session {s}, {30000 + i} prompt tokens (29000 reused), prefill 1.00 s (1 chunks, 0 "
                    f"beside a decode, 1000 tok/s of compute); {out} tokens in {dec:.3f} s ({rate[s]:.1f} tok/s), "
                    f"stop; 20 passes (20 batched), 5.00 tokens/pass; step period 100.0 ms mean, 200.0 max over 10; "
                    f"beside a prefill 150.0 ms mean, 300.0 max over 10")
        t += 0.5
    with open(os.path.join(d, stem + ".jsonl"), "w") as f:
        f.write(json.dumps({"header": True, "label": stem, "mode": "greedy", "streams": n}) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": True, "label": stem, "streams": n, "mode": "greedy"}) + "\n")
    with open(os.path.join(d, stem + ".serve.log"), "w") as f:
        if pairing:
            f.write("pairing: " + ("fair (--fair-pairing 1)" if arm == "on" else "running order (--fair-pairing 0)")
                    + "\n")
        f.write("\n".join(logs) + "\n")
    if trace:
        with open(os.path.join(d, stem + ".trace"), "w") as f:
            f.write(trace_text(g.simulate("BBBF", 0, 120, arm == "on")[2]))


def write_reference(path):
    with open(path, "w") as f:
        for i in range(16):
            f.write(json.dumps({"kind": "request", "id": f"t{i:02d}", "rep": 0, "stream": i % 4,
                                "text_sha256": f"turn{i}", "out": 101}) + "\n")


def make_set(root, pairs_per_machine=8, machines=("m1", "m2", "m3"), on=(30.0, 1.0), off=(25.0, 1.0),
             comb_on=(35.0, 0.6), comb_off=(35.0, 0.6), seed=7, **run_kw):
    rng = random.Random(seed)
    for mi, m in enumerate(machines):
        d = os.path.join(root, m)
        os.makedirs(d)
        seq = 1
        for k in range(pairs_per_machine):
            first = "on" if (k + mi) % 2 == 0 else "off"  # the starting arm alternates per machine
            for arm in (first, "off" if first == "on" else "on"):
                slow = rng.gauss(*(on if arm == "on" else off))
                comb = rng.gauss(*(comb_on if arm == "on" else comb_off))
                write_run(d, f"{seq:02d}-{arm}", arm, slow, max(comb, slow + 1), rng, **run_kw)
                seq += 1


class RunSet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "set")
        self.ref = os.path.join(self.tmp.name, "ref.jsonl")
        write_reference(self.ref)

    def tearDown(self):
        self.tmp.cleanup()

    def judge(self, **kw):
        ok, lines, d = g.judge(self.root, self.ref, boot=2000, **kw)
        self.assertEqual(len(lines), 10, lines)
        return ok, lines, d

    def test_clear_gain_passes(self):
        make_set(self.root)
        ok, lines, _ = self.judge()
        self.assertTrue(ok, "\n".join(lines))
        self.assertTrue(lines[0].startswith("fair gate: PASS"))
        self.assertEqual(lines[9], "reasons: none")

    def test_main_exit_codes_and_ten_lines(self):
        make_set(self.root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = g.main([self.root, "--reference", self.ref, "--boot", "2000"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(buf.getvalue().rstrip("\n").split("\n")), 10)

    def test_no_gain_fails(self):
        make_set(self.root, on=(25.0, 1.0))
        ok, lines, _ = self.judge()
        self.assertFalse(ok)
        self.assertIn("FAIL", lines[5])

    def test_combined_loss_beyond_noise_fails(self):
        make_set(self.root, comb_on=(32.0, 0.6))
        ok, lines, _ = self.judge()
        self.assertFalse(ok)
        self.assertIn("FAIL", lines[7])
        self.assertIn("pass", lines[5])

    def test_combined_loss_within_noise_passes(self):
        make_set(self.root, comb_on=(34.9, 0.6))
        ok, lines, _ = self.judge()
        self.assertTrue(ok, "\n".join(lines))

    def test_one_changed_turn_fails_identity(self):
        make_set(self.root, pairs_per_machine=6)
        d = os.path.join(self.root, "m1")
        stem = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))[0][:-6]
        arm = "on" if stem.endswith("on") else "off"
        write_run(d, stem, arm, 30.0 if arm == "on" else 25.0, 35.0, None, identical=False)
        ok, lines, _ = self.judge(min_pairs=6)
        self.assertFalse(ok)
        self.assertIn("35 of 36 runs identical", lines[1])
        self.assertIn("not identical", lines[9])

    def test_not_alternated_fails(self):
        make_set(self.root, pairs_per_machine=6)
        d = os.path.join(self.root, "m1")
        stems = sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl"))
        a, b = stems[0], stems[1]
        arm = "on" if a.endswith("on") else "off"
        write_run(d, b, arm, 28.0, 35.0, None)
        ok, lines, _ = self.judge(min_pairs=6)
        self.assertFalse(ok)
        self.assertIn("not alternated", lines[9])

    def test_too_few_pairs_fails(self):
        make_set(self.root, pairs_per_machine=2)
        ok, lines, _ = self.judge()
        self.assertFalse(ok)
        self.assertIn("too few pairs", lines[9])

    def test_fair_on_without_trace_fails(self):
        make_set(self.root, trace=False)
        ok, lines, _ = self.judge()
        self.assertFalse(ok)
        self.assertIn("without a trace", lines[9])

    def test_legacy_trace_under_fair_on_fails(self):
        make_set(self.root)
        d = os.path.join(self.root, "m3")
        on = sorted(f for f in os.listdir(d) if f.endswith("-on.trace"))[0]
        with open(os.path.join(d, on), "w") as f:
            f.write(trace_text(g.simulate("BBBB", 0, 120, False)[2]))
        ok, lines, _ = self.judge()
        self.assertFalse(ok)
        self.assertIn("fairness (a): 23 of 24", lines[2])

    def test_no_pairing_line_fails_unjudged(self):
        make_set(self.root, pairing=False)
        ok, lines, _ = self.judge()
        self.assertFalse(ok)
        self.assertIn("no pairing line", lines[9])
        self.assertIn("not judged", lines[1])

    def test_unpaired_last_run_is_left_out(self):
        make_set(self.root)
        write_run(os.path.join(self.root, "m1"), "99-on", "on", 10.0, 20.0, None)
        ok, lines, _ = self.judge()
        self.assertTrue(ok, "\n".join(lines))
        self.assertIn("1 unpaired left out", lines[0])

    def test_garbage_input_is_a_verdict_not_a_traceback(self):
        make_set(self.root, pairs_per_machine=2)
        with open(os.path.join(self.root, "m1", "01-on.jsonl"), "w") as f:
            f.write("{not json\n")
        ok, lines, _ = self.judge()
        self.assertFalse(ok)
        self.assertTrue(lines[9].startswith("reasons: "))


class TraceMode(unittest.TestCase):
    def test_trace_mode(self):
        with tempfile.TemporaryDirectory() as d:
            good, old = os.path.join(d, "good.trace"), os.path.join(d, "old.trace")
            with open(good, "w") as f:
                f.write(trace_text(g.simulate("BBBB", 0, 100, True)[2]))
            lines = g.simulate("BBBB", 0, 100, True)[2]
            for ln in lines:
                ln.pop("cand", None)
            with open(old, "w") as f:
                f.write(trace_text(lines))
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.assertEqual(g.main(["--trace", good]), 0)
                self.assertEqual(g.main(["--trace", good, old]), 1)
            self.assertIn("lacks candidates", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")  # the older bench readers leave files to the collector
