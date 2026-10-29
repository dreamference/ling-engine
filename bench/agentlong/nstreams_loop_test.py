#!/usr/bin/env python3
"""Tests for nstreams_loop.py (host only, no GPU, no docker): the rotating plan pairs up the way fair_gate.py reads a
run set, each run's container and client command lines, the continuation of a set, and a dry run that touches
nothing (no lock, no holds log, no subprocess).

    python3 bench/agentlong/nstreams_loop_test.py
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fair_gate  # noqa: E402
import nstreams_loop as nl  # noqa: E402


def args(*extra):
    return nl.parse(["--image", "img:test", "--machine", "node-x", "--out", "/tmp/nsl-out", *extra])


class Plan(unittest.TestCase):
    def test_pairs_alternate_for_the_gate(self):
        for rounds in range(1, 7):
            runs = nl.plan(rounds)
            for level in nl.LEVELS:
                mine = [r for r in runs if r["level"] == level]
                self.assertEqual([r["n"] for r in mine], list(range(1, 2 * rounds + 1)))
                machines = {"node-x": [{"stem": r["label"], "seq": r["n"], "arm": "on" if r["fair"] else "off"}
                                       for r in mine]}
                pairs, left = fair_gate.pair_up(machines)  # raises on two same-arm runs in a pair
                self.assertEqual(len(pairs["node-x"]), rounds)
                self.assertEqual(left, [])

    def test_each_round_is_the_2x2_and_rotates(self):
        runs = nl.plan(4)
        for r in range(1, 5):
            cells = [(x["level"], x["fair"]) for x in runs if x["round"] == r]
            self.assertEqual(sorted(cells), sorted(nl.ROTATION))
            self.assertEqual(cells[0], nl.ROTATION[(r - 1) % 4])
        self.assertEqual([x["label"] for x in runs[:4]], ["1-off-flat", "2-on-flat", "1-on-paged", "2-off-paged"])

    def test_continuation_matches_one_long_loop(self):
        self.assertEqual(nl.plan(2, start=2), nl.plan(4)[8:])

    def test_done_rounds(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(nl.done_rounds(out, "SET", "node-x"), 0)
            for lv in nl.LEVELS:
                d = os.path.join(out, f"SET-{lv}", "node-x")
                os.makedirs(d)
                for n in (1, 2, 3, 4):
                    open(os.path.join(d, f"{n}-x.jsonl"), "w").close()
            self.assertEqual(nl.done_rounds(out, "SET", "node-x"), 2)
            os.remove(os.path.join(out, "SET-paged", "node-x", "4-x.jsonl"))
            with self.assertRaises(SystemExit):
                nl.done_rounds(out, "SET", "node-x")


class Commands(unittest.TestCase):
    def test_container_line(self):
        a = args("--port", "8311")
        for fair in (0, 1):
            for paged in (False, True):
                c = nl.container_cmd(a, "nstreams-node-x-flat-1", "/o/SET-flat/node-x", "1-on-flat.trace", fair, paged,
                                     "abc")
                self.assertEqual(c[:3], ["docker", "run", "-d"])
                self.assertNotIn("--restart", c)
                self.assertNotIn("--rm", c)
                self.assertIn("LING_STEP_TRACE=/fair/1-on-flat.trace", c)
                self.assertIn("/o/SET-flat/node-x:/fair", c)
                self.assertIn(f"LING_FAIR_PAIRING={fair}", c)
                self.assertIn(f"LING_KV_PAGES={int(paged)}", c)
                serve = c[c.index("img:test") + 1:]
                self.assertEqual(serve[0], "ling-serve")
                self.assertEqual(serve[serve.index("--port") + 1], "8311")
                self.assertEqual(serve[serve.index("--host") + 1], "127.0.0.1")
                self.assertEqual(serve[serve.index("--max-context") + 1], "65536")
                self.assertTrue(serve[serve.index("--model") + 1].endswith("/snapshots/abc"))
                # Four sessions in every cell; the arm's two flags last (ling-serve takes the last paging flag).
                self.assertLess(serve.index("--batch"), serve.index("--fair-pairing"))
                self.assertEqual(serve[serve.index("--sessions") + 1], "4")
                self.assertEqual(serve[-3:], ["--fair-pairing", str(fair), "--kv-pages" if paged else "--no-kv-pages"])
                self.assertEqual(sum(x in ("--kv-pages", "--no-kv-pages") for x in serve), 1)

    def test_client_line(self):
        c = nl.client_cmd(args(), "3-on-flat", "/o/SET-flat/node-x/3-on-flat.jsonl")
        self.assertTrue(c[1].endswith("n_streams.py"))
        self.assertEqual(c[2:], ["--url", "http://127.0.0.1:8300", "--streams", "4", "--mode", "greedy", "--warm",
                                 "--cache-key", "--label", "3-on-flat", "--out", "/o/SET-flat/node-x/3-on-flat.jsonl"])

    def test_reference_is_outside_the_sets(self):
        todo = nl.steps(args("--reference", "--rounds", "1"), "abc", 0)
        ref = todo[0]
        self.assertEqual(ref[0]["label"], "REF")
        self.assertEqual(ref[1], "/tmp/nsl-out/REF/node-x")
        self.assertIn("--no-kv-pages", ref[3])
        self.assertEqual(ref[3][-3:-1], ["--fair-pairing", "0"])
        self.assertEqual([t[1] for t in todo[1:]], ["/tmp/nsl-out/SET-flat/node-x"] * 2 + ["/tmp/nsl-out/SET-paged/node-x"] * 2)
        self.assertEqual(len({t[2] for t in todo}), len(todo))  # one container name per run


class DryRun(unittest.TestCase):
    def test_touches_nothing(self):
        def refuse(*a, **k):
            raise AssertionError(f"dry run started a process: {a}")

        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as out, \
                mock.patch.dict(os.environ, {"HOME": home}), \
                mock.patch.object(subprocess, "run", refuse), mock.patch.object(subprocess, "Popen", refuse), \
                mock.patch.object(nl.fcntl, "flock", refuse):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = nl.main(["--image", "img:test", "--machine", "node-x", "--out", out, "--rounds", "2",
                              "--reference", "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertEqual(os.listdir(home), [])
            self.assertEqual(os.listdir(out), [])
        text = buf.getvalue()
        self.assertIn("9 runs on node-x", text)
        self.assertEqual(text.count("docker run -d"), 9)
        self.assertEqual(text.count("n_streams.py"), 9)
        self.assertIn("round 2 4-off-flat:", text)


if __name__ == "__main__":
    unittest.main()
