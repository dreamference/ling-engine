#!/usr/bin/env python3
"""nstreams_loop.py: the --fair-pairing bootstrap run set (2x2: fair pairing on/off x paging on/off) on one machine,
each run a fresh container of a ling-serve image, for fair_gate.py.

Rounds of four runs in rotating order (flat-off, flat-on, paged-on, paged-off; each round starts one later), so each
round adds one on/off pair to each paging level and neither arm always runs first. A run: a container of IMAGE at
Mightling's ling-serve flags (--sessions 4 --batch 4 explicit in every cell, then `--fair-pairing 0|1` and
`--no-kv-pages|--kv-pages` last, with LING_FAIR_PAIRING and LING_KV_PAGES set to the same values), its step trace
written to the run's directory (LING_STEP_TRACE), mem_guard.sh beside it, its own port on loopback (not behind the
model gate); then n_streams.py (4 warm greedy conversations), `docker stop -t 60` (the trace is flushed), the log
saved as N-LABEL.serve.log, the container removed. Output:

    OUT/SET-flat/MACHINE/N-LABEL.{jsonl,serve.log,trace,log,metrics.txt}
    OUT/SET-paged/MACHINE/N-LABEL.{...}          N numbered per paging level: 1, 2, 3, ... (pairs 1-2, 3-4, ...)
    OUT/REF/MACHINE/REF.{jsonl,serve.log}          with --reference: one flat fair-off run first, outside the sets

Judge each level with `fair_gate.py OUT/SET-flat --reference REF.jsonl` (and SET-paged), the sets of all machines
copied under one root. A second invocation on the same OUT continues the numbering and the rotation (both levels
must hold the same even number of runs). A failed run (no health within --start-timeout, n_streams.py non-zero, or
the serve log naming the wrong arm) stops the loop: its files go to MACHINE/failed/, so the pairs before it stay
whole. Rounds are whole: the loop stops before a round that would end after --until, or once the file OUT/STOP
exists, never inside one.

Under the GPU lock (~/.ling-gpu.lock, held for the whole loop; the wait, the hold and m0 logged in
~/.ling-gpu-holds.log as AGENT): the production server m0-prod8000 is stopped if it runs, and started again at the
end unless KEEP_M0_DOWN=1 or another job waits for the lock. Any other model server still running then (a container
of an sglang, vllm or ling-serve image) refuses the loop: one model server per GB10.

--dry-run prints the plan and every command it would run and touches nothing: no lock, no log line, no docker.

Usage (run it with setsid nohup ... < /dev/null &):
    nstreams_loop.py --image IMAGE --machine node-b --out DIR [--rounds 4] [--reference] [--until 2026-10-31T07:00]
                     [--port 8300] [--max-context 65536] [--dry-run]
"""
import argparse
import datetime
import fcntl
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROTATION = (("flat", 0), ("flat", 1), ("paged", 1), ("paged", 0))  # (paging level, fair pairing) of round 1
LEVELS = ("flat", "paged")
SERVED = "RadixArk/Qwen3.8-27B-NVFP4"
HUB_REPO = "models--RadixArk--Qwen3.8-27B-NVFP4"
C_HF = "/root/.cache/huggingface"
C_DGX = "/root/.cache/dreamference"
DRAFT = C_DGX + "/drafters/dflash2-nvfp4-scalesearch"  # the drafter Mightling's recipe pins (ScaleSearch)
PROD = "m0-prod8000"
SERVER_IMAGES = re.compile(r"sglang|vllm|ling-serve")
PAIRING = re.compile(r"^pairing: (fair|running order)", re.M)
ROUND_MIN = 40  # a run ~10 min (start + ~8 min of n_streams), four a round


def plan(rounds, start=0):
    """The runs of `rounds` rounds from round index `start` (rounds already done), numbered per paging level.

    Returns:
        list of dicts: round, level, fair, n (the run's number in its level), label (N-on|off-LEVEL).
    """
    count = {lv: 2 * start for lv in LEVELS}
    runs = []
    for r in range(start, start + rounds):
        for i in range(4):
            level, fair = ROTATION[(r + i) % 4]
            count[level] += 1
            n = count[level]
            runs.append({"round": r + 1, "level": level, "fair": fair, "n": n,
                         "label": f"{n}-{'on' if fair else 'off'}-{level}"})
    return runs


def done_rounds(out, set_name, machine):
    """Rounds already in OUT (each level's runs counted); refuses levels that are uneven or differ."""
    have = {lv: len(glob.glob(os.path.join(out, f"{set_name}-{lv}", machine, "*.jsonl"))) for lv in LEVELS}
    if have["flat"] != have["paged"] or have["flat"] % 2:
        raise SystemExit(f"cannot continue: {have} runs per level (needs the same even number in both)")
    return have["flat"] // 2


def container_cmd(a, name, run_dir, trace, fair, paged, model_dir):
    """`docker run` of one run's server: Mightling's ling-serve flags, the arm's two flags last."""
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    dgx = os.path.expanduser("~/.cache/dreamference")
    return ["docker", "run", "-d", "--name", name, "--gpus", "all", "--ipc=host", "--network", "host",
            "-v", f"{hf}:{C_HF}", "-v", f"{dgx}:{C_DGX}", "-v", f"{run_dir}:/fair",
            "-e", f"LING_STEP_TRACE=/fair/{trace}", "-e", f"LING_FAIR_PAIRING={fair}", "-e", f"LING_KV_PAGES={int(paged)}",
            a.image, "ling-serve", "--model", f"{C_HF}/hub/{HUB_REPO}/snapshots/{model_dir}",
            "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", str(a.port),
            "--max-context", str(a.max_context), "--draft", a.draft, "--draft-block", "12",
            "--sessions", "4", "--batch", "4", "--graphs", "--pdl",
            "--fair-pairing", str(fair), "--kv-pages" if paged else "--no-kv-pages"]


def client_cmd(a, label, jsonl):
    """n_streams.py: 4 warm greedy conversations against the run's server."""
    return [sys.executable, os.path.join(HERE, "n_streams.py"), "--url", f"http://127.0.0.1:{a.port}",
            "--streams", "4", "--mode", "greedy", "--warm", "--cache-key", "--label", label, "--out", jsonl]


def steps(a, model_dir, start):
    """Every run of the loop as (run, run_dir, container name, docker run, client command)."""
    runs = plan(a.rounds, start)
    if a.reference:
        runs.insert(0, {"round": 0, "level": "flat", "fair": 0, "n": 0, "label": "REF"})
    out = []
    for r in runs:
        sub = "REF" if r["label"] == "REF" else f"{a.set}-{r['level']}"
        run_dir = os.path.join(a.out, sub, a.machine)
        name = f"nstreams-{a.machine}-{'ref' if r['label'] == 'REF' else r['level'] + '-' + str(r['n'])}"
        out.append((r, run_dir, name,
                    container_cmd(a, name, run_dir, r["label"] + ".trace", r["fair"], r["level"] == "paged",
                                  model_dir),
                    client_cmd(a, r["label"], os.path.join(run_dir, r["label"] + ".jsonl"))))
    return out


class Host:
    """The real side effects: lock, holds log, docker, mem_guard, the client."""

    def __init__(self, agent):
        self.agent = agent
        self.holds = os.path.expanduser("~/.ling-gpu-holds.log")
        self.lock_path = os.path.expanduser("~/.ling-gpu.lock")
        self.lock = None
        self.t_lock = 0.0
        self.m0_stopped = False

    def note(self, text):
        with open(self.holds, "a") as f:
            f.write(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} {self.agent} {text}\n")

    @staticmethod
    def say(text):
        print(f"{datetime.datetime.now(datetime.timezone.utc):%H:%M:%S} {text}", flush=True)

    @staticmethod
    def docker(*args, check=False):
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)

    def running(self, name):
        return bool(self.docker("ps", "-q", "-f", f"name=^{name}$").stdout.strip())

    def take(self, desc):
        self.note(f"waiting for the lock ({desc})")
        self.lock = open(self.lock_path, "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.t_lock = time.time()
        self.say("=== lock taken")
        self.note(f"lock taken: {desc}")
        if self.running(PROD):
            self.docker("stop", PROD)
            self.m0_stopped = True
            self.say(f"{PROD} stopped")
            self.note(f"{PROD} stopped")
        others = [ln for ln in self.docker("ps", "--format", "{{.Names}} {{.Image}}").stdout.splitlines()
                  if SERVER_IMAGES.search(ln.split(" ", 1)[-1])]
        if others:
            raise RuntimeError(f"another model server is running: {'; '.join(others)}")

    def release(self, desc):
        # Waiters: every other process with the lock file open (a flock waiting, or a script that opened it to wait).
        fuser = subprocess.run(["fuser", self.lock_path], capture_output=True, text=True)
        waiting = [p for p in fuser.stdout.split() if p.strip().isdigit() and int(p) != os.getpid()]
        if os.environ.get("KEEP_M0_DOWN") != "1" and not waiting:
            self.docker("start", PROD)
            self.say(f"{PROD} started")
            self.note(f"{PROD} started again")
        elif waiting:
            self.say(f"{PROD} left down: {len(waiting)} job(s) waiting for the lock")
            self.note(f"{len(waiting)} job(s) waiting for the lock: {PROD} left stopped for them")
        held = int((time.time() - self.t_lock) / 60)
        self.say(f"=== lock released, held {held} min")
        self.note(f"lock released, held {held} min ({desc})")
        self.lock.close()

    def run_one(self, a, run, run_dir, name, start_cmd, client):
        """One run; True when it completed and its serve log names the arm it was started as."""
        label = run["label"]
        os.makedirs(run_dir, exist_ok=True)
        self.docker("rm", "-f", name)
        started = subprocess.run(start_cmd, capture_output=True, text=True)
        if started.returncode:
            self.say(f"{label}: docker run failed: {started.stderr.strip()[-300:]}")
            return False
        subprocess.Popen([os.path.join(HERE, "mem_guard.sh"), f"container:{name}", os.path.join(run_dir, "mem_guard.log")],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        ok = self.wait_health(a, name)
        rc = None
        if ok:
            with open(os.path.join(run_dir, label + ".log"), "w") as log:
                rc = subprocess.run(client, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            for path, what in (("metrics", ".metrics.txt"), ("spec_stats", ".spec.json")):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/{path}", timeout=10) as r, \
                            open(os.path.join(run_dir, label + what), "wb") as f:
                        f.write(r.read())
                except OSError:
                    pass
        self.docker("stop", "-t", "60", name)
        serve_log = os.path.join(run_dir, label + ".serve.log")
        with open(serve_log, "w") as f:  # one stream, as `docker logs NAME > LOG 2>&1`
            subprocess.run(["docker", "logs", name], stdout=f, stderr=subprocess.STDOUT)
        self.docker("rm", name)
        with open(serve_log, errors="replace") as f:
            arm = PAIRING.search(f.read())
        want = "fair" if run["fair"] else "running order"
        good = ok and rc == 0 and arm is not None and arm.group(1) == want
        self.say(f"{label}: {'ok' if good else 'FAILED'} (healthy {ok}, n_streams exit {rc}, "
                 f"pairing {arm.group(1) if arm else None})")
        return good

    def wait_health(self, a, name):
        end = time.time() + a.start_timeout
        while time.time() < end:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2):
                    return True
            except OSError:
                pass
            if not self.running(name):
                return False
            time.sleep(5)
        return False


def move_failed(run_dir, label):
    failed = os.path.join(run_dir, "failed")
    os.makedirs(failed, exist_ok=True)
    for f in glob.glob(os.path.join(run_dir, label + ".*")):
        shutil.move(f, os.path.join(failed, os.path.basename(f)))


def preflight(a):
    """Before the lock: the image's ling-serve has the two switches and the step trace; the inputs exist.

    The binary's strings are read (its usage text and the variable names), since ling-serve itself needs the GPU's
    driver to start at all, even to print its usage.
    """
    missing = []
    for word in ("--fair-pairing", "--kv-pages", "--no-kv-pages", "LING_STEP_TRACE"):
        found = subprocess.run(["docker", "run", "--rm", "--entrypoint", "sh", a.image, "-c",
                                f'grep -c -a -e "{word}" "$(command -v ling-serve)"'],
                               capture_output=True, text=True, timeout=300)
        if found.stdout.strip() in ("", "0"):
            missing.append(word)
    if missing:
        raise SystemExit(f"{a.image}: ling-serve lacks {' '.join(missing)} (an engine before fair pairing)")
    hub = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub", HUB_REPO)
    snaps = sorted(os.listdir(os.path.join(hub, "snapshots"))) if os.path.isdir(os.path.join(hub, "snapshots")) else []
    model_dir = a.model_rev or (snaps[0] if len(snaps) == 1 else None)
    if not model_dir:
        raise SystemExit(f"model snapshot: {len(snaps)} under {hub}; name one with --model-rev")
    host_draft = a.draft.replace(C_DGX, os.path.expanduser("~/.cache/dreamference"), 1)
    if not os.path.isdir(host_draft):
        raise SystemExit(f"no drafter at {host_draft}")
    if not os.path.exists(os.path.expanduser("~/bench-sets/agentlong/bodies.jsonl")):
        raise SystemExit("no ~/bench-sets/agentlong/bodies.jsonl (n_streams.py's request bodies)")
    return model_dir


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True, help="the candidate ling-serve image")
    ap.add_argument("--machine", required=True, help="the machine's label in the set (node-a, node-b, node-c)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="SET", help="the sets' name prefix (OUT/SET-flat, OUT/SET-paged)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reference", action="store_true", help="one flat fair-off run first, to OUT/REF/MACHINE/REF.jsonl")
    ap.add_argument("--until", help="UTC time (YYYY-MM-DDTHH:MM) no round may end after (a round counts 40 min)")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--max-context", type=int, default=65536,
                    help="every measured window and the m1w reference ran 65,536 (production pins 262,144)")
    ap.add_argument("--draft", default=DRAFT, help="the drafter's path inside the container")
    ap.add_argument("--model-rev", help="the checkpoint's snapshot directory (default: the only one in the hub)")
    ap.add_argument("--start-timeout", type=int, default=900, help="seconds for a server to answer /health")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.out = os.path.abspath(a.out)
    return a


def main(argv=None):
    a = parse(argv)
    start = done_rounds(a.out, a.set, a.machine)
    if a.dry_run:
        model_dir = a.model_rev or "<snapshot>"
    else:
        model_dir = preflight(a)
    until = datetime.datetime.strptime(a.until, "%Y-%m-%dT%H:%M").replace(
        tzinfo=datetime.timezone.utc) if a.until else None
    todo = steps(a, model_dir, start)
    print(f"{len(todo)} runs on {a.machine}: rounds {start + 1}..{start + a.rounds} of {a.set}"
          f"{', the reference first' if a.reference else ''}; image {a.image}")
    if a.dry_run:
        for run, _, _, start_cmd, client in todo:
            print(f"round {run['round']} {run['label']}:")
            print("  " + " ".join(start_cmd))
            print("  " + " ".join(client))
        return 0
    desc = f"nstreams_loop {a.set} {a.machine} {a.rounds} rounds"
    host = Host(os.environ.get("AGENT", "nstreams-loop"))
    status, last_round = 0, None
    try:
        host.take(desc)
        for run, run_dir, name, start_cmd, client in todo:
            if run["round"] != last_round and run["label"] != "REF":  # rounds are whole: stop only between them
                last_round = run["round"]
                if os.path.exists(os.path.join(a.out, "STOP")):
                    host.say("STOP file: no further round")
                    break
                eta = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=ROUND_MIN)
                if until and eta > until:
                    host.say(f"round {run['round']} would end after {a.until}: stopped")
                    break
            if not host.run_one(a, run, run_dir, name, start_cmd, client):
                move_failed(run_dir, run["label"])
                host.say(f"stopped after the failed run {run['label']} (its files in failed/)")
                status = 1
                break
    except RuntimeError as e:
        host.say(f"refused: {e}")
        status = 1
    finally:
        if host.lock:
            host.docker("rm", "-f", *[t[2] for t in todo])
            host.release(desc)
    return status


if __name__ == "__main__":
    sys.exit(main())
