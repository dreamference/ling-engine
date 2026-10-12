#!/usr/bin/env python3
"""Moonshot P0, the GPU half: what a pass of 16-128 rows costs on the GB10, at ~8K, ~32K and ~56K context.

Runs on second-puffin with the GPU otherwise idle (no model server resident: stop the reference server
first and restart it afterwards, as the one-model-server policy says). Uses only `ling-run` from a
build of main; nothing here changes the engine.

Two measurements, because today's rows path stops at 32 rows (`kMaxStreamRows`):

  A  the verify itself, rows path, block B in --blocks (<= 32): `ling-run --spec --block B` prints the
     step's draft / verify / commit milliseconds; greedy, --max-tokens per prompt, one prompt per
     context, --reps times.
  B  passes of N rows through the prefill path, N in --chunks (multiples of 32, up to 128):
     `ling-run --chunk-check N` prefills a prompt in N-token chunks and prints the compute seconds.
     Two prompts per context point (lo, hi); the cost of one N-row pass at context ~(lo+hi)/2 is
     (T(hi) - T(lo)) / (chunks(hi) - chunks(lo)), which cancels the fixed costs. The prefill path
     also commits the DeltaNet state and writes the KV for its rows, so this is an upper bound on a
     verify-and-commit of N rows through a dedicated mid-M path (proposal section 3.4).

The summary prices a copy step of K rows against production's step (block 12: draft + verify + commit)
two ways: `extrapolated` continues the rows path's per-row slope from 16 to 32 rows out to K, plus the
32-row commit; `prefill path` is measurement B. The drafter's pass is not needed on a copy step. The
ratio is the step inflation c(K) that copy_stats.py's break-even (`breakeven_step_cost_for_20pct`)
is compared with.

The prompts are filler text from this repository's own files (no recorded session leaves its machine,
and timing does not depend on content except through the KV, which real text fills realistically).

Usage (on second-puffin, from the repository root of a build of main):
  python3 bench/rowsweep.py --build build --out /tmp/rowsweep [--reps 2] [--dry-run]
  python3 bench/rowsweep.py --summarize /tmp/rowsweep [--stats p0-stats.json]
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = '~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*'
DRAFT = '~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*'
CONTEXTS_A = [8192, 32768, 57344]
PAIRS_B = [(4096, 12288), (24576, 40960), (49152, 64512)]   # midpoints ~8K, ~32K, ~56K
CHARS_PER_TOKEN = 3.4   # code and English prose under the Qwen3.8 tokenizer, roughly; the run reports exact

SPEC_RE = re.compile(r'spec: (\d+) steps, ([\d.]+) tokens/step, ([\d.]+) ms/step '
                     r'\(draft ([\d.]+), verify ([\d.]+), commit ([\d.]+)\)')
CHUNK_RE = re.compile(r'prompt (\d+): (\d+) tokens, (\d+) chunks \(([\d.]+) s\) against (\d+) chunks of '
                      r'(\d+) \(([\d.]+) s\)')
PREFILL_RE = re.compile(r'prefill: (\d+) tokens in')


def one(pattern):
    hits = sorted(glob.glob(os.path.expanduser(pattern)))
    if not hits:
        sys.exit(f'not found: {pattern}')
    return hits[-1]


def filler(n_tokens):
    """Text of about n_tokens from the repository's specs and sources, cycled."""
    files = sorted(glob.glob(f'{REPO}/specs/**/*.md', recursive=True)) + \
        sorted(glob.glob(f'{REPO}/src/**/*.c*', recursive=True))
    want, parts, k = int(n_tokens * CHARS_PER_TOKEN), [], 0
    while sum(map(len, parts)) < want:
        with open(files[k % len(files)], errors='replace') as fh:
            parts.append(f'\n\n// ---- {os.path.relpath(files[k % len(files)], REPO)}\n' + fh.read())
        k += 1
    return ''.join(parts)[:want]


def gpu_idle():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return True, ''
    return not out, out


def run(cmd, log):
    t0 = time.time()
    with open(log, 'w') as fh:
        fh.write(' '.join(cmd) + '\n')
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f'  {os.path.basename(log)}: exit {rc}, {time.time() - t0:.0f} s', flush=True)
    return rc


def measure(a):
    os.makedirs(a.out, exist_ok=True)
    idle, procs = (True, '') if a.dry_run else gpu_idle()
    if not idle and not a.force:
        sys.exit(f'GPU not idle (stop the model server first, restart it after):\n{procs}')
    ling_run = os.path.join(a.build, 'ling-run')
    model, draft = (MODEL, DRAFT) if a.dry_run else (one(MODEL), one(DRAFT))
    flags = a.flags.split()
    blocks = [int(x) for x in a.blocks.split(',')]
    chunks = [int(x) for x in a.chunks.split(',')]
    # A: the rows path. One ling-run per block (one model load); every context, --reps times.
    pa = os.path.join(a.out, 'prompts-a.jsonl')
    with open(pa, 'w') as fh:
        for _ in range(a.reps):
            for L in CONTEXTS_A:
                fh.write(json.dumps(filler(L)) + '\n')
    plan = [([ling_run, '--model', model, '--draft', draft, '--block', str(B), '--spec', '--temperature', '0',
              '--max-tokens', str(a.max_tokens), '--max-context', '65536', '--prompts-file', pa] + flags,
             os.path.join(a.out, f'a-block{B}.log')) for B in blocks]
    # B: the prefill path in N-row chunks (no drafter: chunk-check then skips its continuation).
    pb = os.path.join(a.out, 'prompts-b.jsonl')
    with open(pb, 'w') as fh:
        for lo, hi in PAIRS_B:
            fh.write(json.dumps(filler(lo)) + '\n')
            fh.write(json.dumps(filler(hi)) + '\n')
    plan += [([ling_run, '--model', model, '--chunk-check', str(N), '--max-context', '65536',
               '--prompts-file', pb] + flags, os.path.join(a.out, f'b-chunk{N}.log')) for N in chunks]
    for cmd, log in plan:
        if a.dry_run:
            print(' '.join(cmd), '>', log)
        elif os.path.exists(log) and not a.redo and parse_ok(log):
            print(f'  {os.path.basename(log)}: kept')
        else:
            run(cmd, log)
    if not a.dry_run:
        summarize(a.out, a.stats)


def parse_ok(log):
    txt = open(log, errors='replace').read()
    return bool(SPEC_RE.search(txt) or CHUNK_RE.search(txt))


def summarize(out, stats_path=None):
    A = {}   # (context, block) -> [(draft, verify, commit)]
    for log in sorted(glob.glob(os.path.join(out, 'a-block*.log'))):
        B = int(re.search(r'block(\d+)', log).group(1))
        L = None
        for line in open(log, errors='replace'):
            m = PREFILL_RE.search(line)
            if m:   # each prompt's spec line follows its prefill line: tag it with the nearest context
                L = min(CONTEXTS_A, key=lambda c: abs(c - int(m.group(1))))
                continue
            m = SPEC_RE.search(line)
            if m and L is not None:
                A.setdefault((L, B), []).append(tuple(float(m.group(k)) for k in (4, 5, 6)))
                L = None
    Bm = {}  # (midpoint, N) -> ms per N-row pass
    for log in sorted(glob.glob(os.path.join(out, 'b-chunk*.log'))):
        rows = [CHUNK_RE.search(l) for l in open(log, errors='replace')]
        rows = [r for r in rows if r]
        for k in range(0, len(rows) - 1, 2):
            lo, hi = rows[k], rows[k + 1]
            N = int(hi.group(6))
            dc = int(hi.group(5)) - int(lo.group(5))
            if dc > 0:
                mid = (int(lo.group(2)) + int(hi.group(2))) // 2
                Bm[(mid, N)] = 1e3 * (float(hi.group(7)) - float(lo.group(7))) / dc
    med = lambda xs: statistics.median(xs) if xs else None
    res = {'rows_path_ms': {}, 'prefill_path_ms_per_pass': {f'{k[0]}@{k[1]}': v for k, v in Bm.items()},
           'copy_step_cost_ratio': {}}
    for (L, B), xs in sorted(A.items()):
        res['rows_path_ms'][f'{L}@{B}'] = {'draft': med([x[0] for x in xs]), 'verify': med([x[1] for x in xs]),
                                           'commit': med([x[2] for x in xs]), 'n': len(xs)}
    for L in CONTEXTS_A:
        r = res['rows_path_ms']
        base, v16, v32 = r.get(f'{L}@12'), r.get(f'{L}@16'), r.get(f'{L}@32')
        if not (base and v16 and v32):
            continue
        step12 = base['draft'] + base['verify'] + base['commit']
        slope = (v32['verify'] - v16['verify']) / 16
        mids = sorted((m for m in {k[0] for k in Bm} if abs(m - L) <= 0.4 * L), key=lambda m: abs(m - L))
        for K in (64, 96, 128):
            ext = v32['verify'] + slope * (K - 32) + v32['commit']
            pre = Bm.get((mids[0], K)) if mids else None
            res['copy_step_cost_ratio'][f'{L}@{K}'] = {
                'step12_ms': step12, 'extrapolated_ms': ext, 'extrapolated_ratio': ext / step12,
                'prefill_path_ms': pre, 'prefill_path_ratio': pre / step12 if pre else None}
    if stats_path:
        st = json.load(open(stats_path))
        be = {}
        for K in ('64', '128'):
            ent = st['oracle_model']['S1+S2'][K]['all']['m1']
            be[f'oracle S1+S2 K={K}'] = ent['breakeven_step_cost_for_20pct']
            rl = st['realistic_model'].get(f'm>=16,K={K}', {}).get('m1', {})
            be[f'realistic K={K}'] = rl.get('breakeven_step_cost_for_20pct')
        res['breakeven_from_statistics'] = be
    json.dump(res, sys.stdout, indent=1)
    print()
    with open(os.path.join(out, 'summary.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', default='build', help='build directory with ling-run')
    ap.add_argument('--out', help='directory for prompts, logs and summary.json')
    ap.add_argument('--blocks', default='8,12,16,24,32')
    ap.add_argument('--chunks', default='32,64,96,128')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=192)
    ap.add_argument('--flags', default='--graphs --pdl', help="ling-serve's production flags")
    ap.add_argument('--stats', help="copy_stats.py's JSON, for the break-even comparison")
    ap.add_argument('--summarize', metavar='DIR', help='only parse the logs in DIR')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--redo', action='store_true', help='rerun logs that already parsed')
    ap.add_argument('--force', action='store_true', help='run although the GPU has other processes')
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize, a.stats)
    elif a.out:
        measure(a)
    else:
        ap.error('--out or --summarize')


if __name__ == '__main__':
    main()
