#!/usr/bin/env python3
"""BACKLOG §19 item 27 (a): how much an entropy coder could take off the served checkpoint's weight bytes.

For every NVFP4 matrix (U8 codes, two E2M1 codes per byte, low nibble first, one F8_E4M3 scale per 16 codes):
H(code), H(code | block scale code), and the redundancy 4 - H(code | scale) bits per weight. For every FP8 matrix
(F8_E4M3, a per-tensor F32 scale): H(byte) and 8 - H. For every BF16 tensor: H(exponent byte), DFloat11's target,
with sign and mantissa taken as incompressible. Entropies are empirical (plug-in) over the whole tensor, so they
are a lower bound on any static per-tensor coder and say nothing about decode cost. CPU only.

Usage: code_entropy.py SNAPSHOT_DIR [--workers 4] [--json OUT]
"""
import argparse
import glob
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROWS = 1024  # rows per chunk: bounds the memory of one worker to a few hundred MB


def header(path):
    with open(path, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        return json.loads(h.read(n)), 8 + n


def tensor(path, base, meta, dtype):
    start, end = meta["data_offsets"]
    m = np.memmap(path, dtype=np.uint8, mode="r", offset=base + start, shape=(end - start,))
    return m.view(dtype).reshape(meta["shape"]) if meta["shape"] else m.view(dtype)


def entropy(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def nvfp4(job):
    name, wpath, wbase, wmeta, spath, sbase, smeta = job
    w = tensor(wpath, wbase, wmeta, np.uint8)            # [N, K/2]
    s = tensor(spath, sbase, smeta, np.uint8)            # [N, K/16], F8_E4M3 bit patterns
    joint = np.zeros(256 * 16, dtype=np.int64)
    for r in range(0, w.shape[0], ROWS):
        wb = np.asarray(w[r:r + ROWS])
        sb = np.repeat(np.asarray(s[r:r + ROWS]).astype(np.int32), 8, axis=1)  # one scale per 8 bytes
        base = sb * 16
        joint += np.bincount((base + (wb & 15)).ravel(), minlength=4096)
        joint += np.bincount((base + (wb >> 4)).ravel(), minlength=4096)
    j = joint.reshape(256, 16)
    h_code = entropy(j.sum(0))
    h_scale = entropy(j.sum(1))
    h_cond = entropy(joint) - h_scale                   # H(code | scale) = H(scale, code) - H(scale)
    n = int(joint.sum())
    return dict(name=name, kind="nvfp4", weights=n, h_code=h_code, h_code_given_scale=h_cond,
                redundancy=4 - h_cond, h_scale=h_scale, scale_codes_used=int((j.sum(1) > 0).sum()),
                bytes=n // 2 + n // 16)


def fp8(job):
    name, path, base, meta = job
    w = tensor(path, base, meta, np.uint8)
    counts = np.zeros(256, dtype=np.int64)
    for r in range(0, w.shape[0], ROWS):
        counts += np.bincount(np.asarray(w[r:r + ROWS]).ravel(), minlength=256)
    h = entropy(counts)
    return dict(name=name, kind="fp8", weights=int(counts.sum()), h_byte=h, redundancy=8 - h,
                bytes=int(counts.sum()))


def bf16(job):
    name, path, base, meta = job
    w = tensor(path, base, meta, np.uint16).reshape(-1)
    counts = np.zeros(256, dtype=np.int64)
    step = ROWS * 8192
    for r in range(0, w.shape[0], step):
        counts += np.bincount(((np.asarray(w[r:r + step]) >> 7) & 0xFF).ravel(), minlength=256)
    h = entropy(counts)
    return dict(name=name, kind="bf16", weights=int(counts.sum()), h_exponent=h, redundancy=8 - h,
                bytes=2 * int(counts.sum()))


def run(job):
    return {"nvfp4": nvfp4, "fp8": fp8, "bf16": bf16}[job[0]](job[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    where = {}
    for f in sorted(glob.glob(os.path.join(a.snapshot, "*.safetensors"))):
        hd, base = header(f)
        for k, v in hd.items():
            if k != "__metadata__":
                where[k] = (f, base, v)
    jobs = []
    for k, (f, base, v) in where.items():
        if not k.endswith(".weight") or "visual" in k or "mtp" in k:
            continue
        if v["dtype"] == "U8":
            sf, sbase, sv = where[k + "_scale"]
            jobs.append(("nvfp4", k, f, base, v, sf, sbase, sv))
        elif v["dtype"] == "F8_E4M3":
            jobs.append(("fp8", k, f, base, v))
        elif v["dtype"] == "BF16" and len(v["shape"]) == 2 and min(v["shape"]) > 1:
            jobs.append(("bf16", k, f, base, v))
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(run, jobs))
    if a.json:
        with open(a.json, "w") as out:
            json.dump(rows, out, indent=1)
    # Grouped report, weighted by weight count.
    def group(name):
        for g in ("mlp.gate_up_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "linear_attn.in_proj_qkv",
                  "linear_attn.in_proj_z", "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj",
                  "self_attn.v_proj", "self_attn.o_proj", "lm_head", "embed_tokens"):
            if g in name:
                return g
        return name.split(".")[-2]
    agg = {}
    for r in rows:
        key = (r["kind"], group(r["name"]))
        g = agg.setdefault(key, dict(weights=0, bytes=0, red=0.0, h=0.0, hc=0.0, n=0, lo=9.0, hi=-1.0))
        g["weights"] += r["weights"]; g["bytes"] += r["bytes"]; g["n"] += 1
        g["red"] += r["redundancy"] * r["weights"]
        g["lo"] = min(g["lo"], r["redundancy"]); g["hi"] = max(g["hi"], r["redundancy"])
        if r["kind"] == "nvfp4":
            g["h"] += r["h_code"] * r["weights"]; g["hc"] += r["h_code_given_scale"] * r["weights"]
    print(f"{'kind':6} {'group':26} {'tensors':>7} {'GB':>7} {'H(code)':>8} {'H(c|s)':>8} {'redund.':>8} {'min':>6} {'max':>6} {'GB saved':>8}")
    total = dict()
    for (kind, g), v in sorted(agg.items()):
        red = v["red"] / v["weights"]
        saved = red * v["weights"] / 8 / 1e9
        total.setdefault(kind, [0, 0.0, 0.0])
        total[kind][0] += v["bytes"]; total[kind][1] += saved; total[kind][2] += red * v["weights"]
        h = f"{v['h'] / v['weights']:8.3f}" if kind == "nvfp4" else f"{'':8}"
        hc = f"{v['hc'] / v['weights']:8.3f}" if kind == "nvfp4" else f"{'':8}"
        print(f"{kind:6} {g:26} {v['n']:7d} {v['bytes'] / 1e9:7.3f} {h} {hc} {red:8.3f} {v['lo']:6.3f} {v['hi']:6.3f} {saved:8.3f}")
    for kind, (b, saved, _) in total.items():
        print(f"{kind}: {b / 1e9:.3f} GB, at most {saved:.3f} GB ({100 * saved * 1e9 / b:.1f}%) by a per-tensor entropy coder")


if __name__ == "__main__":
    sys.exit(main())
