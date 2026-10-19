"""BACKLOG §19 item 30, stage 2's side questions, CPU only, on the 208 FP8 projections of the served checkpoint.

  limits   the exponent Huffman's average length held to 5, 6, 7, 8 and 10 bits (package-merge per matrix), and
           what each limit costs as a fraction of today's bytes against 10 (the pair table needs 6)
  fixed    the exponent as a fixed 2- or 3-bit code per lane piece (2 rows x 32 values, 64 values), the window of
           2^w - 1 exponents placed best per piece, the last code an escape to a list (6-bit position + 4-bit
           exponent), a base and a count byte a piece: the table-free alternative to Huffman

Usage: fp8_code_limits.py SNAPSHOT_DIR
"""
import glob
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_code_size import limited_lengths  # noqa: E402


def header(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def main():
    snap = sys.argv[1]
    tot = 0
    lim_bits = {L: 0.0 for L in (5, 6, 7, 8, 10)}
    fix_bits, fix_esc = {2: 0, 3: 0}, {2: 0, 3: 0}
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        hd, base = header(f)
        mm = np.memmap(f, dtype=np.uint8, mode="r")
        for k, v in hd.items():
            if k == "__metadata__" or v["dtype"] != "F8_E4M3" or not k.endswith(".weight") or "visual" in k:
                continue
            N, K = v["shape"]
            a, b = v["data_offsets"]
            w = np.asarray(mm[base + a: base + b]).reshape(N, K)
            e = (w >> 3) & 15
            fr = np.bincount(e.ravel(), minlength=16).astype(np.int64)
            for L in lim_bits:
                lim_bits[L] += float((fr * limited_lengths(fr, L)).sum())
            p = e.reshape(N // 16, 2, 8, K // 32, 32).transpose(0, 2, 3, 1, 4).reshape(-1, 64)
            h = np.stack([(p == s).sum(1) for s in range(16)], 1)
            cs = np.concatenate([np.zeros((p.shape[0], 1), np.int64), h.cumsum(1)], 1)
            for wd in (2, 3):
                win = (1 << wd) - 1
                cover = np.max(np.stack([cs[:, min(b0 + win, 16)] - cs[:, b0] for b0 in range(16)], 1), 1)
                ne = 64 - cover
                fix_esc[wd] += int(ne.sum())
                fix_bits[wd] += p.shape[0] * (64 * (4 + wd) + 16) + int(ne.sum()) * 10
            tot += N * K
    print(f"{tot / 1e9:.3f} GB of FP8 projections")
    for L in lim_bits:
        print(f"limit {L:2d}: exponent {lim_bits[L] / tot:.4f} bits a value, "
              f"{(lim_bits[L] - lim_bits[10]) / 8 / tot:+.4f} of the bytes against 10")
    for wd in (2, 3):
        print(f"fixed {wd}-bit + escapes: {fix_bits[wd] / 8 / tot:.4f} of today's bytes, "
              f"escapes {fix_esc[wd] / tot:.4f} of the values")


if __name__ == "__main__":
    sys.exit(main())
