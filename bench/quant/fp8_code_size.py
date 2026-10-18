#!/usr/bin/env python3
"""BACKLOG §19 item 30, stage 1: the bytes lossless codes achieve on the FP8 projections in the engine's tile layout.

The engine streams an FP8 matrix as tiles of 16 rows by 128 values (2,048 bytes, `retile_fp8`); a coded tile must
be found without decoding the ones before it, so every format carries a per-tile offset, and a format a warp
decodes in parallel carries a start for each lane's piece. Sizes are those of the real encodings' lengths (Huffman
code lengths summed and padded to bytes per stream; rANS at its quantized frequencies plus the final states), with
tables and offsets, not the entropy bound. Formats:

  exp-tile   Huffman on the E4M3 exponent (4 bits), sign and mantissa raw (a 1,024-byte nibble plane), one
             exponent stream per tile, a 4-byte tile offset
  exp-lane   the same, one exponent stream per 16-value piece (a lane's load), byte-padded, a 1-byte length per
             piece (128 per tile) and the 4-byte tile offset
  exp-warp   the same, one exponent stream per 32 pieces (a warp's share of a tile), byte-padded, a 2-byte length
             per stream (4 per tile) and the tile offset
  byte-tile  Huffman on the whole byte (length-limited to 12 bits), one stream per tile, the tile offset
  rans-S     rANS on the whole byte, 12-bit frequencies, S interleaved streams per tile (32-bit states), the offset

CPU only. Usage: fp8_code_size.py SNAPSHOT_DIR [--workers 4] [--json OUT]
"""
import argparse
import glob
import heapq
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TILE_ROWS, TILE_K = 16, 128
TILE_BYTES = TILE_ROWS * TILE_K
OFFSET = 4


def huffman_lengths(freq, limit):
    """Code lengths of a Huffman code over `freq`, flattened until no length exceeds `limit`."""
    f = freq.astype(np.float64)
    used = np.nonzero(f)[0]
    if len(used) == 1:
        out = np.zeros(len(f), dtype=np.int64); out[used] = 1; return out
    while True:
        heap = [(f[i], int(i), (int(i),)) for i in used]
        heapq.heapify(heap)
        depth = np.zeros(len(f), dtype=np.int64)
        while len(heap) > 1:
            a, ia, sa = heapq.heappop(heap)
            b, ib, sb = heapq.heappop(heap)
            for s in sa + sb:
                depth[s] += 1
            heapq.heappush(heap, (a + b, min(ia, ib), sa + sb))
        if depth.max() <= limit:
            return depth
        f[used] = np.sqrt(f[used]) + f[used].mean() * 1e-3   # flatten and retry: conservative


def rans_cost(freq):
    """Bits per symbol at 12-bit quantized frequencies (every used symbol at least 1/4096)."""
    total = 1 << 12
    used = freq > 0
    q = np.zeros(len(freq), dtype=np.float64)
    q[used] = np.maximum(1, np.floor(freq[used] / freq.sum() * total))
    while q.sum() > total:                      # trim the largest until the table sums to 4,096
        q[np.argmax(q)] -= q.sum() - total
    q[np.argmax(q)] += total - q.sum()
    cost = np.zeros(len(freq))
    cost[used] = -np.log2(q[used] / total)
    return cost


def header(path):
    with open(path, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        return json.loads(h.read(n)), 8 + n


def measure(job):
    name, path, base, meta = job
    N, K = meta["shape"]
    start, end = meta["data_offsets"]
    w = np.memmap(path, dtype=np.uint8, mode="r", offset=base + start, shape=(N, K))
    tiles_r, tiles_k = N // TILE_ROWS, K // TILE_K
    tiles = tiles_r * tiles_k
    # Global symbol counts first (the per-tensor tables).
    byte_freq = np.zeros(256, dtype=np.int64)
    for r in range(0, N, 1024):
        byte_freq += np.bincount(np.asarray(w[r:r + 1024]).ravel(), minlength=256)
    exp_freq = np.zeros(16, dtype=np.int64)
    for b in range(256):
        exp_freq[(b >> 3) & 15] += byte_freq[b]
    exp_len = huffman_lengths(exp_freq, 15)
    byte_len = huffman_lengths(byte_freq, 12)
    rcost = rans_cost(byte_freq)
    exp_of = np.array([(b >> 3) & 15 for b in range(256)])
    exp_bits_of = exp_len[exp_of]                          # per byte value
    byte_bits_of = byte_len
    sizes = {k: 0 for k in ("exp-tile", "exp-lane", "exp-warp", "byte-tile", "rans-4", "rans-32")}
    for r0 in range(0, tiles_r, 64):                       # 64 tile rows (1,024 matrix rows) at a time
        blk = np.asarray(w[r0 * TILE_ROWS:(r0 + 64) * TILE_ROWS])
        nr = blk.shape[0] // TILE_ROWS
        eb = exp_bits_of[blk]                              # [rows, K] bits of each value's exponent code
        bb = byte_bits_of[blk]
        rc = rcost[blk]
        # Pieces: 16 consecutive values of one row (a lane's 16-byte load in retile_fp8).
        piece_e = eb.reshape(nr * TILE_ROWS, K // 16, 16).sum(-1)
        sizes["exp-lane"] += int(((piece_e + 7) // 8).sum()) + piece_e.size * 1
        # Tiles: [nr, 16, tiles_k, 128] -> sums per tile.
        tile_e = eb.reshape(nr, TILE_ROWS, tiles_k, TILE_K).sum((1, 3))
        tile_b = bb.reshape(nr, TILE_ROWS, tiles_k, TILE_K).sum((1, 3))
        tile_r = rc.reshape(nr, TILE_ROWS, tiles_k, TILE_K).sum((1, 3))
        sizes["exp-tile"] += int(((tile_e + 7) // 8).sum())
        sizes["byte-tile"] += int(((tile_b + 7) // 8).sum())
        # A warp's share: 32 of a tile's 128 pieces, in the tiled order ((h*2+q)*8+g)*4+t, so h,q pick the warp.
        pe = piece_e.reshape(nr, 2, 8, tiles_k, 4, 2)      # [tile row, h, g, chunk, t, q]
        warp_e = pe.transpose(0, 3, 1, 5, 2, 4).reshape(nr, tiles_k, 4, 32).sum(-1)
        sizes["exp-warp"] += int(((warp_e + 7) // 8).sum()) + warp_e.size * 2
        for S in (4, 32):
            sizes[f"rans-{S}"] += int(np.ceil(tile_r / 8).sum()) + tile_r.size * 4 * S
    raw_plane = tiles * TILE_BYTES // 2                   # sign + mantissa, 4 bits a value
    out = {}
    for k, v in sizes.items():
        b = v + tiles * OFFSET
        if k.startswith("exp"):
            b += raw_plane + 16
        elif k == "byte-tile":
            b += 256
        else:
            b += 512
        out[k] = b
    h_byte = float(-(byte_freq[byte_freq > 0] / byte_freq.sum() * np.log2(byte_freq[byte_freq > 0] / byte_freq.sum())).sum())
    pe_ = exp_freq[exp_freq > 0] / exp_freq.sum()
    h_exp = float(-(pe_ * np.log2(pe_)).sum())
    return dict(name=name, bytes=N * K, h_byte=h_byte, h_exp=h_exp, sizes=out,
                exp_len=exp_len.tolist(), max_byte_len=int(byte_len.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    jobs = []
    for f in sorted(glob.glob(os.path.join(a.snapshot, "*.safetensors"))):
        hd, base = header(f)
        for k, v in hd.items():
            if k != "__metadata__" and v["dtype"] == "F8_E4M3" and k.endswith(".weight") and "visual" not in k:
                jobs.append((k, f, base, v))
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(measure, jobs))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    raw = sum(r["bytes"] for r in rows)
    print(f"{len(rows)} FP8 matrices, {raw / 1e9:.3f} GB; H(byte) {np.average([r['h_byte'] for r in rows], weights=[r['bytes'] for r in rows]):.3f} bits,"
          f" H(exponent) {np.average([r['h_exp'] for r in rows], weights=[r['bytes'] for r in rows]):.3f} bits (bound: {raw * (1 - np.average([r['h_byte'] for r in rows], weights=[r['bytes'] for r in rows]) / 8) / 1e9:.3f} GB off)")
    groups = {}
    for r in rows:
        g = r["name"].split(".")[-2]
        groups.setdefault(g, []).append(r)
    fmts = list(rows[0]["sizes"])
    print(f"{'group':14} {'GB':>7} " + " ".join(f"{f:>10}" for f in fmts))
    for g, rs in sorted(groups.items()):
        b = sum(r["bytes"] for r in rs)
        print(f"{g:14} {b / 1e9:7.3f} " + " ".join(f"{sum(r['sizes'][f] for r in rs) / b:10.4f}" for f in fmts))
    print(f"{'all (ratio)':14} {raw / 1e9:7.3f} " + " ".join(f"{sum(r['sizes'][f] for r in rows) / raw:10.4f}" for f in fmts))
    print(f"{'GB off/step':14} {'':7} " + " ".join(f"{(raw - sum(r['sizes'][f] for r in rows)) / 1e9:10.3f}" for f in fmts))
    print(f"{'ms at 245GB/s':14} {'':7} " + " ".join(f"{(raw - sum(r['sizes'][f] for r in rows)) / 245e6:10.2f}" for f in fmts))


if __name__ == "__main__":
    sys.exit(main())
