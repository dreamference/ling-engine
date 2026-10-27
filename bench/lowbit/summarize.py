"""Composes lowbit_gemm's per-matrix timings into one verify step's weight time per candidate format map.

Part of the low-bit study (specs/DREAMFERENCE_LING_ENGINE_LOWBIT.md section 6). For each shape, format
and row count it takes the fastest tiling (median over the CSV files given), multiplies by the number
of such matrices in one target pass (MODEL section 5.1), and prints, per candidate precision map, the
weight bytes and milliseconds of one verify pass at 16 rows and the saving against today's map
(FFN and LM head NVFP4, every DeltaNet and attention projection FP8).

Usage: python summarize.py lowbit_gemm.csv [lowbit_gemm_2.csv ...] [--rows 16]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict

# Matrices of each shape in one target pass (MODEL section 5.1).
COUNT = {
    "ffn_gate_up": 64,
    "ffn_down": 64,
    "gdn_in": 48,
    "gdn_out": 48,
    "attn_q": 16,
    "attn_kv": 16,
    "attn_o": 16,
    "lm_head": 1,
}
GROUP = {
    "ffn_gate_up": "ffn",
    "ffn_down": "ffn",
    "gdn_in": "proj",
    "gdn_out": "proj",
    "attn_q": "proj",
    "attn_kv": "proj",
    "attn_o": "proj",
    "lm_head": "head",
}
# Candidate precision maps: the format of the FFN, of the DeltaNet and attention projections, of the head.
CANDIDATES = {
    "today (FFN fp4, projections fp8)": {"ffn": "fp4", "proj": "fp8", "head": "fp4"},
    "all-NVFP4 (projections fp4)": {"ffn": "fp4", "proj": "fp4", "head": "fp4"},
    "3-bit FFN, projections fp8": {"ffn": "b3", "proj": "fp8", "head": "fp4"},
    "3-bit FFN, projections fp4": {"ffn": "b3", "proj": "fp4", "head": "fp4"},
    "3-bit FFN and projections": {"ffn": "b3", "proj": "b3", "head": "fp4"},
    "3-bit everything (head too)": {"ffn": "b3", "proj": "b3", "head": "b3"},
    "2:4 NVFP4 FFN, projections fp8": {"ffn": "sp4", "proj": "fp8", "head": "fp4"},
    "2:4 NVFP4 FFN, projections fp4": {"ffn": "sp4", "proj": "fp4", "head": "fp4"},
    "FFN fp4, 2:4 FP8 projections": {"ffn": "fp4", "proj": "sp8", "head": "fp4"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--rows", type=int, default=16)
    args = ap.parse_args()
    best: dict[tuple[str, str], list[float]] = defaultdict(list)
    nbytes: dict[tuple[str, str], int] = {}
    for path in args.csv:
        per_file: dict[tuple[str, str], float] = {}
        with open(path) as fh:
            for row in csv.DictReader(line for line in fh if not line.startswith("#")):
                if int(row["rows"]) != args.rows:
                    continue
                key = (row["shape"], row["fmt"])
                per_file[key] = min(per_file.get(key, 1e9), float(row["ms"]))
                nbytes[key] = int(row["bytes"])
        for key, ms in per_file.items():
            best[key].append(ms)
    ms_of = {key: statistics.median(v) for key, v in best.items()}
    print(f"rows {args.rows}; median over {len(args.csv)} run(s) of each matrix's fastest tiling")
    print(f"{'candidate':36s} {'GB':>6s} {'ms':>7s} {'GB/s':>6s} {'d GB':>6s} {'d ms':>7s}")
    base = None
    for name, fmap in CANDIDATES.items():
        gb = sum(COUNT[s] * nbytes[(s, fmap[GROUP[s]])] for s in COUNT) / 1e9
        ms = sum(COUNT[s] * ms_of[(s, fmap[GROUP[s]])] for s in COUNT)
        if base is None:
            base = (gb, ms)
        print(f"{name:36s} {gb:6.2f} {ms:7.2f} {gb / ms * 1e3:6.0f} {gb - base[0]:6.2f} {ms - base[1]:7.2f}")


if __name__ == "__main__":
    main()
