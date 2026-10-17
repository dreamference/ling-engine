#!/usr/bin/env python3
"""ScaleSearch for the DFlash2 drafter's NVFP4 weights (BACKLOG §19.22): each 16-value block's FP8 E4M3 scale
is replaced by the neighbouring representable scale (code offsets -2 to +6 around the served one) that gives
the block the lowest squared error against the BF16 original; the values are re-rounded to E2M1 under it.
Same format, same tensors, same per-tensor global scales and activation scales, so the same kernels read it at
the same cost.

    scalesearch.py BF16_DIR NVFP4_DIR OUT_DIR

Runs anywhere torch and safetensors are installed (production's SGLang image, on the CPU:
`docker run --rm --network none -v ...:ro --entrypoint python3 IMAGE -I scalesearch.py ...`). Before changing
anything it checks that re-rounding the BF16 weights with the served scales reproduces the served bytes
(the layout: first value in the low nibble, sign in bit 3), and prints per tensor the squared error of the
served weights and of the searched ones against BF16.
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
OFFSETS = [0, -2, -1, 1, 2, 3, 4, 5, 6]  # the served scale first: a tie keeps it
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()


def e2m1_codes(x):
    """Round to the nearest E2M1 value (ties down, saturating at 6): 4-bit codes, sign in bit 3."""
    idx = torch.bucketize(x.abs(), BOUNDS, right=False)
    return idx.to(torch.uint8) | ((x < 0) & (idx > 0)).to(torch.uint8) << 3


def decode(codes):
    v = E2M1[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -v, v)


def pack(codes):  # [..., K] 4-bit codes -> [..., K/2] bytes, first value in the low nibble
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack(b):
    return torch.stack([b & 15, b >> 4], dim=-1).flatten(-2)


def search(w, code0, g, rows=512):
    """w [N][K] float32 (BF16 original), code0 [N][K/16] uint8 (served E4M3 codes), g the global scale."""
    N, K = w.shape
    best_codes = torch.empty_like(code0)
    best_q = torch.empty(N, K, dtype=torch.uint8)
    err_served = err_best = 0.0
    for r0 in range(0, N, rows):
        wb = w[r0:r0 + rows].reshape(-1, K // 16, 16)
        c0 = code0[r0:r0 + rows].to(torch.int16)
        best_e = None
        for off in OFFSETS:
            c = (c0 + off).clamp(1, 126)  # positive, finite E4M3 codes (0x7f is NaN)
            s = E4M3[c.long()].unsqueeze(-1) * g
            q = e2m1_codes(wb / s)
            e = ((decode(q) * s - wb) ** 2).sum(-1)
            if off == 0:
                err_served += e.sum().item()
                best_e, bc, bq = e, c.to(torch.uint8), q
            else:
                better = e < best_e
                best_e = torch.where(better, e, best_e)
                bc = torch.where(better, c.to(torch.uint8), bc)
                bq = torch.where(better.unsqueeze(-1), q, bq)
        err_best += best_e.sum().item()
        best_codes[r0:r0 + rows] = bc
        best_q[r0:r0 + rows] = bq.reshape(-1, K)
    return best_codes, pack(best_q), err_served, err_best


def main():
    bf16_dir, fp4_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    meta = None
    with safe_open(os.path.join(fp4_dir, "model.safetensors"), "pt") as f4, \
            safe_open(os.path.join(bf16_dir, "model.safetensors"), "pt") as fb:
        meta = f4.metadata()
        names = list(f4.keys())
        for name in names:
            out[name] = f4.get_tensor(name)
        tot_s = tot_b = 0.0
        for name in sorted(n for n in names if n.endswith(".weight_scale")):
            base = name[: -len(".weight_scale")]
            w = fb.get_tensor(base + ".weight").float()
            g = out[base + ".weight_scale_2"].float().item()
            code0 = out[name].view(torch.uint8)
            served = out[base + ".weight"]
            # The layout check: the served scales and RTN reproduce the served bytes.
            s = E4M3[code0.long()].repeat_interleave(16, dim=1) * g
            same = (pack(e2m1_codes(w / s)) == served).float().mean().item()
            if same < 0.999:
                raise SystemExit(f"{base}: re-rounding with the served scales matches only {same:.4f} of the bytes")
            codes, q, es, eb = search(w, code0, g)
            changed = (codes != code0).float().mean().item()
            out[name] = codes.view(torch.float8_e4m3fn)
            out[base + ".weight"] = q
            tot_s += es
            tot_b += eb
            print(f"{base}: bytes reproduced {same:.5f}; scales changed {changed:.3f}; squared error {es:.4e} -> {eb:.4e} "
                  f"({100 * (1 - eb / es):.1f}% less)", flush=True)
        print(f"all: squared error {tot_s:.4e} -> {tot_b:.4e} ({100 * (1 - tot_b / tot_s):.1f}% less)")
    save_file(out, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    for fn in os.listdir(fp4_dir):
        if fn != "model.safetensors" and not os.path.isdir(os.path.join(fp4_dir, fn)):
            shutil.copy(os.path.realpath(os.path.join(fp4_dir, fn)), os.path.join(out_dir, fn))
    print("written", out_dir)


if __name__ == "__main__":
    main()
