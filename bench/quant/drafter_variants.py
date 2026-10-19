#!/usr/bin/env python3
"""Cheaper-drafter probes (drafter/lowp): copies of the DFlash2 drafter in the same format, so the same kernels
load them at the same cost, and only the acceptance changes. What a real kernel would save is bounded from bytes.

    drafter_variants.py SRC_DIR OUT_DIR [--prune24 all|mlp|attn] [--bf16 fp8|nvfp4 --bf16-which fc,kproj,sel]

--prune24: in every NVFP4 projection chosen (all: q k v o gate up down; mlp: gate up down; attn: q k v o), in each
  group of 4 consecutive input-dim weights the 2 of largest |value| are kept and the other two E2M1 codes are set
  to 0 (exact zero). Scales, global scales and input scales are unchanged; the 4 share one 16-block scale, so the
  E2M1 magnitude code orders them as the dequantized values do (first index wins a tie).
--bf16: the BF16 matrices the drafter reads at full width (fc, the conv kernel projections, the selector's
  hidden projection) rounded through FP8 E4M3 (one scale per output row) or NVFP4 (E4M3 per 16, FP32 global) and
  stored back as BF16: the acceptance an FP8 / NVFP4 kernel for them would give, at today's cost.

Runs where torch and safetensors are installed (production's SGLang image on the CPU, as scalesearch.py:
`docker run --rm --network none -v SRC:/src:ro -v OUT:/out -v HERE:/q:ro --entrypoint python3 IMAGE -I
/q/drafter_variants.py /src /out --prune24 all`). Prints per tensor what changed and its squared error.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")


def unpack(b):  # [..., K/2] bytes -> [..., K] codes, first value in the low nibble
    return torch.stack([b & 15, b >> 4], dim=-1).flatten(-2)


def pack(c):
    return c[..., 0::2] | (c[..., 1::2] << 4)


def decode(codes):
    v = E2M1[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -v, v)


def e2m1_codes(x):
    idx = torch.bucketize(x.abs(), BOUNDS, right=False)
    return idx.to(torch.uint8) | ((x < 0) & (idx > 0)).to(torch.uint8) << 3


def prune24(wq, sc, g, rows=1024):
    """wq [N][K/2] packed E2M1, sc [N][K/16] E4M3 codes, g the global scale: the 2:4 copy and its error."""
    N = wq.shape[0]
    out = torch.empty_like(wq)
    e_drop = e_all = 0.0
    nz_drop = 0
    for r0 in range(0, N, rows):
        c = unpack(wq[r0:r0 + rows])  # [n][K]
        n, K = c.shape
        grp = c.reshape(n, K // 4, 4)
        mag = (grp & 7).to(torch.int16)
        # Rank by magnitude, first index winning a tie: key = mag * 4 + (3 - position).
        key = (mag * 4 + torch.tensor([3, 2, 1, 0], dtype=torch.int16)).float()
        top = key.topk(2, dim=-1).indices
        keep = torch.zeros_like(grp, dtype=torch.bool).scatter_(-1, top, True)
        s = (E4M3[sc[r0:r0 + rows].long()] * g).repeat_interleave(4, dim=1).unsqueeze(-1)  # per group of 4
        v = decode(grp) * s
        e_all += (v ** 2).sum().item()
        e_drop += (v[~keep] ** 2).sum().item()
        nz_drop += ((grp & 7) != 0)[~keep].sum().item()
        out[r0:r0 + rows] = pack(torch.where(keep, grp, torch.zeros_like(grp)).reshape(n, K))
    return out, e_drop / e_all, nz_drop / (wq.numel() * 2)


def fake_fp8(w):
    """One E4M3 scale per output row (amax / 448)."""
    w = w.float()
    s = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    return ((w / s).to(torch.float8_e4m3fn).float() * s)


def fake_nvfp4(w):
    """E4M3 per 16 (amax / 6 over the global scale), FP32 global scale amax / (448 * 6)."""
    w = w.float()
    N, K = w.shape
    g = (w.abs().max() / (448.0 * 6.0)).item()
    b = w.reshape(N, K // 16, 16)
    s = (b.abs().amax(-1, keepdim=True) / 6.0 / g).to(torch.float8_e4m3fn).float().clamp(min=2 ** -9) * g
    return (decode(e2m1_codes(b / s)) * s).reshape(N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--prune24", choices=["all", "mlp", "attn"])
    ap.add_argument("--bf16", choices=["fp8", "nvfp4"])
    ap.add_argument("--bf16-which", default="fc,kproj,sel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {}
    with safe_open(os.path.join(a.src, "model.safetensors"), "pt") as f:
        meta = f.metadata()
        for name in f.keys():
            out[name] = f.get_tensor(name)
    tot = []
    if a.prune24:
        mods = {"all": MLP + ATTN, "mlp": MLP, "attn": ATTN}[a.prune24]
        for name in sorted(n for n in out if n.endswith(".weight") and out[n].dtype == torch.uint8):
            base = name[: -len(".weight")]
            if not any(base.endswith(m) for m in mods):
                continue
            sc = out[base + ".weight_scale"].view(torch.uint8)
            g = out[base + ".weight_scale_2"].float().item()
            out[name], rel, frac = prune24(out[name], sc, g)
            tot.append(rel)
            print(f"{base}: 2:4, nonzero codes dropped {frac:.3f} of all; relative squared error {rel:.4f}", flush=True)
    if a.bf16:
        which = set(a.bf16_which.split(","))
        for name in sorted(out):
            t = out[name]
            if t.dtype != torch.bfloat16 or t.dim() != 2 or not name.endswith(".weight"):
                continue
            kind = ("fc" if name == "fc.weight" else "kproj" if "kernel_projection" in name
                    else "sel" if name.startswith("candidate_selector.hidden_projection") else None)
            if kind not in which:
                continue
            q = (fake_fp8 if a.bf16 == "fp8" else fake_nvfp4)(t)
            rel = ((q - t.float()) ** 2).sum().item() / (t.float() ** 2).sum().item()
            out[name] = q.to(torch.bfloat16)
            tot.append(rel)
            print(f"{name}: {a.bf16} round trip, relative squared error {rel:.5f}", flush=True)
    meta = dict(meta or {})
    meta["drafter_variant"] = json.dumps({"prune24": a.prune24, "bf16": a.bf16, "bf16_which": a.bf16_which})
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata=meta)
    for fn in os.listdir(a.src):
        p = os.path.join(a.src, fn)
        if fn != "model.safetensors" and not os.path.isdir(p):
            shutil.copy(os.path.realpath(p), os.path.join(a.out, fn))
    print("written", a.out, "tensors changed", len(tot))


if __name__ == "__main__":
    main()
