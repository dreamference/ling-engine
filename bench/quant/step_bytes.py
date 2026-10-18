#!/usr/bin/env python3
"""BACKLOG §19 item 27 (b): every byte one decode step of ling-serve reads, by dtype and by owner.

Weights come from the safetensors headers of the target and the drafter as the engine loads them (the B26 scale
codes and B16's tiling change the layout, not the count, except as noted); the state from the configs and the
engine's buffers (BF16 KV for the target's 16 attention layers and the drafter's ring, FP32 DeltaNet and conv
state read and written once). One sequence, one verify pass of a block of 12. With --entropy JSON files from
code_entropy.py, each row also gets the bytes a per-tensor entropy coder could at most take off. CPU only.

Usage: step_bytes.py TARGET_SNAPSHOT DRAFTER_DIR [--context 25700 65536] [--entropy T.json D.json]
"""
import argparse
import glob
import json
import os
import struct
import sys

DTYPE = {"U8": "NVFP4 codes", "F8_E4M3": "FP8", "BF16": "BF16", "F32": "FP32", "F16": "FP16"}


def tensors(folder):
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            hd = json.loads(h.read(n))
        for k, v in hd.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], v["data_offsets"][1] - v["data_offsets"][0])
    return out


def owner_target(name):
    if "visual" in name or "mtp" in name:
        return None                       # not read in a decode step
    if "embed_tokens" in name:
        return None                       # row lookups only (10 KB per token), counted below
    if "lm_head" in name:
        return "target: LM head"
    if "mlp." in name:
        return "target: FFN"
    if "linear_attn." in name:
        return "target: DeltaNet projections" if "proj" in name else "target: DeltaNet norms, conv, A, dt"
    if "self_attn." in name:
        return "target: attention projections" if "proj." in name else "target: attention norms"
    return "target: norms"


def owner_drafter(name):
    if "codebook" in name:
        return None                       # gathered: E x 16 rows of 256 per codebook, counted below
    if name.startswith("fc."):
        return "drafter: fc (target features)"
    if "kernel_projection" in name or "base_kernel" in name:
        return "drafter: conv kernels"
    if "candidate_selector" in name:
        return "drafter: selector projection"
    if "proj" in name:
        return "drafter: layers"
    return "drafter: norms"


def kind(name, dtype):
    if dtype == "F8_E4M3" and name.endswith("weight_scale"):
        return "NVFP4 block scales (FP8)"
    return DTYPE.get(dtype, dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("drafter")
    ap.add_argument("--context", type=int, nargs="+", default=[25700, 65536])
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--entropy", nargs=2, metavar=("TARGET_JSON", "DRAFTER_JSON"))
    a = ap.parse_args()
    red = {}
    if a.entropy:
        for path, prefix in zip(a.entropy, ("t:", "d:")):
            for r in json.load(open(path)):
                bits = 4 if r["kind"] == "nvfp4" else 8
                red[prefix + r["name"]] = r["redundancy"] / (bits if r["kind"] == "nvfp4" else (8 if r["kind"] == "fp8" else 16))
    rows = {}

    def add(owner, k, nbytes, saved=0.0):
        r = rows.setdefault((owner, k), [0, 0.0])
        r[0] += nbytes
        r[1] += saved

    t = tensors(a.target)
    for name, (dt, nb) in t.items():
        o = owner_target(name)
        if o:
            add(o, kind(name, dt), nb, nb * red.get("t:" + name, 0.0) if not name.endswith("_scale") else 0.0)
            if o == "target: LM head":     # the drafter scores its candidates against the target's head
                add("drafter: scoring on the target's LM head", kind(name, dt), nb,
                    nb * red.get("t:" + name, 0.0) if not name.endswith("_scale") else 0.0)
    d = tensors(a.drafter)
    for name, (dt, nb) in d.items():
        o = owner_drafter(name)
        if o:
            add(o, kind(name, dt), nb, nb * red.get("d:" + name, 0.0) if not name.endswith("_scale") else 0.0)
    tc = json.load(open(os.path.join(a.target, "config.json")))["text_config"]
    dc = json.load(open(os.path.join(a.drafter, "config.json")))
    E = a.block - 1
    add("target: embedding rows", "BF16", (a.block + 1) * tc["hidden_size"] * 2)
    add("drafter: selector codebook rows", "BF16", 2 * E * 16 * 256 * 2)
    n_attn = tc["num_hidden_layers"] // tc["full_attention_interval"]
    n_lin = tc["num_hidden_layers"] - n_attn
    state = n_lin * tc["linear_num_value_heads"] * tc["linear_key_head_dim"] * tc["linear_value_head_dim"] * 4
    conv_ch = 2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"] + tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
    conv = n_lin * conv_ch * (tc["linear_conv_kernel_dim"] - 1) * 4
    add("state: DeltaNet recurrent (read + write)", "FP32", 2 * state)
    add("state: conv window (read + write)", "FP32", 2 * conv)
    ring = dc["sliding_window"]
    add("state: drafter KV ring", "BF16", dc["num_hidden_layers"] * 2 * dc["num_key_value_heads"] * dc["head_dim"] * 2 * ring)
    weights_total = sum(v[0] for v in rows.values())
    kv_per_token = n_attn * 2 * tc["num_key_value_heads"] * tc["head_dim"] * 2

    print(f"{'owner':46} {'dtype':26} {'MB':>9} {'%':>6} {'entropy-coder MB':>17}")
    for (o, k), (nb, saved) in sorted(rows.items(), key=lambda x: (x[0][0].split(':')[0], -x[1][0])):
        print(f"{o:46} {k:26} {nb / 1e6:9.1f} {100 * nb / weights_total:6.2f} {saved / 1e6:17.1f}")
    by = {}
    for (o, k), (nb, saved) in rows.items():
        b = by.setdefault(k, [0, 0.0]); b[0] += nb; b[1] += saved
    print("\nby dtype, without the target's KV:")
    for k, (nb, saved) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"  {k:26} {nb / 1e9:7.3f} GB {100 * nb / weights_total:6.2f}%   at most {saved / 1e9:.3f} GB off")
    print(f"  {'total':26} {weights_total / 1e9:7.3f} GB")
    for ctx in a.context:
        kv = kv_per_token * ctx
        print(f"target KV at {ctx} tokens (BF16, {kv_per_token // 1024} KB a token): {kv / 1e9:.3f} GB;"
              f" step total {(weights_total + kv) / 1e9:.3f} GB, BF16 {100 * (by['BF16'][0] + kv) / (weights_total + kv):.1f}%"
              f", FP8 {100 * by['FP8'][0] / (weights_total + kv):.1f}%")


if __name__ == "__main__":
    sys.exit(main())
