#!/usr/bin/env python3
"""Writes a PEFT LoRA adapter for the gates of LORA §6, shaped from the model's config.json (numpy only).

    make_adapter.py MODEL_DIR OUT_DIR [--rank 16] [--alpha 32] [--zero] [--b-std 0.002] [--seed 1]
                    [--targets attn,gdn] [--drafter DRAFT_DIR]

--zero writes every B as zeros (the zero adapter: output identical to the base); otherwise A ~ N(0, 1/K) and
B ~ N(0, b_std), a small perturbation that changes some greedy tokens. `attn` is q, k, v and o on the full-attention
layers, `gdn` in_proj_qkv, in_proj_z and out_proj on the DeltaNet layers, `gdn-split` the same with in_proj_qkv as
in_proj_q, in_proj_k and in_proj_v (the only form SGLang v0.5.19 serves beside in_proj_z: it takes the four split
projections or one merged in_proj_qkvz). --drafter adds a drafter/ part (q, k, v, o
on every drafter layer).
"""

import argparse
import json
import os
import struct

import numpy as np


def bf16_bytes(x):
    """Rounds float32 to BF16 (to nearest even) and returns the raw little-endian bytes."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype("<u2").tobytes()


def write_safetensors(path, tensors):
    """Writes {name: float32 array} as BF16 safetensors."""
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        b = bf16_bytes(a)
        header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def modules(cfg, targets, drafter):
    """Yields (layer, group, module, out, in) for every projection the adapter targets."""
    t = cfg.get("text_config", cfg)
    H, heads, kv, hd = t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    q_out = heads * hd * (2 if t.get("attn_output_gate") and not drafter else 1)
    n = t["num_hidden_layers"]
    types = t.get("layer_types") or ["full_attention"] * n
    for layer in range(n):
        if types[layer] == "full_attention" or drafter:
            if "attn" in targets or drafter:
                yield layer, "self_attn", "q_proj", q_out, H
                yield layer, "self_attn", "k_proj", kv * hd, H
                yield layer, "self_attn", "v_proj", kv * hd, H
                yield layer, "self_attn", "o_proj", H, heads * hd
        elif "gdn" in targets or "gdn-split" in targets:
            qk = t["linear_num_key_heads"] * t["linear_key_head_dim"]
            v = t["linear_num_value_heads"] * t["linear_value_head_dim"]
            if "gdn-split" in targets:
                yield layer, "linear_attn", "in_proj_q", qk, H
                yield layer, "linear_attn", "in_proj_k", qk, H
                yield layer, "linear_attn", "in_proj_v", v, H
            else:
                yield layer, "linear_attn", "in_proj_qkv", 2 * qk + v, H
            yield layer, "linear_attn", "in_proj_z", v, H
            yield layer, "linear_attn", "out_proj", H, v


def write_adapter(model_dir, out, args, targets, drafter, rng):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    os.makedirs(out, exist_ok=True)
    tensors, names = {}, set()
    for layer, group, mod, n_out, n_in in modules(cfg, targets, drafter):
        key = f"base_model.model.model.layers.{layer}.{group}.{mod}"
        tensors[key + ".lora_A.weight"] = rng.normal(0.0, 1.0 / np.sqrt(n_in), (args.rank, n_in)).astype(np.float32)
        b = np.zeros((n_out, args.rank), np.float32) if args.zero else rng.normal(0.0, args.b_std, (n_out, args.rank))
        tensors[key + ".lora_B.weight"] = b.astype(np.float32)
        names.add(mod)
    write_safetensors(os.path.join(out, "adapter_model.safetensors"), tensors)
    conf = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        # SGLang names the split projections' module in_proj_qkvz and wraps only the modules target_modules names:
        # "all-linear" with --lora-target-modules all wraps them all (the engine reads the keys, not this list).
        "target_modules": "all-linear" if "gdn-split" in targets and not drafter else sorted(names),
        "base_model_name_or_path": os.path.basename(os.path.normpath(model_dir)),
    }
    json.dump(conf, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    print(f"{out}: {len(tensors) // 2} modules ({', '.join(sorted(names))}), rank {args.rank}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_dir")
    p.add_argument("out_dir")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--zero", action="store_true")
    p.add_argument("--b-std", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--targets", default="attn,gdn")
    p.add_argument("--drafter")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    targets = set(args.targets.split(","))
    write_adapter(args.model_dir, args.out_dir, args, targets, False, rng)
    if args.drafter:
        write_adapter(args.drafter, os.path.join(args.out_dir, "drafter"), args, targets, True, rng)


if __name__ == "__main__":
    main()
