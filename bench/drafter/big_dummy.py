"""Writes a bigger DFlash2 drafter with dummy weights, for timing only (BIG_DRAFTER spec, item 2).

The source is a real DFlash2 checkpoint, NVFP4 (maurienne-ai/...-NVFP4-RTNcal: U8 codes, F8_E4M3 scales, F32 scalars)
or BF16 (incoai/Qwen3.8-27B-DFlash2). The output keeps every shape the engine pins (hidden 5120, the target's
embedding, head, fc and selector) and changes only the two axes a bigger block drafter of this kind can grow along:

  --layers L        layer i is a copy of the source's layer i % 5
  --intermediate I  the MLP width: gate_proj and up_proj rows, and down_proj columns, repeated cyclically up to I

The values are real weights rearranged, so every activation stays finite, but the drafter proposes nonsense:
acceptance collapses to about one token per step. The draft's cost does not depend on its proposals (a fixed block
of rows, attention over a window capped at 2048 keys), so ms per draft is what this measures, nothing else.

Pure Python (node-b's host Python has no numpy): the safetensors file is written from byte slices of the source.

Usage: big_dummy.py SRC_DIR OUT_DIR --layers L --intermediate I
"""
import argparse
import json
import mmap
import os
import shutil
import struct

ELEM = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}


def read_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return header, mm, 8 + n


def k_per_byte(dtype, name):
    """Elements of the logical K per stored column: NVFP4 codes pack two per byte, scales cover 16."""
    if name.endswith(".weight_scale"):
        return 16
    if dtype == "U8":
        return 2
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--intermediate", type=int, required=True)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    L0, I0 = cfg["num_hidden_layers"], cfg["intermediate_size"]
    I = a.intermediate
    if I % 1024:
        raise SystemExit("--intermediate must be a multiple of 1024 (the BF16 rows path's K)")
    header, mm, base = read_header(os.path.join(a.src, "model.safetensors"))
    header.pop("__metadata__", None)

    # The output tensors in a fixed order: (name, dtype, shape, producer of its bytes).
    plan = []

    def src_bytes(name):
        t = header[name]
        b, e = t["data_offsets"]
        return mm[base + b:base + e]

    def rows_tiled(name, new_rows):
        t = header[name]
        rows = t["shape"][0]
        row_bytes = (t["data_offsets"][1] - t["data_offsets"][0]) // rows

        def produce(w):
            raw = src_bytes(name)
            left = new_rows
            while left > 0:
                take = min(left, rows)
                w.write(raw[:take * row_bytes])
                left -= take
        return [new_rows] + t["shape"][1:], produce

    def cols_tiled(name, new_k):
        t = header[name]
        rows, cols = t["shape"]
        per = k_per_byte(t["dtype"], name)
        new_cols = new_k // per
        esz = ELEM[t["dtype"]]
        row_bytes, new_row_bytes = cols * esz, new_cols * esz

        def produce(w):
            raw = src_bytes(name)
            reps = -(-new_row_bytes // row_bytes)
            for r in range(rows):
                row = raw[r * row_bytes:(r + 1) * row_bytes]
                w.write((row * reps)[:new_row_bytes])
        return [rows, new_cols], produce

    def copied(name):
        t = header[name]
        return t["shape"], lambda w: w.write(src_bytes(name))

    for name in sorted(header):
        if not name.startswith("layers."):
            plan.append((name, header[name]["dtype"]) + tuple(copied(name)))
    for i in range(a.layers):
        j = i % L0
        prefix_src, prefix_out = f"layers.{j}.", f"layers.{i}."
        for name in sorted(n for n in header if n.startswith(prefix_src)):
            out = prefix_out + name[len(prefix_src):]
            t = header[name]
            is_mlp_in = ".mlp.gate_proj." in name or ".mlp.up_proj." in name
            is_mlp_out = ".mlp.down_proj." in name
            if t["shape"] and I != I0 and is_mlp_in and len(t["shape"]) == 2:
                shape, prod = rows_tiled(name, I)
            elif t["shape"] and I != I0 and is_mlp_out and len(t["shape"]) == 2:
                shape, prod = cols_tiled(name, I)
            else:
                shape, prod = copied(name)
            plan.append((out, t["dtype"], shape, prod))

    out_header, off = {}, 0
    for name, dtype, shape, _ in plan:
        n = ELEM[dtype]
        for s in shape:
            n *= s
        out_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    out_header["__metadata__"] = {"format": "pt", "producer": f"big_dummy.py layers={a.layers} intermediate={I}"}
    hj = json.dumps(out_header, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)

    os.makedirs(a.out, exist_ok=True)
    cfg["num_hidden_layers"] = a.layers
    cfg["max_window_layers"] = a.layers
    cfg["layer_types"] = ["sliding_attention"] * a.layers
    cfg["intermediate_size"] = I
    q = cfg.get("quantization_config")
    if q:  # the excluded (BF16) modules of the new layers
        ex = [m for m in q.get("exclude_modules", []) if not m.startswith("layers.")]
        for i in range(a.layers):
            ex += [f"layers.{i}.attention_conv.kernel_projection", f"layers.{i}.mlp_conv.kernel_projection"]
        q["exclude_modules"] = ex
        q["ignore"] = list(ex)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    hq = os.path.join(a.src, "hf_quant_config.json")
    if os.path.exists(hq):
        shutil.copy(hq, os.path.join(a.out, "hf_quant_config.json"))
    with open(os.path.join(a.out, "model.safetensors"), "wb") as w:
        w.write(struct.pack("<Q", len(hj)))
        w.write(hj)
        for name, dtype, shape, prod in plan:
            start = w.tell()
            prod(w)
            want = out_header[name]["data_offsets"][1] - out_header[name]["data_offsets"][0]
            if w.tell() - start != want:
                raise SystemExit(f"{name}: wrote {w.tell() - start} bytes, expected {want}")
    params = sum(
        (lambda t: (t["data_offsets"][1] - t["data_offsets"][0]))(out_header[n]) for n in out_header if n != "__metadata__")
    print(f"{a.out}: layers {a.layers}, intermediate {I}, {params / 1e9:.2f} GB on disk")


if __name__ == "__main__":
    main()
