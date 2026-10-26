"""Reference dumps for `ling-run --vision-check` (VISION §20.5 items 2 and 3, §20.8 window 1).

Runs inside production's image (lmsysorg/sglang v0.5.21, Mightling's pin) on the GPU, with the checkpoint mounted:

  python3 reference.py MODEL_DIR OUT_DIR IMAGE...

For each image NAME.EXT it writes OUT_DIR/NAME.json and:
  NAME.decoded.u8        the RGB pixels production's loader gives (nvJPEG on the GPU for a JPEG, Pillow otherwise)
  NAME.resized.u8        the processor's resize of them ([3][h][w], torchvision on the GPU, antialiased bicubic)
  NAME.pixel_values.bf16 the processor's pixel_values cast to BF16 ([patches][1536], as the tower's conv receives them)
  NAME.hf_fp32.f32       transformers' tower in FP32 on those values (the truth the tolerances are measured against)
  NAME.hf_bf16.bf16      the same tower in BF16 (the reference's own noise)
  NAME.hfprodpos.bf16    the BF16 tower with production's position embedding (BF16 weights, BF16 products)
  NAME.sgl.bf16          SGLang's own tower module, when it can be built outside a server
"""
import json
import os
import sys
import traceback

import numpy as np
import torch
from safetensors import safe_open


def dump(path, t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t.view(torch.int16).numpy().tofile(path)
    elif t.dtype == torch.uint8:
        t.numpy().tofile(path)
    else:
        t.float().numpy().tofile(path)


def load_visual(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for name, shard in idx.items():
        if name.startswith("model.visual."):
            with safe_open(os.path.join(model_dir, shard), "pt") as f:
                sd[name[len("model.visual."):]] = f.get_tensor(name)
    return sd


def hf_tower(model_dir, dtype):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = AutoConfig.from_pretrained(model_dir).vision_config
    cfg._attn_implementation = "sdpa"
    with torch.device("cuda"):
        m = Qwen3_5VisionModel(cfg).to(dtype)
    missing, unexpected = m.load_state_dict({k: v.to(dtype) for k, v in load_visual(model_dir).items()}, strict=False)
    assert not unexpected, unexpected
    assert all("rotary" in k for k in missing), missing
    return m.eval()


def prod_pos_embed(m, grid):
    """production's fast_pos_embed_interpolate_from_list (qwen3_vl.py): BF16 weights, BF16 products, one BF16 sum."""
    side = m.num_grid_per_side
    t, h, w = grid
    hi = torch.linspace(0, side - 1, h, dtype=torch.float32, device="cuda")
    wi = torch.linspace(0, side - 1, w, dtype=torch.float32, device="cuda")
    hf, wf = hi.to(torch.long), wi.to(torch.long)
    hc, wc = torch.clamp(hf + 1, max=side - 1), torch.clamp(wf + 1, max=side - 1)
    dh, dw = hi - hf, wi - wf
    dhg, dwg = torch.meshgrid(dh, dw, indexing="ij")
    hfg, wfg = torch.meshgrid(hf, wf, indexing="ij")
    hcg, wcg = torch.meshgrid(hc, wc, indexing="ij")
    w11 = dhg * dwg
    w10 = dhg - w11
    w01 = dwg - w11
    w00 = 1 - dhg - w01
    hg = torch.stack([hfg, hfg, hcg, hcg])
    wg = torch.stack([wfg, wcg, wfg, wcg])
    idx = (hg * side + wg).reshape(4, -1)
    wts = torch.stack([w00, w01, w10, w11], 0).reshape(4, -1, 1).to(torch.bfloat16)
    e = m.pos_embed(idx)
    e *= wts
    comb = e.sum(0)
    ms = 2
    comb = comb.reshape(h // ms, ms, w // ms, ms, -1).permute(0, 2, 1, 3, 4).reshape(-1, comb.shape[-1])
    return comb


def hf_forward_prodpos(m, pv, grid_thw):
    """transformers' tower with production's position embedding."""
    from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids

    x = m.patch_embed(pv)
    x = x + prod_pos_embed(m, grid_thw[0].tolist())
    pos = get_vision_position_ids(grid_thw, m.spatial_merge_size)
    rot = m.rotary_pos_emb(pos).reshape(x.shape[0], -1)
    emb = torch.cat((rot, rot), -1)
    pe = (emb.cos(), emb.sin())
    cu = get_vision_cu_seqlens(grid_thw)
    for blk in m.blocks:
        x = blk(x, cu_seqlens=cu, position_embeddings=pe)
    return m.merger(x)


def sgl_tower(model_dir):
    """SGLang's Qwen3-VL tower outside a server: one process group of one rank and default server arguments."""
    import torch.distributed as dist
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.server_args import ServerArgs

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(ServerArgs(model_path=model_dir))
    except Exception:
        pass
    if not dist.is_initialized():
        ps.init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29571",
                                        backend="nccl")
        ps.initialize_model_parallel(1, 1)
    from transformers import AutoConfig
    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

    cfg = AutoConfig.from_pretrained(model_dir).vision_config
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        m = Qwen3VLMoeVisionModel(cfg, norm_eps=1e-6, prefix="visual")
    torch.set_default_dtype(torch.float32)
    sd = load_visual(model_dir)
    m.load_weights(list(sd.items()))
    return m.eval()


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    from transformers import AutoImageProcessor
    from sglang.srt.utils.common import load_image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    import torchvision.transforms.v2.functional as tvF

    proc = AutoImageProcessor.from_pretrained(model_dir)
    print("processor:", type(proc).__name__, proc.size, flush=True)
    m32 = hf_tower(model_dir, torch.float32)
    m16 = hf_tower(model_dir, torch.bfloat16)
    try:
        sgl = sgl_tower(model_dir)
    except Exception:
        traceback.print_exc()
        print("SGLang's tower could not be built outside a server; only transformers' references", flush=True)
        sgl = None
    for path in sys.argv[3:]:
        name = os.path.splitext(os.path.basename(path))[0]
        data = open(path, "rb").read()
        img, _ = load_image(data, True)  # production's loader: nvJPEG for a JPEG
        if isinstance(img, torch.Tensor):
            chw = img if img.shape[0] == 3 else img.expand(3, -1, -1)
            loader = "nvjpeg"
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
            chw = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).cuda()
            loader = "pil"
        h, w = chw.shape[1:]
        dump(f"{out}/{name}.decoded.u8", chw.permute(1, 2, 0))
        nh, nw = smart_resize(h, w, factor=32, min_pixels=proc.size["shortest_edge"], max_pixels=proc.size["longest_edge"])
        resized = tvF.resize(chw.unsqueeze(0), [nh, nw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)[0]
        dump(f"{out}/{name}.resized.u8", resized)
        feats = proc(images=[img if loader == "pil" else chw], return_tensors="pt", device="cuda")
        pv, grid = feats["pixel_values"].cuda(), feats["image_grid_thw"].cuda()
        dump(f"{out}/{name}.pixel_values.bf16", pv.to(torch.bfloat16))
        pvb = pv.to(torch.bfloat16)
        with torch.no_grad():
            dump(f"{out}/{name}.hf_fp32.f32", m32(pvb.float(), grid).pooler_output)
            dump(f"{out}/{name}.hf_bf16.bf16", m16(pvb, grid).pooler_output)
            dump(f"{out}/{name}.hfprodpos.bf16", hf_forward_prodpos(m16, pvb, grid))
            if sgl is not None:
                dump(f"{out}/{name}.sgl.bf16", sgl(pvb, grid))
        meta = {"image": os.path.abspath(path), "loader": loader, "decoded": [int(h), int(w)], "resized": [nh, nw],
                "grid": grid[0].tolist(), "tokens": int(grid[0].prod().item() // 4)}
        json.dump(meta, open(f"{out}/{name}.json", "w"))
        print(name, meta, flush=True)


if __name__ == "__main__":
    main()
