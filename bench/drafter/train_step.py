"""Times one training step of a DFlash2-shaped drafter with dummy data (BIG_DRAFTER spec, item 5).

Not a trainer: a stand-in with the drafter's shapes and the costs that matter, so the step time on a GB10 can be
measured instead of read off a datasheet. Per layer: RMSNorm, attention (32 query heads, 8 KV heads, head dim 128)
and a SwiGLU MLP of width --intermediate; in front, the fc over the five target layers' features (25,600 -> 5,120);
behind, the target's frozen LM head (248,320 x 5,120, no weight gradient) and a cross-entropy over the full
vocabulary, chunked. The two DFlash2 grouped convolutions and the selector are left out (under 3% of the FLOPs).
Attention is causal SDPA over the sequence, a stand-in for the block-diffusion mask (same order of FLOPs at a
2,048-key window). BF16 weights with FP32 AdamW states (torch.optim.AdamW, fused), gradient checkpointing per layer.

Also times a plain BF16 GEMM (8192^3) for the machine's reachable dense rate.

Run inside the SGLang image (torch with SM121): docker run --rm --gpus all -v DIR:/w IMAGE python3 /w/train_step.py
Usage: train_step.py [--layers 5,10] [--intermediate 17408] [--seq 4096] [--batch 2] [--steps 6]
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

H, HEADS, KV, HD, V, FEAT = 5120, 32, 8, 128, 248320, 5


class Layer(torch.nn.Module):
    def __init__(self, inter):
        super().__init__()
        self.n1 = torch.nn.RMSNorm(H)
        self.n2 = torch.nn.RMSNorm(H)
        self.q = torch.nn.Linear(H, HEADS * HD, bias=False)
        self.k = torch.nn.Linear(H, KV * HD, bias=False)
        self.v = torch.nn.Linear(H, KV * HD, bias=False)
        self.o = torch.nn.Linear(HEADS * HD, H, bias=False)
        self.gate = torch.nn.Linear(H, inter, bias=False)
        self.up = torch.nn.Linear(H, inter, bias=False)
        self.down = torch.nn.Linear(inter, H, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        h = self.n1(x)
        q = self.q(h).view(b, s, HEADS, HD).transpose(1, 2)
        k = self.k(h).view(b, s, KV, HD).transpose(1, 2)
        v = self.v(h).view(b, s, KV, HD).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, s, HEADS * HD))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Drafter(torch.nn.Module):
    def __init__(self, layers, inter):
        super().__init__()
        self.fc = torch.nn.Linear(FEAT * H, H, bias=False)
        self.layers = torch.nn.ModuleList(Layer(inter) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(H)

    def forward(self, feats):
        x = self.fc(feats)
        for L in self.layers:
            x = checkpoint(L, x, use_reentrant=False)
        return self.norm(x)


def chunked_ce(h, head, labels, chunk=4096):
    """Cross-entropy over the full vocabulary through a frozen head, a chunk of tokens at a time."""
    h = h.reshape(-1, H)
    labels = labels.reshape(-1)
    total = 0.0
    for i in range(0, h.shape[0], chunk):
        logits = F.linear(h[i:i + chunk], head).float()
        total = total + F.cross_entropy(logits, labels[i:i + chunk], reduction="sum")
    return total / h.shape[0]


def flops_per_token(layers, inter, seq):
    """Training FLOPs per token: 6 x the trainable matrices, 4 x the frozen head (forward + input gradient), and the
    attention's QK^T and AV (causal: half of seq keys on average) forward and backward, recomputation excluded."""
    lin = FEAT * H * H + layers * (H * (HEADS + 2 * KV) * HD + HEADS * HD * H + 3 * H * inter)
    attn = layers * 2 * 2 * HEADS * HD * (seq / 2) * 3
    return 6 * lin + 4 * V * H + attn


def gemm_rate():
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.time()
    n = 20
    for _ in range(n):
        a @ b
    torch.cuda.synchronize()
    return 2 * 8192 ** 3 * n / (time.time() - t) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="5,10")
    ap.add_argument("--intermediate", type=int, default=17408)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    torch.manual_seed(0)
    print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "bf16_gemm_tflops": round(gemm_rate(), 1)}), flush=True)
    head = (torch.randn(V, H, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(False)
    for nl in [int(x) for x in a.layers.split(",")]:
        model = Drafter(nl, a.intermediate).cuda().to(torch.bfloat16)
        params = sum(p.numel() for p in model.parameters())
        # FP32 master weights live in the optimizer's states only through fused AdamW on BF16 params; the memory of a
        # mixed-precision run with FP32 masters is reported from the parameter count instead (16 bytes per param).
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        feats = torch.randn(a.batch, a.seq, FEAT * H, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, V, (a.batch, a.seq), device="cuda")
        times = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.steps):
            torch.cuda.synchronize()
            t = time.time()
            loss = chunked_ce(model(feats), head, labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t)
        st = sorted(times[1:])[len(times[1:]) // 2]
        tok = a.batch * a.seq
        fpt = flops_per_token(nl, a.intermediate, a.seq)
        print(json.dumps({"layers": nl, "intermediate": a.intermediate, "params_b": round(params / 1e9, 3),
                          "tokens_per_step": tok, "step_s_median": round(st, 3), "tokens_per_s": round(tok / st),
                          "gflop_per_token": round(fpt / 1e9, 2),
                          "achieved_tflops_excl_recompute": round(fpt * tok / st / 1e12, 1),
                          "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                          "steps_s": [round(x, 3) for x in times]}), flush=True)
        del model, opt, feats, labels
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
