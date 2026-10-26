"""BACKLOG §19.19 offline: would a sparse drafter head keep the drafter's proposals?

Reads the target's NVFP4 LM head from the checkpoint, factorizes it (rank R, randomized SVD), and replays dumped drafter
steps (LING_DUMP_DRAFT_H: per step E rows of the head's input and their top-16 candidates). For each step the sparse
head scores the rows [fixed list] + [the D rows with the best rank-R score, the maximum over the step's E rows], and a
drafted position keeps its proposal exactly when its full-head top-16 lie inside that set (the exact logits of the
selected rows are the full head's). Reports that rate per position and per step, for several (fixed, dynamic, rank).

Runs in a container with torch and a GPU:  python3 analyze.py CHECKPOINT_DIR DUMP
The dump's first half (the replay's earlier sessions) gives the candidates' frequencies, which make the fixed list;
the second half is evaluated.
"""
import json, os, struct, sys
import numpy as np
import torch

def load_head(ckdir):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckdir, "model.safetensors.index.json")))["weight_map"]
    def get(name):
        with safe_open(os.path.join(ckdir, idx[name]), "pt", device="cuda") as f:
            return f.get_tensor(name)
    w = get("lm_head.weight")            # [N, K/2] uint8, two E2M1 per byte (low nibble first)
    s = get("lm_head.weight_scale")      # [N, K/16] float8_e4m3fn
    s2 = get("lm_head.weight_scale_2").float().item()
    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    N = w.shape[0]
    out = torch.empty(N, w.shape[1] * 2, device="cuda", dtype=torch.float32)
    for r0 in range(0, N, 16384):
        b = w[r0:r0 + 16384].to(torch.int64)
        v = torch.stack([e2m1[b & 15], e2m1[b >> 4]], dim=-1).reshape(b.shape[0], -1)
        sc = s[r0:r0 + 16384].float().repeat_interleave(16, dim=1)
        out[r0:r0 + 16384] = v * sc * s2
    return out

def read_dump(path, limit=None):
    steps = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            E, H, K = struct.unpack("3i", hdr)
            h = np.frombuffer(f.read(4 * E * H), dtype=np.float32).reshape(E, H)
            c = np.frombuffer(f.read(4 * E * K), dtype=np.int32).reshape(E, K)
            steps.append((h, c))
            if limit and len(steps) >= limit:
                break
    return steps

def main():
    ck, dump = sys.argv[1], sys.argv[2]
    W = load_head(ck)
    N = W.shape[0]
    print(f"head {tuple(W.shape)}", flush=True)
    allsteps = read_dump(dump)
    half = len(allsteps) // 2
    freq = np.zeros(N, dtype=np.int64)
    for h, c in allsteps[:half]:
        np.add.at(freq, c.ravel(), 1)
    order = torch.from_numpy(np.argsort(-freq, kind="stable")).cuda()
    print(f"frequency list from the first {half} steps: {int((freq > 0).sum())} tokens seen", flush=True)
    steps = allsteps[half:]
    print(f"{len(steps)} steps to evaluate", flush=True)
    for R in (256, 512):
        torch.manual_seed(0)
        U, S, V = torch.svd_lowrank(W, q=R, niter=3)   # W ~ U diag(S) V^T
        US = (U * S).contiguous()                       # [N, R]
        Vt = V.t().contiguous()                         # [R, K]
        for fixed, dyn in ((32768, 16384), (16384, 32768), (32768, 32768)):
            pos_ok = pos_n = step_ok = 0
            top1 = top4 = early = early_n = 0
            fixed_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
            fixed_mask[order[:fixed]] = True
            for h, c in steps:
                hh = torch.from_numpy(h.copy()).cuda()
                approx = (hh @ Vt.t()) @ US.t()          # [E, N]
                best = approx.max(dim=0).values
                best[fixed_mask] = -float("inf")
                sel = fixed_mask.clone()
                sel[torch.topk(best, dyn).indices] = True
                cc = torch.from_numpy(c.astype(np.int64)).cuda()
                ins = sel[cc]                             # [E, 16]: each candidate selected (descending order)
                inside = ins.all(dim=1)                   # per drafted position: its top-16 all selected
                pos_ok += int(inside.sum()); pos_n += inside.numel(); step_ok += int(inside.all())
                top1 += int(ins[:, 0].sum()); top4 += int(ins[:, :4].all(dim=1).sum())
                early += int(inside[:4].sum()); early_n += min(4, inside.numel())
            print(f"rank {R} fixed {fixed} dynamic {dyn}: positions kept {pos_ok / pos_n:.4f} (first 4 positions "
                  f"{early / early_n:.4f}), top-1 inside {top1 / pos_n:.4f}, top-4 inside {top4 / pos_n:.4f}, whole steps "
                  f"kept {step_ok / len(steps):.4f}; rows read {fixed + dyn} of {N} ({(fixed + dyn) / N:.1%})", flush=True)

if __name__ == "__main__":
    main()
