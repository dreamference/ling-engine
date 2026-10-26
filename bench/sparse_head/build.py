"""BACKLOG §19.19: writes the drafter's sparse head (the files ling-serve --draft-head sparse --sparse-head-dir reads).

  us     BF16 [vocab][R]   the target LM head's rank-R left factor times its singular values (randomized SVD of the
                           dequantized NVFP4 head)
  vt     BF16 [R][hidden]  the right factor
  fixed  I32 [F]           the rows always scored: the F most frequent of the drafter's top-16 candidates in a dump
                           (LING_DUMP_DRAFT_H), ties to the lower id

Runs where torch and a GPU are (the SGLang image):  python3 build.py CHECKPOINT_DIR DUMP OUT_DIR [R] [F] [STEPS]
STEPS: use the dump's first STEPS steps only (default: its first half, the analysis's frequency half).
"""
import os, sys
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_head, read_dump  # noqa: E402

def main():
    ck, dump, out = sys.argv[1], sys.argv[2], sys.argv[3]
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    F = int(sys.argv[5]) if len(sys.argv) > 5 else 32768
    W = load_head(ck)
    N = W.shape[0]
    steps = read_dump(dump)
    use = int(sys.argv[6]) if len(sys.argv) > 6 else len(steps) // 2
    freq = np.zeros(N, dtype=np.int64)
    for h, c in steps[:use]:
        np.add.at(freq, c.ravel(), 1)
    fixed = np.sort(np.argsort(-freq, kind="stable")[:F]).astype(np.int32)
    torch.manual_seed(0)
    U, S, V = torch.svd_lowrank(W, q=R, niter=3)
    os.makedirs(out, exist_ok=True)
    save_file({"us": (U * S).to(torch.bfloat16).contiguous().cpu(), "vt": V.t().to(torch.bfloat16).contiguous().cpu(),
               "fixed": torch.from_numpy(fixed)}, os.path.join(out, "model.safetensors"),
              metadata={"rank": str(R), "fixed": str(F), "frequency_steps": str(use), "dump": os.path.basename(dump)})
    print(f"wrote {out}: rank {R}, {F} fixed rows from {use} steps' candidates ({int((freq > 0).sum())} tokens seen)")

if __name__ == "__main__":
    main()
