"""Add NVTX ranges to SGLang 0.5.19's DFLASH worker, for Nsight Systems (M0).

Writes a patched copy of speculative/dflash_worker_v2.py that is bind-mounted over the original
in a scratch container. Every range is a torch.cuda.nvtx push/pop on the host; nsys projects the
GPU work each range launched onto it (`nsys stats -r nvtx_gpu_proj_sum`), so the phases of one
speculative step can be timed on the GPU even inside CUDA graphs.

Usage: python3 patch_dflash_nvtx.py ORIGINAL.py PATCHED.py
"""
import sys

src = open(sys.argv[1]).read()

# (anchor, replacement) pairs; every anchor must occur exactly once.
EDITS = [
    # Draft forward (the drafter's graph, including the selector's top-k over the target head).
    ("        with torch.inference_mode():\n            draft_out = self.draft_model_runner.forward(forward_batch)\n",
     "        torch.cuda.nvtx.range_push('m0.draft')\n"
     "        with torch.inference_mode():\n            draft_out = self.draft_model_runner.forward(forward_batch)\n"),
    ("        draft_tokens = self._draft_block_tokens_buf[:bs]\n        draft_tokens[:, 0].copy_(block_ids[:, 0])\n",
     "        torch.cuda.nvtx.range_pop()\n"
     "        draft_tokens = self._draft_block_tokens_buf[:bs]\n        draft_tokens[:, 0].copy_(block_ids[:, 0])\n"),
    # Target verify forward.
    ("        target_out = self.target_worker.forward_batch_generation(\n            batch=None,\n",
     "        torch.cuda.nvtx.range_push('m0.verify')\n"
     "        target_out = self.target_worker.forward_batch_generation(\n            batch=None,\n"),
    ("        logits_output = target_out.logits_output\n        can_run_cuda_graph = target_out.can_run_cuda_graph\n",
     "        logits_output = target_out.logits_output\n        can_run_cuda_graph = target_out.can_run_cuda_graph\n"
     "        torch.cuda.nvtx.range_pop()\n        torch.cuda.nvtx.range_push('m0.accept')\n"),
    # Accept ends where the DeltaNet commit starts.
    ("        if self._need_mamba_verify_commit:\n            assert seq_lens_pre_verify is not None\n",
     "        torch.cuda.nvtx.range_pop()\n        torch.cuda.nvtx.range_push('m0.mamba_commit')\n"
     "        if self._need_mamba_verify_commit:\n            assert seq_lens_pre_verify is not None\n"),
    ("        # --- 3) Materialize committed verify-input tokens into draft KV cache.\n",
     "        torch.cuda.nvtx.range_pop()\n        torch.cuda.nvtx.range_push('m0.draft_kv_append')\n"
     "        # --- 3) Materialize committed verify-input tokens into draft KV cache.\n"),
    ("            commit_lens=commit_lens,\n        )\n\n        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.\n",
     "            commit_lens=commit_lens,\n        )\n        torch.cuda.nvtx.range_pop()\n\n"
     "        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.\n"),
]

for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor found {n} times: {old[:80]!r}")
    src = src.replace(old, new)

# The whole decode step, from block preparation to the next draft input.
old = "        block_ids = self._draft_block_ids_buf[:bs]\n        prefix_lens = batch.seq_lens\n"
assert src.count(old) == 1
src = src.replace(old, "        torch.cuda.nvtx.range_push('m0.step')\n        torch.cuda.nvtx.range_push('m0.prepare')\n" + old)
old = "        noise_embedding = embed_module(block_ids)\n"
assert src.count(old) == 1
src = src.replace(old, "        torch.cuda.nvtx.range_pop()\n" + old)
old = ("        next_draft_input = self._make_next_draft_input_decode(\n"
       "            bonus_tokens=bonus,\n            new_seq_lens=new_seq_lens,\n        )\n")
assert src.count(old) == 1
src = src.replace(old, old + "        torch.cuda.nvtx.range_pop()\n")

open(sys.argv[2], "w").write(src)
print("patched", sys.argv[2])
