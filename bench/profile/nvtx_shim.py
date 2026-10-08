"""Stand-in for the `nvtx` package, which the SGLang image lacks (M0).

SGLang's scheduler spans (SGLANG_ENABLE_NVTX_SCHEDULER=1) call only `nvtx.annotate(name,
color=...)` as a context manager; this routes it to torch.cuda.nvtx. Mounted as nvtx.py into the
container's site-packages by serve.py --nvtx-shim.
"""
from contextlib import contextmanager

import torch


@contextmanager
def annotate(message=None, color=None, domain=None, category=None):
    torch.cuda.nvtx.range_push(str(message))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
