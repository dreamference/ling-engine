"""The library path for 2:4 sparsity on GB10: cuSPARSELt (through PyTorch) against dense cuBLAS at verify shapes.

Part of the low-bit study (specs/DREAMFERENCE_LING_ENGINE_LOWBIT.md section 6). Dummy weights, no model.
For each shape and row count it times y = W x^T with W dense (FP16, and FP8 through torch._scaled_mm)
and W 2:4 sparse through cuSPARSELt (FP16, and FP8 E4M3 when the library takes it on this GPU), and
prints one CSV line per arm with the bytes of W the arm reads and the effective GB/s. An arm the
library refuses on sm_121 prints its error instead of a time: that refusal is a result.

Run in a GPU window: python sparse_lib.py [reps=20]
"""

from __future__ import annotations

import sys

import torch

SHAPES = [("ffn_gate_up", 34816, 5120), ("ffn_down", 5120, 17408), ("gdn_in", 16384, 5120)]
ROWS = [16, 32]


def timed(fn, reps: int) -> float:
    """Milliseconds per call of fn, after two warm-up calls."""
    fn()
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / reps


def two_four(n: int, k: int, dtype: torch.dtype) -> torch.Tensor:
    """A random [n, k] matrix with two of every four consecutive values zero."""
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.05
    mask = torch.tensor([1, 1, 0, 0], device="cuda", dtype=torch.bool).repeat(n, k // 4)
    return (w * mask).to(dtype)


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    props = torch.cuda.get_device_properties(0)
    print(f"# {props.name} sm_{props.major}{props.minor}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    try:
        print(f"# cuSPARSELt {torch.backends.cusparselt.version()}")
    except Exception as exc:  # noqa: BLE001 - report whatever the build says
        print(f"# cuSPARSELt version unavailable: {exc}")
    print("shape,N,K,rows,arm,bytes,ms,gbps,note")
    for name, n, k in SHAPES:
        for rows in ROWS:
            x16 = torch.randn(rows, k, device="cuda", dtype=torch.float16)
            arms = []
            w16 = two_four(n, k, torch.float16)
            arms.append(("dense_fp16", n * k * 2, lambda w=w16, x=x16: torch.mm(w, x.t())))
            w8 = w16.to(torch.float8_e4m3fn)
            x8 = x16.to(torch.float8_e4m3fn)
            one = torch.ones((), device="cuda")
            arms.append(
                (
                    "dense_fp8",
                    n * k,
                    lambda w=w8, x=x8, s=one: torch._scaled_mm(
                        w, x.t(), scale_a=s, scale_b=s, out_dtype=torch.float16
                    ),
                )
            )
            for label, dtype, xin in (("sparse_fp16", torch.float16, x16), ("sparse_fp8", torch.float8_e4m3fn, x8)):
                try:
                    wc = torch._cslt_compress(two_four(n, k, dtype))
                    nbytes = wc.numel() * wc.element_size()
                    xt = xin.t().contiguous()
                    arms.append(
                        (
                            label,
                            nbytes,
                            lambda wc=wc, xt=xt, d=dtype: torch._cslt_sparse_mm(
                                wc, xt, out_dtype=torch.float16 if d != torch.float16 else None
                            ),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - a refusal is a result
                    print(f"{name},{n},{k},{rows},{label},,,,refused: {str(exc).splitlines()[0][:160]}")
            for label, nbytes, fn in arms:
                try:
                    ms = timed(fn, reps)
                    print(f"{name},{n},{k},{rows},{label},{nbytes},{ms:.4f},{nbytes / (ms * 1e6):.1f},")
                except Exception as exc:  # noqa: BLE001
                    print(f"{name},{n},{k},{rows},{label},{nbytes},,,failed: {str(exc).splitlines()[0][:160]}")
            sys.stdout.flush()
            del arms
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
