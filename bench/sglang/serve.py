"""Start a scratch SGLang container that is production's, with chosen changes (M0).

The base command is exactly what Mightling's `puffin-admin server start` runs for
`qwen3.8-27b-nvfp4-dflash2` (registry recipe, SGLang v0.5.19 image pinned by digest), so every
tuning result is one flag away from production. Changes are given as flag edits.

Usage:
  serve.py NAME [--image IMG] [--port 30000] [--set FLAG VALUE]... [--drop FLAG]...
           [--add FLAG]... [--log-requests] [--nsys] [--nvtx-patch FILE] [--env K=V]...
           [--print]

  --set   replaces a flag's value (or adds the flag with that value)
  --drop  removes a flag and its value
  --add   adds a bare flag
  --nsys  launches the server inside an nsys session named m0 (CUDA graph nodes traced); windows
          are recorded with `docker exec NAME /nsys/bin/nsys start|stop --session=m0`
"""
import argparse
import os
import shlex
import subprocess
import sys

HOME = os.path.expanduser("~")
IMAGE = "lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
MODEL = ("/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/"
         "52d1adc5f38aa5ebf099c29ed7025ba34cfbb854")
DRAFT = ("/root/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/"
         "snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1")
TEMPLATE = "/root/.cache/dreamference/chat-templates/RadixArk--Qwen3.8-27B-NVFP4-52d1adc5f38a-c84b7d991a0e.jinja"

# Production's server arguments, flag by flag (value None for a bare flag).
PROD = [
    ("--model-path", MODEL), ("--trust-remote-code", None),
    ("--served-model-name", "RadixArk/Qwen3.8-27B-NVFP4"), ("--host", "0.0.0.0"), ("--port", "8000"),
    ("--tp-size", "1"), ("--mem-fraction-static", "0.5"), ("--context-length", "262144"),
    ("--tool-call-parser", "qwen3_coder"), ("--reasoning-parser", "qwen3"),
    ("--speculative-algorithm", "DFLASH"), ("--speculative-draft-model-path", DRAFT),
    ("--speculative-num-draft-tokens", "16"), ("--speculative-draft-model-quantization", "modelopt_fp4"),
    ("--chat-template", TEMPLATE), ("--attention-backend", "flashinfer"),
    ("--sampling-backend", "pytorch"), ("--chunked-prefill-size", "8192"),
    ("--disable-prefill-cuda-graph", None), ("--cuda-graph-max-bs", "8"),
    ("--disable-flashinfer-autotune", None), ("--mamba-radix-cache-strategy", "extra_buffer"),
    ("--mamba-ssm-dtype", "bfloat16"), ("--max-mamba-cache-size", "96"),
    ("--max-running-requests", "8"), ("--enable-torch-compile", None), ("--torch-compile-max-bs", "4"),
    ("--num-continuous-decode-steps", "2"), ("--sleep-on-idle", None), ("--enable-metrics", None),
]

# Nsight Systems 2026.3.2 (CUPTI 13.4) records no kernels launched through cuLaunchKernelEx or CUDA
# graph nodes under driver 580 on GB10, only runtime calls and plain cudaLaunchKernel kernels;
# 2025.3.2, the release that ships with CUDA 13.0, records them all.
NSYS_DIR = os.environ.get("M0_NSYS_DIR", "/opt/nvidia/nsight-systems/2025.3.2")
WORKER = "/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--port", default="30000")
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("FLAG", "VALUE"))
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--log-requests", action="store_true")
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--nsys-extra", default="")
    ap.add_argument("--nvtx-patch")
    ap.add_argument("--nvtx-shim", help="nvtx.py stand-in; also turns on SGLang's scheduler spans")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--inductor-dir", default="/root/.cache/dreamference/sglang/inductor",
                    help="a separate directory keeps an experiment's compile cache apart")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()

    flags = [list(f) for f in PROD]
    flags = [f for f in flags if f[0] not in a.drop]
    for k, v in a.set:
        for f in flags:
            if f[0] == k:
                f[1] = v
                break
        else:
            flags.append([k, v])
    for k in a.add:
        flags.append([k, None])
    for f in flags:
        if f[0] == "--port":
            f[1] = a.port
    if a.log_requests:
        flags += [["--log-requests", None], ["--log-requests-level", "1"], ["--log-requests-format", "json"]]
    server = ["python3", "-m", "sglang.launch_server"]
    for k, v in flags:
        server.append(k)
        if v is not None:
            server.append(v)

    docker = ["docker", "run", "-d", "--ipc=host", "--network", "host", "--name", a.name,
              "--gpus", "all", "--cpus=14.0", "--memory=85g", "--memory-swap=85g", "--oom-score-adj=800",
              "-v", f"{HOME}/.cache/huggingface:/root/.cache/huggingface",
              "-v", f"{HOME}/.cache/dreamference:/root/.cache/dreamference",
              "-v", f"{HOME}/m0/out:/out",
              "-e", "HF_HUB_OFFLINE=1", "-e", f"TORCHINDUCTOR_CACHE_DIR={a.inductor_dir}",
              "-e", "SGLANG_PROFILE_WITH_STACK=0"]
    for e in a.env:
        docker += ["-e", e]
    if a.nvtx_patch:
        docker += ["-v", f"{os.path.abspath(a.nvtx_patch)}:{WORKER}:ro"]
    if a.nvtx_shim:
        docker += ["-v", f"{os.path.abspath(a.nvtx_shim)}:/opt/sglang/lib/python3.12/site-packages/nvtx.py:ro",
                   "-e", "SGLANG_ENABLE_NVTX_SCHEDULER=1"]
    if a.nsys:
        docker += ["--privileged", "-v", f"{NSYS_DIR}:/nsys:ro"]
        # An interactive session: `nsys start/stop --session=m0` (docker exec) records each window.
        # A cudaProfilerApi capture range inside the container produced CUPTI events that never
        # reached the report (nsys 2026.3.2, GB10), so that route is not used.
        docker += ["-e", "NSYS_TMPDIR=/out/nsys-tmp"]
        server = ["/nsys/bin/nsys", "launch", "--session-new=m0", "-t", "cuda,nvtx,osrt",
                  "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
                  ] + shlex.split(a.nsys_extra) + server
    cmd = docker + [a.image] + server
    print(shlex.join(cmd))
    if not a.print:
        os.makedirs(f"{HOME}/m0/out", exist_ok=True)
        subprocess.run(["docker", "rm", "-f", a.name], capture_output=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
