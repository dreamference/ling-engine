# bench/micro: the five measurements of spec section 16.1

The research spec ([specs/DREAMFERENCE_LING_ENGINE_RESEARCH.md](../../specs/DREAMFERENCE_LING_ENGINE_RESEARCH.md) §16.1) asks for five measurements. M0 wrote the device probes for the first three in [bench/membw/](../membw/); this folder holds what runs them together, the compile-only probe for the sm_100 path, and the driver for the clocks measurement. The measured numbers are in [reports/M0.md](../../reports/M0.md) §1–2 (idle machine, 2026-10-08) and [reports/micro-2026-10-10.md](../../reports/micro-2026-10-10.md) (the reference server resident).

| §16.1 item | What | Probe | Run |
| --- | --- | --- | --- |
| 1 | DRAM latency (pointer chase) and bytes in flight | `../membw/latency.cu` | `build/micro/latency` |
| 2 | Bandwidth with 4…48 SMs, plus the full read/write/copy sweep | `../membw/latency.cu`, `../membw/membw.cu` | `build/micro/latency`, `build/micro/membw [GiB=8] [seconds=12] [sustained]` |
| 3 | What sm_121 exposes: block-scaled NVFP4 and FP8 `mma.sync`, TMA, clusters and DSMEM, PDL | `../membw/isa_probe.cu` | `build/micro/isa_probe` |
| 3 | …and what it refuses: `tcgen05`/TMEM | `tcgen05_probe.cu` (compile-only; it is meant to fail for sm_121a) | see below |
| 4 | Clocks, power and temperature across a 10-minute decode | `decode_clocks.py` | see below |
| 5 | DFlash2's acceptance histogram | ling-serve's `GET /spec_stats` and `../replay/spec_report.py` | M0 §4 and M1 §4 hold the histogram; not re-run beside the reference server |

## Build

```bash
bench/micro/build.sh            # -> build/micro/{latency,membw,isa_probe}
```

The script takes the compiler from `build/CMakeCache.txt` (`CMAKE_CUDA_COMPILER`), or `/usr/local/cuda/bin/nvcc`, and compiles with `-gencode arch=compute_121a,code=sm_121a`. `-arch=sm_121a` alone emits `compute_121` PTX, which ptxas rejects for the block-scaled and `kind::f8f6f4` MMAs (M0 §2; it still does with CUDA 13.0).

## Run

Items 1–3 need the GPU otherwise idle: a resident model server that is not decoding costs nothing measurable, a decoding one shares the bus. Each probe takes 10–60 s and up to 8 GiB of device memory.

```bash
build/micro/isa_probe
build/micro/latency                    # pointer chase, then the SM sweep
build/micro/membw 8 15                 # the full sweep, then 15 s sustained
# the sm_100 path, expected to fail for sm_121a and to compile for sm_100a:
nvcc -std=c++20 -gencode arch=compute_121a,code=sm_121a -c bench/micro/tcgen05_probe.cu -o /dev/null
nvcc -std=c++20 -gencode arch=compute_100a,code=sm_100a -c bench/micro/tcgen05_probe.cu -o /dev/null
```

Item 4 drives any server with the chat-completions API with back-to-back long completions on two streams while `nvidia-smi --query-gpu` logs the SM clock, power, temperature, utilization and the active throttle-reason mask once a second (`clocks.mem` is `[N/A]` on GB10 and is skipped):

```bash
python3 bench/micro/decode_clocks.py --url http://127.0.0.1:8000 --duration 600 --streams 2 --max-tokens 4096 --csv /tmp/clocks.csv
```

It prints min / median / max per column, the throttle masks seen with their counts, and the tokens generated. Each stream finishes its last request after the deadline, so a run overshoots by up to one request. The CSV is for inspection and is not committed.

Item 5 is read from a ling-serve instance (`GET /spec_stats` before and after a replay, `bench/replay/spec_report.py BEFORE.json AFTER.json`). It needs the model loaded (M1 §How to run), so it is a replay job for a GPU without the reference server, not a probe.
