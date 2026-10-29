# LoRA adapters

> *The owner's decision of 2026-10-29, for Mightling 1.8.1 (it moves to 1.8.2 if it slips; it holds neither 1.8.1 nor
> the rename).* Mightling's side, the adapter library, its manifest, the signature and the gate in front of the engine,
> is `specs/DREAMFERENCE_MIGHTLING_ADAPTERS.md` in Mightling's repository; this document is the engine's half.

## 1. What is served

A PEFT LoRA adapter of the target model, rank 16 at most, one rank and one alpha for every module (no `rank_pattern`,
`alpha_pattern`, DoRA, biases or `modules_to_save`), BF16, F16 or F32 weights (held as BF16).

It may target the projections that are FP8 in the checkpoint, the projections whose input the engine has in FP32 on
every path:

| Layers | Modules | Base shape (N x K) |
|---|---|---|
| 16 full attention | `self_attn.q_proj` (query and gate, as stored) | 12288 x 5120 |
| | `self_attn.k_proj`, `self_attn.v_proj` | 1024 x 5120 |
| | `self_attn.o_proj` | 5120 x 6144 |
| 48 Gated DeltaNet | `linear_attn.in_proj_qkv`, or its split form `in_proj_q`, `in_proj_k`, `in_proj_v` (2048, 2048, 6144 rows) | 10240 x 5120 |
| | `linear_attn.in_proj_z` | 6144 x 5120 |
| | `linear_attn.out_proj` | 5120 x 6144 |

Refused by name at load, with the reason: the NVFP4 MLP, the BF16 `in_proj_a` and `in_proj_b`, the embedding, the head,
the vision tower and the MTP head. A refused adapter is never half loaded.

The forward adds `B (A x) * s` (s = alpha / r, alpha / sqrt(r) with `use_rslora`) to the base projection's FP32 output,
for the rows of the sequences on the adapter only. A row without an adapter is never touched: a run without adapters is
bit for bit main.

## 2. The two levels and the library

- **The library** (`--lora-library DIR`): the only place an adapter is loaded from. A load names a directory, which must
  resolve (realpath: neither `..` nor a symlink leaves it) under the library's root. A request never names a path.
- **The loaded set** (`--max-loaded-loras`, default 8): the adapters in host memory, by name. A full set refuses a load;
  Mightling's gate unloads the least recently used one first. An unload drops the name at once; a request already on it
  finishes on it.
- **The GPU pool** (`--max-loras-per-batch`, default 4, SGLang's meaning: the base counts as one, so 3 adapter slots):
  the adapters a batch may mix. A sequence on an adapter holds its slot; a slot is refilled least recently used first,
  and a request waits while every slot holds an adapter some running sequence uses.
- **The id** of a loaded adapter is FNV-1a over its name and its weight files. A reloaded adapter with other weights has
  another id, so nothing computed under the old one is reused.

## 3. Kernels

Each layer has up to four sites, the modules that read one input taken together: Qkv (q, k, v), O, Gdn (in_qkv and
in_z, or the split form's four), Out. A site is one **shrink**, `t = x A^T` over its stacked A blocks (split-K partials,
no atomics, so the sum's order is fixed), and one **expand**, `y += s * B t`, each B segment into its rows of its output.
The base product runs once for the whole batch; the adapter work runs per distinct adapter over its rows only. A row's
slot is a device array, written before the forward like the positions, so captured graphs read it; graphs with
adapter rows are captured apart from the base ones.

Budget: 0.5 to 1 ms per distinct adapter per step.

## 4. Speculation, sessions and the prefix store

- The DFlash2 drafter may carry its own adapter (`drafter/` in the package, checked against the drafter's shapes); a
  request on the adapter drafts with it, so acceptance follows the target's adapter. Verification is the target's, so
  greedy output with speculation on equals speculation off on every adapter.
- Sessions, checkpoints and the prefix store are keyed by a share scope (P4c). A request on an adapter uses
  `lora_scope(scope, adapter)`: the base keeps its scope, an adapter mixes its id in. A switched adapter re-prefills;
  nothing is shared across adapters, or between an adapter and the base.

## 5. API

- `GET /v1/models` lists the base and `base+<adapter>` for every loaded adapter.
- `model: "base+<adapter>"` selects the adapter on `/v1/responses`, `/v1/chat/completions` and `/v1/completions`; an
  adapter not loaded is a 404 in the API's error shape, a malformed name a 400.
- `POST /load_lora_adapter {"lora_name", "lora_path"}` and `POST /unload_lora_adapter {"lora_name"}`, SGLang's routes
  and bodies, behind `--api-key` (a bearer token). Unlike SGLang's, the key guards only these routes: the inference
  routes stay as they are, behind Mightling's model gate.
- `--lora-paths name=path,...` loads adapters at start, from the library like any load.

## 6. Gates before the merge

1. **No adapter: bit for bit main.** agentlong 16 of 16 turns identical; `--spec-check`, `--prefix-check`,
   `--graph-check`, `--batch-check` and the stream tests pass.
2. **A zero adapter** (every B zero) gives output identical to the base.
3. **Agreement with an adapter**, against SGLang with the same adapter (the reference) and transformers with PEFT
   (the second opinion): the agreement gate's lines with the vision tower's tolerance
   ([VISION §20.5](./DREAMFERENCE_LING_ENGINE_VISION.md), items 3 and 5).
4. **Speed**: one adapter within 5 % of the base; the cost per distinct adapter measured. Parity bar: 54 to 56 tok/s
   with an adapter, against SGLang's 57.7 on the base.

## 7. Tests

`tests/lora_tests.cpp` (host): both forms of the DeltaNet's input projection, every refusal, the library's path checks,
the loaded set, the id, the scopes and the pool's slots.
