# ling-engine: Images, the vision encoder

Part of the ling-engine specification; the index is [README.md](./README.md). This document is new (2026-10-26) and has no counterpart in the single-file `SPEC.md`; it takes the next free section number, §20, so citations of the form `§20.n` work like the others.

## 20. Images

**The owner's decision (2026-10-26).** Image input is ling-engine's next priority, ahead of adapters and of batches beyond two: it is the most visible gap before the engine can become Mightling's default server ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), "Order", item 3, moved to the front; [BACKLOG §19.16](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) is its scope). Today `ling-serve` renders an image part as the text `[image]` ([API §7](./DREAMFERENCE_LING_ENGINE_API.md)), so a screenshot sent to Ask or pasted into `ling` reaches the model as five characters.

What this section fixes, in order: what the checkpoint's vision tower is (§20.1), what production does with an image, read from SGLang v0.5.21's code for its behaviour only (§20.2), what the engine builds and in which phases (§20.3), the memory and time it costs on GB10 (§20.4), what "exact" means for an image prompt (§20.5), the API as Mightling's clients use it (§20.6), images and the prefix caches (§20.7), and the tests (§20.8).

Sources: the served checkpoint's `config.json`, `preprocessor_config.json`, `chat_template.jinja` and safetensors index (`RadixArk/Qwen3.8-27B-NVFP4`, snapshot `52d1adc5`); production's image, `lmsysorg/sglang@sha256:b1259f3e…` (v0.5.21, Mightling's registry pin since 2026-10-14), its SGLang sources and the transformers 5.12.1, torch 2.13.0, torchvision 0.28.0 and Pillow 12.3.0 inside it; the clients' code in the Mightling repository (Codex at `codex/`, the launcher `ling-rs/`, `ling web`, the desktop app, the phone bridges).

### 20.1 The checkpoint's vision architecture

The checkpoint is `Qwen3_5ForConditionalGeneration` (`model_type` `qwen3_5`): the language model of [§5](./DREAMFERENCE_LING_ENGINE_MODEL.md) plus a Qwen3-VL-style vision tower, `model.visual.*`, 333 tensors, all in BF16 in shard 1 (ModelOpt left every vision quantizer disabled), about 0.46 B parameters, 0.92 GB.

| Part | Shape | Notes |
| --- | --- | --- |
| Patch embedding | Conv3d 3 → 1152, kernel = stride = (2, 16, 16), bias | Equivalent to a linear map of the flattened 1,536-value patch (channel, frame, row, column order). An image is one frame repeated twice (`temporal_patch_size` 2), so both frame slices of the kernel see the same pixels. |
| Position embedding | learned table 2,304 × 1152 (a 48 × 48 grid) | Bilinearly interpolated to the image's patch grid with `linspace(0, 47, n)` sample points (corners aligned) and added to the patch embeddings. |
| 27 blocks | pre-LayerNorm (eps 1e-6, with bias) → attention → residual; LayerNorm → MLP → residual | Attention: one `qkv` linear 1152 → 3456 with bias, 16 heads of dimension 72, a 2-D rotary embedding, softmax scale 72^-0.5, **bidirectional within one image** (`cu_seqlens` per image), `proj` 1152 → 1152 with bias. MLP: 1152 → 4304 → 1152 with biases, `gelu_pytorch_tanh`. |
| Vision rotary | per head, rotate-half over all 72 dims | 18 frequencies `10000^(-2j/36)`; each patch has a (row, column) position on the patch grid; angles `[row·f, col·f, row·f, col·f]` (36 + 36), so dimension d < 18 turns with the row, 18 ≤ d < 36 with the column, and d + 36 pairs with d. Computed in FP32. |
| Merger | LayerNorm(1152) per patch → 4 consecutive patches concatenated (4,608) → linear 4608 → 4608 + exact (erf) GELU → linear 4608 → 5120 | Patches arrive in 2 × 2 merge-window order (below), so a window is four consecutive rows. Output: one 5,120-wide embedding per 2 × 2 window, the language model's hidden size. |
| Deepstack | none | `deepstack_visual_indexes` is empty: only the merger's output enters the language model, and only at its input. |

**Patch order.** The processor emits the patch grid (gh × gw, both even) in merge-window order: for each window row, each window column, the window's four patches row by row. Token k of an image is window (k / gw', k mod gw') with gw' = gw/2, so the image's `gh/2 × gw/2` tokens are in row-major order over the merged grid.

**Tokens in the prompt.** The chat template writes `<|vision_start|><|image_pad|><|vision_end|>` (ids 248053, 248056, 248054) where an image part is, in user, assistant and tool messages alike (a system message with an image is an error), with no "Picture n:" label (`add_vision_id` unset). The processor then repeats `<|image_pad|>` once per merged token: N = (gh/2)·(gw/2). The language model's embedding of each of those positions is replaced by the merger's output, in order; nothing else changes in the language model (no scaling, no extra layers).

**M-RoPE.** The language model's attention layers use interleaved multimodal RoPE ([§5.4](./DREAMFERENCE_LING_ENGINE_MODEL.md)): 64 rotary dimensions, 32 frequencies, `mrope_section` [11, 11, 10] interleaved, so frequency i takes the height position when i mod 3 = 1 and i < 33, the width position when i mod 3 = 2 and i < 30, and the temporal position otherwise. A text token has one position p for all three. An image starting at prompt index s, whose text before it ends at position P − 1, takes temporal P, height P + r and width P + c for its token in merged row r and column c; the text after it continues at P + max(gh/2, gw/2), not at s + N. So after an image a token's position is its index plus a negative delta, the sum over the images before it of max(gh/2, gw/2) − N. Text-only prompts are unchanged: their positions are their indices and the three components coincide, which is exactly today's RoPE. The vision tower's own 2-D rotary is separate and has nothing to do with the prompt position.

**Preprocessing** (`Qwen2VLImageProcessorFast`, `preprocessor_config.json`): `patch_size` 16, `merge_size` 2, so the resize factor is 32; `shortest_edge` 65,536 and `longest_edge` 16,777,216 are min_pixels and max_pixels (64 to 16,384 tokens per image); mean and std 0.5 per channel; bicubic resampling.

- **Size.** `smart_resize`: h' = round(h/32)·32, w' = round(w/32)·32 with Python's `round` (halves to even: 48 px gives 64, 80 px gives 64); if h'·w' > max_pixels, β = √(h·w/max_pixels), h' = max(32, ⌊h/β/32⌋·32) and the same for w; if h'·w' < min_pixels, β = √(min_pixels/(h·w)), h' = ⌈h·β/32⌉·32 and the same for w; an aspect ratio above 200 is an error. A 1920 × 1080 screenshot becomes 1920 × 1088: a 68 × 120 patch grid, 2,040 tokens.
- **Resampling.** torchvision's `resize` with `antialias=True` on a uint8 tensor that is not on the CPU: cast to FP32, PyTorch's separable antialiased bicubic (a = −0.5, the PIL kernel, support widened by the scale when downsampling, weights normalised over the clipped window), clamp to [0, 255], round half to even, back to uint8. No resampling when the size is already a multiple of 32 and within bounds.
- **Normalisation.** In FP32, (x − 127.5)/127.5 (mean and std folded with the 1/255 rescale), then the patch tensor `[gh·gw][1536]` in the order above, cast to BF16 at the patch embedding.

### 20.2 What production does (SGLang v0.5.21, read for behaviour)

The production server answers an image request as follows. Every line is a behaviour the engine copies or deliberately departs from (§20.6).

1. **Parts.** Chat completions: every message's content list is scanned; an `image_url` or `input_image` part becomes `{"type": "image"}` for the template and its URL joins the request's image list in order, in any role, tool messages included (`jinja_template_utils.process_content_for_template_format`). `detail` is recorded and ignored by the Qwen processor. Responses: `input_image` in a message becomes `image_url`; **`function_call_output.output` given as a list is flattened to its text, so an image in a tool result never reaches the model** (`serving_responses._normalize_response_message_for_chat`).
2. **Loading.** `data:` URLs, http(s) URLs, file paths and bare base64 are all accepted. A JPEG (by its SOI/EOI markers) is decoded on the GPU by nvJPEG through `torchvision.io.decode_jpeg(device="cuda")` (`gpu_image_decode` defaults to on), falling back to Pillow if nvJPEG fails; everything else goes through Pillow and `convert("RGB")`, which drops alpha without compositing (no EXIF rotation on either path).
3. **Processing** on `cuda:0` with the fast processor (§20.1), so the resize runs on the GPU, in FP32.
4. **Placement.** The `<|image_pad|>` of each placeholder is expanded to N tokens. For the radix cache the expanded ids are replaced by a per-image value, 1,000,000 + (hash mod 2^30), the hash being of the processed pixels, so two different images never share a cached prefix and the same image does (`schedule_batch._compute_pad_value`).
5. **M-RoPE positions** from the expanded ids and the grids (`_compute_image_only_mrope_positions_from_offsets`, the same as transformers' `get_rope_index` for `qwen3_5`), and `mrope_position_delta` for the decode.
6. **The tower** in BF16: patch embedding (BF16 conv), the position embedding interpolated with **BF16 weights and BF16 products summed** (transformers keeps the weights in FP32; production does not), BF16 residual stream, LayerNorms with BF16 output, linear layers with BF16 output, rotary in FP32 rounded to BF16, attention by the Triton vision backend on this GPU (BF16 in, FP32 softmax), merger output BF16, scattered over the image tokens' embeddings.
7. **Speculation.** DFLASH runs over the image prompt like any other; the drafter conditions on the target's hidden states at the image positions as well.

### 20.3 What the engine builds

Four phases, each exact (§20.5) before the next starts; the branch is `vision/image-input`, merged into main with `--no-ff` only after the full test set (§20.8).

**(a) The tower on the GPU** (`src/core/vision.cu`, `vision.hpp`). Loaded at start unless `--vision off`: the 333 tensors in BF16, fc2's 4,304-wide K padded to 4,352 with zero columns once at load so the BF16 GEMM's 64-wide K tiles fit. The forward mirrors production's rounding points (§20.2 item 6), with FP32 accumulation inside every operation: patch embedding and every linear layer on `gemm_bf16` (the engine's own TMA GEMM, FP32 out) in row chunks into an FP32 scratch, followed by one epilogue kernel (bias, activation, BF16 rounding); LayerNorm with BF16 output; rotary on q and k in FP32; attention by a FlashAttention-2-style kernel of its own (BF16 m16n8k16 MMAs, head dimension 72 padded to 80 for QK^T, 64 queries per block, online softmax in FP32, never the N × N score matrix); BF16 residual adds; the merger. One image per call; its output, `[N][5120]` BF16, goes to an embedding cache keyed by the image's hash (LRU, `--vision-cache-mb`, 512 by default), so a resumed or repeated prefill does not encode it again. `ling-run --vision-check IMAGE DUMP` compares the tower's output with a reference dump.

**(b) Image tokens in the prefill** (`engine.cpp`, `kernels.cu`, `attention_body.cuh`). The prompt carries one id per image token, ≥ 1,000,000 (§20.7); the engine never looks such an id up in a vocabulary table:
- after `kernels::embed` (which reads id 0 in its place), the image's rows of the BF16 embedding cache are converted to FP32 into the residual stream `h_` at their prompt positions, so a chunk may start or end inside an image;
- `attn_prepare` takes the rope position apart from the cache slot (as the tree verify already does) and gets two inputs: a per-row (t, h, w) table for prefill chunks of prompts with images (frequency i picks its component as in §20.1), and a per-sequence delta for every pass after the prompt (decode, verify, the tree's rows, the batched verify). The delta lives beside `pos_dev_` on the device, so the captured step graphs read the current session's value;
- the drafter keeps the slot positions it was trained on (it only proposes, so this changes acceptance at most, never the output; a later item measures whether the delta helps it);
- the lookup source and the sparse head's context vocabulary skip ids at or above the vocabulary size.
Text-only prompts pass a null table and a zero delta, which compute bit for bit what main computes.

**(c) The API and preprocessing** (`src/serve/vision_input.cpp`, `chat_template.cpp`, `api.cpp`, `main.cpp`). The renderer writes production's placeholder and records each image's URL in render order; the HTTP handler (off the engine thread) decodes and preprocesses each image (§20.1), hashes the decoded pixels, takes the image's id from the registry, tokenizes, checks that the number of `<|image_pad|>` tokens equals the number of images, and expands each into N copies of its id. Decoders: nvJPEG for JPEG as production does (libjpeg-turbo when nvJPEG refuses a file, as production falls back to Pillow), libpng for PNG, libwebp for WebP; the format is sniffed from the bytes, never taken from the MIME label (the phone bridges label a Telegram photo `image/jpeg` whatever it is). `GET /model_info` and `/get_model_info` answer `has_image_understanding`, which Mightling's launcher reads (`ling-rs/src/vision.rs`) to advertise `input_modalities: ["text", "image"]` to Codex and to tell the phone bridges they may attach photos.

**(d) End to end**: greedy identity against the reference on a fixed set of real images (§20.8), then the image prefill's speed (§20.4).

### 20.4 Memory and time on GB10

**Memory** (engine-owned, allocated at start, never grown while serving):

| Item | Size |
| --- | --- |
| Tower weights (BF16, fc2 padded) | 0.92 GB |
| Tower scratch at the processor's maximum image (65,536 patches): residual, norm output, qkv, attention output in BF16; one 8,192-row chunk of FP32 GEMM output and of BF16 MLP activation | 1.15 GB |
| Embedding cache (`--vision-cache-mb`) | 0.5 GB |
| **Total** | **≈ 2.6 GB** |

INTEGRATION §12's "embedding and vision ~3.5 GB" line counted the BF16 token embedding (2.54 GB) and the tower's weights; the scratch and cache come on top. The prefix KV store's arena is sized from what the engine leaves above its reserve (BACKLOG 35), so it shrinks by that much rather than the footprint growing. `--vision-max-pixels` lowers the scratch (Codex never sends more than 10,000 tokens of image, §20.6: 40,000 patches, 0.8 GB of scratch); `--vision off` frees all of it and answers `has_image_understanding: false`.

**Time** (estimates; phase (d) measures them against production on the same GB10). The tower costs 0.85 GFLOP per patch in its linear layers (0.43 G multiply-adds: 27 blocks, the merger, the patch embedding) and 4·N·72·16 FLOP per patch per block in attention, N the image's patches:

| Image | Patches → tokens | Linear layers | Attention | Tower at ~70 / ~50 TFLOPS | Language model prefill of the tokens (at ~2,300 tokens/s) |
| --- | --- | --- | --- | --- | --- |
| 1024 × 768 photo | 3,072 → 768 | 2.6 TFLOP | 1.2 TFLOP | ≈ 0.06 s | ≈ 0.33 s |
| 1920 × 1080 screenshot | 8,160 → 2,040 | 6.9 TFLOP | 8.3 TFLOP | ≈ 0.26 s | ≈ 0.9 s |
| Codex's `high` cap, 2,500 tokens | 10,000 → 2,500 | 8.5 TFLOP | 12.4 TFLOP | ≈ 0.37 s | ≈ 1.1 s |
| Codex's `original` cap, 10,000 tokens | 40,000 → 10,000 | 34 TFLOP | 199 TFLOP | ≈ 4.5 s | ≈ 4.3 s |

The BF16 rates are what others measured on GB10 (about 101 TFLOPS for a good BF16 GEMM, 41–69 TFLOPS for FlashAttention at head dimension 64; [RESEARCH §16](./DREAMFERENCE_LING_ENGINE_RESEARCH.md)); the language model's prefill dominates every image Codex sends by default. The tower runs inside the prefill's first chunk that needs it, on the prefill's stream (and on the side stream's green context when the prefill runs beside a decode, M4), so the other session's step period is bounded by the same contended-chunk rule as a text chunk. Decoding and preprocessing run on the HTTP thread, not the engine's: a 1080p PNG decodes in about 20 ms and resizes in about 30 ms on one core (estimate).

### 20.5 Exactness

The engine's standing rule ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), exactness policy and agreement gate) carries over, with production as the reference and transformers as the second opinion.

1. **Text stays bit for bit.** With no image in the prompt, every pass computes what main computes: `--spec-check`, `--prefix-check`, `--graph-check`, `--batch-check`, the stream tests and agentlong greedy identical to main token for token.
2. **Preprocessing is exact where it can be.** Grids and token counts equal production's on every test image (production's `usage.prompt_tokens` gives text tokens + N, which checks `smart_resize` and the expansion with nothing else); PNG and WebP pixels equal Pillow's bit for bit; the resized, normalised patch tensor equals the reference processor's to within one uint8 level at the rare pixels where FP32 summation order rounds the other way (counted and reported); JPEG pixels equal nvJPEG's (the same library production calls) and, on the fallback, libjpeg-turbo's with Pillow's settings.
3. **The tower matches the reference embeddings to within the reference's own noise.** The floor is measured, not assumed: transformers' tower in BF16 against the same tower in FP32 on the same patch tensor, and production's (SGLang's module) against transformers' BF16. The engine's output must sit no further from the FP32 tower than production's does, by relative L2 error and by cosine per token, on every test image.
4. **Positions are exact.** The (t, h, w) table and the delta equal transformers' `get_rope_index` on prompts with one image, several, images at the start and the end, and an image in a tool result.
5. **End to end within the text tolerance.** On the image prompt set (§20.8), teacher-forced on production's greedy continuation, the agreement gate's lines as for text: the flip rate not significantly above production's own self-noise on the same set (its greedy decode against its own prefill, one-sided Fisher test at α = 0.05, threshold recomputed for this set), every flip's gap ≤ 1.75 nats, top-5 agreement 100%, no concentration of flips by position or near the image. Free-running greedy outputs are reported beside it (identical, or the first divergence and its margin).
6. **Image prompts are themselves deterministic and row-invariant.** Within the engine, greedy output with speculation on equals speculation off on image prompts, a resumed prefill (cut inside an image and after it) equals a cold one bit for bit in the kept state, and two streams equal one.

### 20.6 The API, as Mightling's clients send images

What the clients send (from their code):

| Client | Route | Part | URL | Formats | Size |
| --- | --- | --- | --- | --- | --- |
| `ling` (Codex): an attached or pasted image | `/v1/responses` | `{"type":"input_image","image_url":"data:…","detail":"high"}` in a user message, between the text parts `<image name=[Image #1]>` and `</image>` | `data:` only (an http(s) image becomes a text placeholder in Codex) | PNG, JPEG, WebP (GIF and others re-encoded to PNG) | at most 2,048 px a side and 2,500 tiles of 32 px (`high`, the default); 6,000 px and 10,000 tiles (`original`) |
| `ling`: `view_image`, MCP tool images, Code Mode `exec` output | `/v1/responses` | `function_call_output` whose `output` is a list holding `input_image` (and `input_text`) | `data:` | as above | as above |
| `ling web`, the desktop app (Ask and Work) | through `ling app-server`, then as `ling` | `localImage` → as `ling` | | PNG, JPEG, GIF, WebP picked | Codex resizes |
| Telegram, Matrix, Signal bridges | as `ling`, and only when the catalog lists `image` | | | mostly JPEG | Codex resizes |
| Image-search ranker (`dreamference/chat/image_search_service.py`) | `/v1/chat/completions` | `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,…"}}` | `data:` | JPEG | 256 px |

The engine's API:

- **Chat completions**: an `image_url` part (`image_url` an object with `url`, or a string) or an `input_image` part, in user, assistant or tool messages, as production; in a system message, 400 with the template's message.
- **Responses**: `input_image` in a message (`image_url` a string, or `file_id`, which is refused: there is no file store); **in a `function_call_output` or `custom_tool_call_output` list, rendered as image parts** of the tool message. This departs from production on purpose (§20.2 item 1): production drops those images, so Codex's `view_image` and Code Mode screenshots never reach the model; the chat template renders an image in a tool message, and the same prompt sent to production as a chat completion with the image in the tool message's content is the reference for it. A list without images still renders as its concatenated text, as today.
- `detail` is accepted and ignored, as production ignores it for this model.
- **URLs**: `data:<mime>;base64,<payload>`, or bare base64. http(s), `file://` and paths are refused with a 400 (`invalid_request_error`, code `image_url_not_supported`): the server is air-gapped by design and no client sends them.
- **Formats**: PNG, JPEG, WebP, sniffed from the bytes. GIF, BMP, TIFF, HEIC and anything else: 400, code `unsupported_image_format`, naming the format (production would decode GIF, BMP and TIFF through Pillow; Codex re-encodes them to PNG before they leave the client). Animated WebP and APNG: first frame, as Pillow.
- **Size limits**: a `data:` payload above 20 MB decoded, or more than 64 images in one request, is a 400. The token count, image tokens included, goes through the existing context-length check, so an image that does not fit is `context_length_exceeded` like text.
- `usage.prompt_tokens` counts the image tokens, as production; `cached_tokens` counts reused image tokens too.
- `/model_info` and `/get_model_info`: `{"model_path": …, "has_image_understanding": true}` (false with `--vision off`).
- Video parts stay a 400.

### 20.7 Images and the prefix caches

Everything the engine reuses is keyed on token prefixes: a session is bound when the prompt extends its history ([§11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md), M4 stage 1), the checkpoint store keeps states at prompt positions, the prefix KV store is a radix tree over token ids (BACKLOG 35). With every image rendered as the same `<|image_pad|>` id, a prompt with a new screenshot would match a cached prompt with an old one and reuse its KV and DeltaNet state: wrong output, silently.

So, as production does, an image's tokens carry the image's identity: the HTTP layer gives each distinct image an id from a registry, 1,000,000 + k, keyed by a 64-bit hash of the decoded RGB pixels and their size (the same picture sent as PNG and as WebP is one image). The registry probes past a taken id on a hash collision, so two images never share an id in a process, which production's `hash mod 2^30` does not guarantee; the caches live only in the process's memory, so the ids need no persistence. The prompt the engine sees holds N copies of that id where the image goes; every structure that matches prefixes then works unchanged. The places that read a token id as a vocabulary index treat these ids explicitly: the embedding (replaced, §20.3 (b)), the lookup source (proposes nothing from them) and the sparse head's context vocabulary (skips them); the sampler, penalties and stop strings only ever see generated tokens. A resumed prefill may start inside an image: the image's remaining rows come from the embedding cache, or the tower runs again. Codex re-sends every earlier image with each turn; with the ids stable, a turn after a screenshot reuses the screenshot's KV and runs the tower only for new images.

### 20.8 Test plan

**CPU (no model, in `ling-api-tests` and a new `ling-vision-tests`, run on every build):**
- `smart_resize` against transformers' on 40 sizes, rounding halves, both bounds, aspect 200 refused;
- the resampler against golden outputs produced by the reference processor on the GPU (in production's image) for 12 images (up, down, odd ratios, already aligned): counts of differing uint8 values, expected 0, allowed ≤ 1 level at ≤ 0.1% of values;
- the patch tensor order against transformers' `pixel_values`;
- decoders: PNG (8/16-bit, palette, grey, alpha), WebP (lossy, lossless, alpha), JPEG (4:2:0, 4:4:4, grey, progressive) against Pillow's pixels and, for JPEG, nvJPEG's;
- the template: image parts in user, tool and assistant messages, system refused, text-only unchanged; Responses with `input_image`, with images in `function_call_output`, `file_id` refused; token counts and positions of the expanded ids;
- M-RoPE tables and deltas against `get_rope_index` dumps on eight prompts;
- the registry: the same pixels in two formats give one id, a forced collision probes;
- `/model_info`.

**GPU (a ≤ 45-minute window each, under the lock, `m0-prod8000` stopped and restarted per policy):**
- window 1 (references, no engine): production's image (v0.5.21) on the test images: the processor's `pixel_values` and grids, transformers' tower output in FP32 and BF16, SGLang's tower output, production's greedy continuations, top-k logprobs and `usage` on the image prompt set, production's decode-vs-prefill self-noise on that set;
- window 2, phase (a): `ling-run --vision-check` on every test image against window 1's dumps; the tower's time per image;
- window 3, phase (b) and (c): the position tables on the GPU, `--spec-check`, `--prefix-check` (cut inside and after an image), `--graph-check`, `--batch-check` on image prompts; the text-only checks against main;
- window 4, phase (d): ling-serve on the image prompt set, the agreement gate (§20.5 item 5), `tests/server` with the new image tests, the image prefill's time against production's (time to first token, the tower's share);
- before the merge: the full text set: `ctest` (kernels, tree, megakernel, stream, prefill, spec, sampling, api), `--spec-check`, `--batch-check`, `--prefix-check`, `--graph-check`, agentlong greedy identical to main.

**The image prompt set** (`bench/vision/`): 24 prompts over 12 real images, kept in the repository only where their licence allows and otherwise fetched by a script and checked by SHA-256: four screenshots (a terminal, a web page, an IDE with code, a dense dashboard), four photos (an indoor scene, a street, a document photographed at an angle, a whiteboard), two code screenshots, a chart, a small icon (upscaled to the minimum); questions that need the image (read the error, transcribe the function, count the objects), three with two images, two with the image in a tool result, two multi-turn where a later turn re-sends the image (the prefix path).
