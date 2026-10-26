// The checkpoint's vision tower on the GPU (specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.1, §20.3 (a)): patches of
// one image in, one 5,120-wide embedding per 2 x 2 merge window out, in BF16.
//
// The forward mirrors production's rounding points (§20.2 item 6): every operation accumulates in FP32 and rounds its
// output to BF16 where torch's BF16 module does (the patch embedding, the interpolated position embedding and its
// add, each LayerNorm, each linear layer, the activation, each residual add, the rotary in FP32 rounded to BF16, the
// attention's output). Linear layers run on gemm_bf16 (FP32 out) in row chunks followed by one epilogue kernel;
// attention is a FlashAttention-2-style kernel of this file (head dimension 72, bidirectional within the image).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ling {

class Checkpoint;

struct VisionConfig {
  int depth = 27;
  int hidden = 1152;
  int heads = 16;
  int head_dim = 72;
  int inter = 4304;      // the blocks' MLP width
  int inter_pad = 4352;  // padded to the BF16 GEMM's 64-wide K tiles (fc2's weight padded with zero columns at load)
  int out = 5120;        // the language model's hidden size
  int patch = 16, temporal = 2, merge = 2, channels = 3;
  int pos_side = 48;     // the learned position table is pos_side x pos_side
  int patch_dim() const { return channels * temporal * patch * patch; }  // 1536
  int merged() const { return hidden * merge * merge; }                  // 4608
};

class VisionTower {
 public:
  // Loads model.visual.* from the checkpoint directory onto the current GPU and allocates scratch for images of up to
  // `max_patches` patches (the processor's maximum is 65,536).
  VisionTower(const std::string& model_dir, int max_patches);
  ~VisionTower();
  VisionTower(const VisionTower&) = delete;
  VisionTower& operator=(const VisionTower&) = delete;

  const VisionConfig& config() const { return cfg_; }
  int max_patches() const { return max_patches_; }
  size_t weight_bytes() const { return weight_bytes_; }
  size_t scratch_bytes() const { return scratch_bytes_; }

  // One image: `patches` (host, BF16 bit patterns) [gh * gw][1536] in merge-window order (the processor's
  // pixel_values cast to BF16); `out` (device) receives [gh * gw / 4][5120] BF16. gh and gw are even.
  // tap >= -1 (a check): also copies the residual stream after the patch and position embeddings (-1) or after block
  // `tap` into *tap_out ([gh * gw][1152] BF16 bits), before the merger.
  void forward(const uint16_t* patches, int gh, int gw, __nv_bfloat16* out, cudaStream_t s, int tap = -2,
               std::vector<uint16_t>* tap_out = nullptr);

 private:
  struct Block {
    const __nv_bfloat16 *n1w, *n1b, *n2w, *n2b;
    const __nv_bfloat16 *qkv_w, *qkv_b, *proj_w, *proj_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
  };
  // y = x . W^T + b over M rows, in chunks of kChunk rows through the FP32 scratch; then the epilogue (act: 0 none,
  // 1 tanh GELU, 2 erf GELU) into `out` (BF16, row stride ld_out, columns N .. ld_out zeroed), or added into it
  // (residual: out = bf16(out + bf16(y + b))).
  void linear(const __nv_bfloat16* x, int M, int K, const __nv_bfloat16* w, const __nv_bfloat16* b, int N,
              __nv_bfloat16* out, int ld_out, int act, bool residual, cudaStream_t s);
  const __nv_bfloat16* upload(const Checkpoint& ck, const std::string& name, std::vector<int64_t> shape);

  VisionConfig cfg_;
  int max_patches_ = 0;
  size_t weight_bytes_ = 0, scratch_bytes_ = 0;
  std::vector<void*> owned_;
  const __nv_bfloat16 *patch_w_ = nullptr, *patch_b_ = nullptr, *pos_ = nullptr;
  std::vector<Block> blocks_;
  const __nv_bfloat16 *mn_w_ = nullptr, *mn_b_ = nullptr, *m1_w_ = nullptr, *m1_b_ = nullptr, *m2_w_ = nullptr,
                      *m2_b_ = nullptr;
  // Scratch.
  __nv_bfloat16 *x_ = nullptr, *xn_ = nullptr, *qkv_ = nullptr, *attn_ = nullptr, *act_ = nullptr;
  float* y_ = nullptr;  // kChunk x max N FP32
  int* pos_idx_ = nullptr;       // [4][max_patches] the bilinear corners
  __nv_bfloat16* pos_wt_ = nullptr;  // [4][max_patches] their BF16 weights
  int* rot_pos_ = nullptr;       // [max_patches][2] (row, col)
};

namespace vision_kernels {
// The kernels, exposed for ling-vision-tests.
// out[r] = bf16(LayerNorm(x[r]) * w + b), rows of `cols` (FP32 statistics).
void layernorm(const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, __nv_bfloat16* out, int rows,
               int cols, float eps, cudaStream_t s);
// The rotary on q and k of qkv [n][3][heads][72] in place: dimension d < 36 and d + 36 turn by the angle
// pos.row * f[d] (d < 18) or pos.col * f[d - 18], f[j] = 10000^(-2j/36); FP32 math, BF16 out.
void rotary(__nv_bfloat16* qkv, const int* pos, int n, int heads, cudaStream_t s);
// Bidirectional attention over n tokens: qkv [n][3][heads][72] -> out [n][heads][72], softmax scale 72^-0.5.
void attention(const __nv_bfloat16* qkv, __nv_bfloat16* out, int n, int heads, cudaStream_t s);
// The epilogue of linear().
void bias_act(const float* y, const __nv_bfloat16* b, int M, int N, __nv_bfloat16* out, int ld_out, int act,
              bool residual, cudaStream_t s);
// x[p] = bf16(x[p] + bf16(sum_i bf16(table[idx[i][p]] * wt[i][p]))), the interpolated position embedding.
void add_pos(__nv_bfloat16* x, const __nv_bfloat16* table, const int* idx, const __nv_bfloat16* wt, int n, int hidden,
             cudaStream_t s);
}  // namespace vision_kernels

// The host side of the position embedding and the rotary positions for a gh x gw patch grid, in merge-window order
// (§20.1): the four bilinear corners into the pos_side x pos_side table and their weights (FP32, as production
// computes them before the BF16 cast), and each patch's (row, col).
void vision_positions(int gh, int gw, int merge, int pos_side, std::vector<int>& idx4, std::vector<float>& wt4,
                      std::vector<int>& rowcol);
// torch.linspace(start, end, steps) in FP32, as its kernel computes it (the first half from start, the rest from end).
std::vector<float> torch_linspace(float start, float end, int steps);

}  // namespace ling
