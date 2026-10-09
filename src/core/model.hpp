// The model's weights on the GPU, in the checkpoint's own encodings, and its shape from config.json.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>

namespace ling {

class Checkpoint;
struct TensorView;

// Uploads a checkpoint's NVFP4 (values and scales) or FP8 matrix and rewrites it on the GPU into the tiled
// layout; returns the tiled device buffer (the caller frees it) and adds its size to *bytes.
const uint8_t* upload_tiled(const TensorView& values, const TensorView* scales, int N, int K, size_t* bytes);

struct ModelConfig {
  int hidden = 0;            // 5120
  int intermediate = 0;      // 17408
  int layers = 0;            // 64
  int vocab = 0;             // 248320
  int heads = 0;             // 24 query heads (full attention)
  int kv_heads = 0;          // 4
  int head_dim = 0;          // 256
  int rotary_dim = 0;        // head_dim * partial_rotary_factor = 64
  float rope_theta = 0.f;    // 1e7
  float eps = 1e-6f;
  int lin_k_heads = 0;       // 16
  int lin_v_heads = 0;       // 48
  int lin_k_dim = 0;         // 128
  int lin_v_dim = 0;         // 128
  int conv_kernel = 0;       // 4
  std::vector<bool> full_attention;  // per layer
  int eos_token = 0;

  int lin_conv_channels() const { return 2 * lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim; }
  int lin_value_size() const { return lin_v_heads * lin_v_dim; }
  static ModelConfig from_file(const std::string& config_json);
};

// NVFP4: E2M1 values with one E4M3 scale per 16 and one FP32 global scale. `w` is the tiled blob
// (values and block scales together, kernels::retile_nvfp4), the order stream_gemm reads.
struct Fp4Weight {
  const uint8_t* w = nullptr;
  float scale2 = 0.f;
  float in_scale = 1.f;  // the activations' global scale (input_scale), for the prefill GEMM
  int N = 0, K = 0;
};

// FP8 E4M3 with one FP32 per-tensor scale; `w` is tiled (kernels::retile_fp8).
struct Fp8Weight {
  const uint8_t* w = nullptr;
  float scale = 0.f;
  float in_scale = 1.f;  // the activations' static per-tensor scale (input_scale), for the prefill GEMM
  int N = 0, K = 0;
};

struct Bf16Weight {
  const __nv_bfloat16* w = nullptr;
  int N = 0, K = 0;
};

struct LayerWeights {
  bool full = false;
  const __nv_bfloat16* input_norm = nullptr;
  const __nv_bfloat16* post_norm = nullptr;
  Fp4Weight gate, up, down;
  // Gated DeltaNet
  Fp8Weight in_qkv, in_z, out;
  Bf16Weight in_a, in_b;
  const __nv_bfloat16* conv = nullptr;   // [C][4]
  const __nv_bfloat16* A_log = nullptr;
  const __nv_bfloat16* dt_bias = nullptr;
  const __nv_bfloat16* lin_norm = nullptr;
  // Full attention
  Fp8Weight q, k, v, o;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;
};

class Model {
 public:
  // Loads the text model from a checkpoint directory onto the current GPU.
  explicit Model(const std::string& dir);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const ModelConfig& config() const { return cfg_; }
  const std::vector<LayerWeights>& layers() const { return layers_; }
  const __nv_bfloat16* embed() const { return embed_; }
  const __nv_bfloat16* final_norm() const { return final_norm_; }
  const Fp4Weight& lm_head() const { return lm_head_; }
  size_t device_bytes() const { return bytes_; }
  const std::string& dir() const { return dir_; }

 private:
  const void* upload(const Checkpoint& ck, const std::string& name, const std::string& dtype);
  float scalar(const Checkpoint& ck, const std::string& name);
  Fp4Weight fp4(const Checkpoint& ck, const std::string& prefix);
  Fp8Weight fp8(const Checkpoint& ck, const std::string& prefix);
  Bf16Weight bf16(const Checkpoint& ck, const std::string& name);

  std::string dir_;
  ModelConfig cfg_;
  std::vector<LayerWeights> layers_;
  const __nv_bfloat16* embed_ = nullptr;
  const __nv_bfloat16* final_norm_ = nullptr;
  Fp4Weight lm_head_;
  std::vector<void*> allocations_;
  size_t bytes_ = 0;
};

}  // namespace ling
