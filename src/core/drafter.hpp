// The DFlash2 drafter's weights on the GPU (maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal): five
// sliding-window attention layers with NVFP4 projections, DFlash2's grouped convolutions around each
// sublayer, the fc projection of the target's hidden states, and the candidate selector's codebooks. It
// has no embedding or head of its own: it uses the target's.
#pragma once

#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "core/model.hpp"

namespace ling {

struct DraftConfig {
  int hidden = 0;        // 5120
  int intermediate = 0;  // 17408
  int layers = 0;        // 5
  int heads = 0;         // 32
  int kv_heads = 0;      // 8
  int head_dim = 0;      // 128
  float rope_theta = 0.f;
  float eps = 1e-6f;
  int sliding_window = 0;  // 2048 keys, the query's own position included
  int trained_block = 0;   // 8
  int mask_token = 0;
  int conv_groups = 0;  // hidden / conv_group_size
  int conv_taps = 0;    // 2
  int selector_rank = 0;   // 256
  int selector_top_k = 0;  // 16
  std::vector<int> target_layers;  // the target layers whose outputs it conditions on: 5, 19, 33, 47, 61

  static DraftConfig from_file(const std::string& config_json);
};

struct DraftLayer {
  const __nv_bfloat16* input_norm = nullptr;
  const __nv_bfloat16* post_norm = nullptr;
  Fp4Weight q, k, v, o, gate, up, down;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;
  Bf16Weight attn_kproj, mlp_kproj;  // [2 * taps * groups][hidden]
  const __nv_bfloat16* attn_base = nullptr;  // [2][taps][hidden]
  const __nv_bfloat16* mlp_base = nullptr;
};

class DraftModel {
 public:
  explicit DraftModel(const std::string& dir);
  ~DraftModel();
  DraftModel(const DraftModel&) = delete;
  DraftModel& operator=(const DraftModel&) = delete;

  const DraftConfig& config() const { return cfg_; }
  const std::vector<DraftLayer>& layers() const { return layers_; }
  const Bf16Weight& fc() const { return fc_; }  // [hidden][features * hidden]
  const __nv_bfloat16* hidden_norm() const { return hidden_norm_; }
  const __nv_bfloat16* norm() const { return norm_; }
  const Bf16Weight& selector_projection() const { return sel_proj_; }  // [rank][hidden]
  const __nv_bfloat16* predecessor_codebook() const { return pred_; }   // [vocab][rank]
  const __nv_bfloat16* successor_codebook() const { return succ_; }
  size_t device_bytes() const { return bytes_; }

 private:
  DraftConfig cfg_;
  std::vector<DraftLayer> layers_;
  Bf16Weight fc_, sel_proj_;
  const __nv_bfloat16* hidden_norm_ = nullptr;
  const __nv_bfloat16* norm_ = nullptr;
  const __nv_bfloat16* pred_ = nullptr;
  const __nv_bfloat16* succ_ = nullptr;
  std::vector<void*> allocations_;
  size_t bytes_ = 0;
};

}  // namespace ling
