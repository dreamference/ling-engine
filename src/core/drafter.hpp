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
  // A BF16 drafter (the original's projections unquantized; bench/drafter/big_dummy.py measures its cost): these
  // replace q .. down on the BF16 rows path.
  bool bf16 = false;
  Bf16Weight bq, bk, bv, bo, bgate, bup, bdown;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;
  Bf16Weight attn_kproj, mlp_kproj;  // [2 * taps * groups][hidden]; w null when the NVFP4 copies replace them
  // BACKLOG 33: the conv kernel projections in NVFP4, rounded at load (w null with the copies off). The block's rows
  // path reads them instead of the BF16 originals, which are then not kept: every drafter pass has at most
  // kMaxStreamRows rows.
  Fp4Weight attn_kproj4, mlp_kproj4;
  const __nv_bfloat16* attn_base = nullptr;  // [2][taps][hidden]
  const __nv_bfloat16* mlp_base = nullptr;
};

class DraftModel {
 public:
  // nvfp4_copies (BACKLOG 33): round the fc and the conv kernel projections to NVFP4 at load for the rows path.
  explicit DraftModel(const std::string& dir, bool nvfp4_copies = true);
  ~DraftModel();
  DraftModel(const DraftModel&) = delete;
  DraftModel& operator=(const DraftModel&) = delete;

  const DraftConfig& config() const { return cfg_; }
  const std::vector<DraftLayer>& layers() const { return layers_; }
  const Bf16Weight& fc() const { return fc_; }  // [hidden][features * hidden]
  // BACKLOG 33: the fc's NVFP4 copy for at most kMaxStreamRows rows (w null with the copies off); the BF16 fc stays
  // for the prompt's longer chunks.
  const Fp4Weight& fc4() const { return fc4_; }
  bool nvfp4_copies() const { return fc4_.w != nullptr; }
  double nvfp4_rel_sq_error() const { return rel_sq_error_; }  // the largest of the copies' relative squared errors
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
  Fp4Weight fc4_;
  double rel_sq_error_ = 0;
  const __nv_bfloat16* hidden_norm_ = nullptr;
  const __nv_bfloat16* norm_ = nullptr;
  const __nv_bfloat16* pred_ = nullptr;
  const __nv_bfloat16* succ_ = nullptr;
  std::vector<void*> allocations_;
  size_t bytes_ = 0;
};

}  // namespace ling
