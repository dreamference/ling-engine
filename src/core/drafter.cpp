#include "core/drafter.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/safetensors.hpp"

namespace ling {
namespace {

void check(cudaError_t e, const std::string& what) {
  if (e != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(e));
}

}  // namespace

DraftConfig DraftConfig::from_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  nlohmann::json t = nlohmann::json::parse(f);
  DraftConfig c;
  c.hidden = t["hidden_size"];
  c.intermediate = t["intermediate_size"];
  c.layers = t["num_hidden_layers"];
  c.heads = t["num_attention_heads"];
  c.kv_heads = t["num_key_value_heads"];
  c.head_dim = t["head_dim"];
  c.eps = t.value("rms_norm_eps", 1e-6);
  c.rope_theta = t.contains("rope_parameters") ? t["rope_parameters"].value("rope_theta", 1e6) : t.value("rope_theta", 1e6);
  c.sliding_window = t.value("sliding_window", 0);
  const nlohmann::json& d = t["dflash_config"];
  c.trained_block = d.value("block_size", 8);
  c.mask_token = d["mask_token_id"];
  c.conv_taps = d.value("conv_kernel_size", 0);
  c.conv_groups = c.hidden / d.value("conv_group_size", 16);
  c.selector_rank = d.value("selector_rank", 0);
  c.selector_top_k = d.value("selector_top_k", 0);
  c.target_layers = d["target_layer_ids"].get<std::vector<int>>();
  if (c.head_dim != 128 || c.conv_taps != 2 || c.selector_rank != 256 || c.selector_top_k != 16 || c.heads != 4 * c.kv_heads)
    throw std::runtime_error("unsupported drafter shape (expected head dim 128, 2 conv taps, selector rank 256, top-k 16, "
                             "4 query heads per KV head)");
  for (const auto& lt : t["layer_types"])
    if (lt.get<std::string>() != "sliding_attention") throw std::runtime_error("drafter layers must be sliding_attention");
  if (t.value("is_causal", true)) throw std::runtime_error("expected a non-causal (block diffusion) drafter");
  return c;
}

DraftModel::DraftModel(const std::string& dir) {
  cfg_ = DraftConfig::from_file(dir + "/config.json");
  Checkpoint ck(dir);
  auto upload = [&](const std::string& name, const std::string& dtype) -> const void* {
    const TensorView& t = ck.get(name);
    if (t.dtype != dtype) throw std::runtime_error(name + " is " + t.dtype + ", expected " + dtype);
    void* p = nullptr;
    check(cudaMalloc(&p, t.bytes), "cudaMalloc " + name);
    check(cudaMemcpy(p, t.data, t.bytes, cudaMemcpyHostToDevice), "upload " + name);
    allocations_.push_back(p);
    bytes_ += t.bytes;
    return p;
  };
  auto bf = [&](const std::string& name) { return static_cast<const __nv_bfloat16*>(upload(name, "BF16")); };
  auto bf_w = [&](const std::string& name) {
    Bf16Weight w;
    const TensorView& t = ck.get(name);
    w.N = static_cast<int>(t.shape[0]);
    w.K = static_cast<int>(t.shape[1]);
    w.w = bf(name);
    return w;
  };
  auto fp4 = [&](const std::string& prefix) {
    Fp4Weight w;
    const TensorView& t = ck.get(prefix + ".weight");
    w.N = static_cast<int>(t.shape[0]);
    w.K = static_cast<int>(t.shape[1] * 2);
    w.w = static_cast<const uint8_t*>(upload(prefix + ".weight", "U8"));
    w.scale = static_cast<const uint8_t*>(upload(prefix + ".weight_scale", "F8_E4M3"));
    const TensorView& s2 = ck.get(prefix + ".weight_scale_2");
    if (s2.dtype != "F32" || s2.bytes != 4) throw std::runtime_error(prefix + ".weight_scale_2 is not an F32 scalar");
    std::memcpy(&w.scale2, s2.data, 4);
    return w;
  };
  fc_ = bf_w("fc.weight");
  hidden_norm_ = bf("hidden_norm.weight");
  norm_ = bf("norm.weight");
  sel_proj_ = bf_w("candidate_selector.hidden_projection.weight");
  pred_ = bf("candidate_selector.predecessor_codebook");
  succ_ = bf("candidate_selector.successor_codebook");
  if (fc_.K != cfg_.hidden * static_cast<int>(cfg_.target_layers.size()))
    throw std::runtime_error("drafter fc does not match its target layer count");
  for (int i = 0; i < cfg_.layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    DraftLayer L;
    L.input_norm = bf(p + "input_layernorm.weight");
    L.post_norm = bf(p + "post_attention_layernorm.weight");
    L.q = fp4(p + "self_attn.q_proj");
    L.k = fp4(p + "self_attn.k_proj");
    L.v = fp4(p + "self_attn.v_proj");
    L.o = fp4(p + "self_attn.o_proj");
    L.gate = fp4(p + "mlp.gate_proj");
    L.up = fp4(p + "mlp.up_proj");
    L.down = fp4(p + "mlp.down_proj");
    L.q_norm = bf(p + "self_attn.q_norm.weight");
    L.k_norm = bf(p + "self_attn.k_norm.weight");
    L.attn_kproj = bf_w(p + "attention_conv.kernel_projection.weight");
    L.mlp_kproj = bf_w(p + "mlp_conv.kernel_projection.weight");
    L.attn_base = bf(p + "attention_conv.base_kernel");
    L.mlp_base = bf(p + "mlp_conv.base_kernel");
    if (L.attn_kproj.N != 2 * cfg_.conv_taps * cfg_.conv_groups) throw std::runtime_error("drafter conv projection shape");
    layers_.push_back(L);
  }
}

DraftModel::~DraftModel() {
  for (void* p : allocations_) cudaFree(p);
}

}  // namespace ling
