#include "core/model.hpp"

#include "core/kernels.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

#include "core/safetensors.hpp"

namespace ling {
namespace {

const std::string kPrefix = "model.language_model.";

void check(cudaError_t e, const std::string& what) {
  if (e != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(e));
}

}  // namespace

ModelConfig ModelConfig::from_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  nlohmann::json root = nlohmann::json::parse(f);
  const nlohmann::json& t = root.contains("text_config") ? root["text_config"] : root;
  ModelConfig c;
  c.hidden = t["hidden_size"];
  c.intermediate = t["intermediate_size"];
  c.layers = t["num_hidden_layers"];
  c.vocab = t["vocab_size"];
  c.heads = t["num_attention_heads"];
  c.kv_heads = t["num_key_value_heads"];
  c.head_dim = t["head_dim"];
  double partial = t.value("partial_rotary_factor", 1.0);
  double theta = 10000.0;
  if (t.contains("rope_parameters")) {
    partial = t["rope_parameters"].value("partial_rotary_factor", partial);
    theta = t["rope_parameters"].value("rope_theta", theta);
  } else {
    theta = t.value("rope_theta", theta);
  }
  c.rotary_dim = static_cast<int>(c.head_dim * partial);
  c.rope_theta = static_cast<float>(theta);
  c.eps = t.value("rms_norm_eps", 1e-6);
  c.lin_k_heads = t["linear_num_key_heads"];
  c.lin_v_heads = t["linear_num_value_heads"];
  c.lin_k_dim = t["linear_key_head_dim"];
  c.lin_v_dim = t["linear_value_head_dim"];
  c.conv_kernel = t["linear_conv_kernel_dim"];
  for (const auto& lt : t["layer_types"]) c.full_attention.push_back(lt.get<std::string>() == "full_attention");
  c.eos_token = t.value("eos_token_id", 0);
  if (static_cast<int>(c.full_attention.size()) != c.layers) throw std::runtime_error("layer_types size mismatch");
  if (c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.conv_kernel != 4 || c.head_dim != 256)
    throw std::runtime_error("v0 supports linear head dims 128, conv kernel 4 and attention head dim 256 only");
  return c;
}

Model::Model(const std::string& dir) : dir_(dir) {
  cfg_ = ModelConfig::from_file(dir + "/config.json");
  Checkpoint ck(dir);
  embed_ = static_cast<const __nv_bfloat16*>(upload(ck, kPrefix + "embed_tokens.weight", "BF16"));
  final_norm_ = static_cast<const __nv_bfloat16*>(upload(ck, kPrefix + "norm.weight", "BF16"));
  lm_head_ = fp4(ck, "lm_head");
  for (int i = 0; i < cfg_.layers; ++i) {
    const std::string p = kPrefix + "layers." + std::to_string(i) + ".";
    LayerWeights L;
    L.full = cfg_.full_attention[i];
    L.input_norm = static_cast<const __nv_bfloat16*>(upload(ck, p + "input_layernorm.weight", "BF16"));
    L.post_norm = static_cast<const __nv_bfloat16*>(upload(ck, p + "post_attention_layernorm.weight", "BF16"));
    L.gate = fp4(ck, p + "mlp.gate_proj");
    L.up = fp4(ck, p + "mlp.up_proj");
    L.down = fp4(ck, p + "mlp.down_proj");
    if (L.full) {
      L.q = fp8(ck, p + "self_attn.q_proj");
      L.k = fp8(ck, p + "self_attn.k_proj");
      L.v = fp8(ck, p + "self_attn.v_proj");
      L.o = fp8(ck, p + "self_attn.o_proj");
      L.q_norm = static_cast<const __nv_bfloat16*>(upload(ck, p + "self_attn.q_norm.weight", "BF16"));
      L.k_norm = static_cast<const __nv_bfloat16*>(upload(ck, p + "self_attn.k_norm.weight", "BF16"));
    } else {
      const std::string a = p + "linear_attn.";
      L.in_qkv = fp8(ck, a + "in_proj_qkv");
      L.in_z = fp8(ck, a + "in_proj_z");
      L.out = fp8(ck, a + "out_proj");
      L.in_a = bf16(ck, a + "in_proj_a.weight");
      L.in_b = bf16(ck, a + "in_proj_b.weight");
      L.conv = static_cast<const __nv_bfloat16*>(upload(ck, a + "conv1d.weight", "BF16"));
      L.A_log = static_cast<const __nv_bfloat16*>(upload(ck, a + "A_log", "BF16"));
      L.dt_bias = static_cast<const __nv_bfloat16*>(upload(ck, a + "dt_bias", "BF16"));
      L.lin_norm = static_cast<const __nv_bfloat16*>(upload(ck, a + "norm.weight", "BF16"));
      if (L.in_qkv.N != cfg_.lin_conv_channels() || L.in_z.N != cfg_.lin_value_size())
        throw std::runtime_error("unexpected linear-attention projection shapes in layer " + std::to_string(i));
    }
    layers_.push_back(L);
  }
}

Model::~Model() {
  for (void* p : allocations_) cudaFree(p);
}

const void* Model::upload(const Checkpoint& ck, const std::string& name, const std::string& dtype) {
  const TensorView& t = ck.get(name);
  if (t.dtype != dtype) throw std::runtime_error(name + " is " + t.dtype + ", expected " + dtype);
  void* d = nullptr;
  check(cudaMalloc(&d, t.bytes), "cudaMalloc " + name);
  check(cudaMemcpy(d, t.data, t.bytes, cudaMemcpyHostToDevice), "upload " + name);
  allocations_.push_back(d);
  bytes_ += t.bytes;
  return d;
}

float Model::scalar(const Checkpoint& ck, const std::string& name) {
  const TensorView& t = ck.get(name);
  if (t.dtype != "F32" || t.bytes != 4) throw std::runtime_error(name + " is not an F32 scalar");
  float v;
  std::memcpy(&v, t.data, 4);
  return v;
}

const uint8_t* upload_tiled(const TensorView& values, const TensorView* scales, int N, int K, size_t* bytes) {
  auto to_device = [](const TensorView& t) {
    void* d = nullptr;
    check(cudaMalloc(&d, t.bytes), "cudaMalloc staging");
    check(cudaMemcpy(d, t.data, t.bytes, cudaMemcpyHostToDevice), "upload staging");
    return static_cast<uint8_t*>(d);
  };
  const size_t tiled = scales ? kernels::tiled_bytes_nvfp4(N, K) : kernels::tiled_bytes_fp8(N, K);
  void* out = nullptr;
  check(cudaMalloc(&out, tiled), "cudaMalloc tiled");
  uint8_t* v = to_device(values);
  uint8_t* sc = scales ? to_device(*scales) : nullptr;
  if (scales) kernels::retile_nvfp4(v, sc, static_cast<uint8_t*>(out), N, K, nullptr);
  else kernels::retile_fp8(v, static_cast<uint8_t*>(out), N, K, nullptr);
  check(cudaDeviceSynchronize(), "retile");
  cudaFree(v);
  if (sc) cudaFree(sc);
  *bytes += tiled;
  return static_cast<const uint8_t*>(out);
}

Fp4Upload upload_nvfp4(const TensorView& values, const TensorView& scales, int N, int K, size_t* bytes,
                       std::vector<void*>& owned) {
  static const bool codes = [] {
    const char* e = std::getenv("LING_SCALE_CODES");
    return !(e && std::atoi(e) == 0);
  }();
  uint8_t table[kernels::kScaleCodes], code_of[256];
  Fp4Upload u;
  if (!codes || !kernels::scale_code_table(scales.data, scales.bytes, table, code_of)) {
    u.w = upload_tiled(values, &scales, N, K, bytes);
    owned.push_back(const_cast<uint8_t*>(u.w));
    return u;
  }
  auto dev = [&](size_t n, const void* host) {
    void* d = nullptr;
    check(cudaMalloc(&d, n), "cudaMalloc coded");
    if (host) check(cudaMemcpy(d, host, n, cudaMemcpyHostToDevice), "upload coded");
    return static_cast<uint8_t*>(d);
  };
  uint8_t* v = dev(values.bytes, values.data);
  uint8_t* sc = dev(scales.bytes, scales.data);
  uint8_t* map = dev(256, code_of);
  uint8_t* blob = dev(kernels::tiled_bytes_nvfp4_coded(N, K), nullptr);
  uint8_t* raw = dev(scales.bytes, nullptr);
  uint8_t* tab = dev(kernels::kScaleCodes, table);
  kernels::retile_nvfp4_coded(v, sc, map, blob, raw, N, K, nullptr);
  check(cudaDeviceSynchronize(), "retile coded");
  cudaFree(v), cudaFree(sc), cudaFree(map);
  owned.push_back(blob), owned.push_back(raw), owned.push_back(tab);
  *bytes += kernels::tiled_bytes_nvfp4_coded(N, K) + scales.bytes + kernels::kScaleCodes;
  u.w = blob;
  u.table = tab;
  u.scales = raw;
  return u;
}

Nvfp4Host quantize_nvfp4(const __nv_bfloat16* w, int N, int K) {
  if (N <= 0 || K <= 0 || K % 16) throw std::runtime_error("quantize_nvfp4: K must be a positive multiple of 16");
  static constexpr float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  static constexpr float kBounds[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
  auto bf = [&](size_t i) { return __bfloat162float(w[i]); };
  auto e4m3 = [](uint8_t c) {
    __nv_fp8_e4m3 v;
    v.__x = c;
    return static_cast<float>(v);
  };
  const int nt = std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 16);
  // Rows [r0, r1) of each thread's share; fn(t, r0, r1).
  auto parallel = [&](auto&& fn) {
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t) {
      const int r0 = static_cast<int>(int64_t(N) * t / nt), r1 = static_cast<int>(int64_t(N) * (t + 1) / nt);
      ts.emplace_back([&fn, t, r0, r1] { fn(t, r0, r1); });
    }
    for (std::thread& th : ts) th.join();
  };
  std::vector<float> amax(nt, 0.f);
  parallel([&](int t, int r0, int r1) {
    float m = 0.f;
    for (size_t i = size_t(r0) * K; i < size_t(r1) * K; ++i) m = std::max(m, std::fabs(bf(i)));
    amax[t] = m;
  });
  const float mx = *std::max_element(amax.begin(), amax.end());
  Nvfp4Host q;
  q.scale2 = mx > 0.f ? mx / (448.f * 6.f) : 1.f;
  q.values.assign(size_t(N) * K / 2, 0);
  q.scales.assign(size_t(N) * K / 16, 0);
  std::vector<double> err(nt, 0.0), ref(nt, 0.0);
  const float g = q.scale2;
  parallel([&](int t, int r0, int r1) {
    double e = 0, s2 = 0;
    for (int r = r0; r < r1; ++r)
      for (int b = 0; b < K / 16; ++b) {
        const size_t base = size_t(r) * K + size_t(b) * 16;
        float bm = 0.f;
        for (int j = 0; j < 16; ++j) bm = std::max(bm, std::fabs(bf(base + j)));
        const float sv = std::max(bm / 6.f / g, 0x1p-9f);
        const uint8_t sc = __nv_cvt_float_to_fp8(sv, __NV_SATFINITE, __NV_E4M3);
        q.scales[size_t(r) * (K / 16) + b] = sc;
        const float s = e4m3(sc) * g;
        for (int j = 0; j < 16; ++j) {
          const float x = bf(base + j), y = x / s, a = std::fabs(y);
          uint8_t c = 0;
          while (c < 7 && kBounds[c] < a) ++c;
          const float dq = (x < 0.f ? -kE2M1[c] : kE2M1[c]) * s;
          if (x < 0.f && c > 0) c |= 8;
          q.values[(base + j) / 2] |= static_cast<uint8_t>(j % 2 ? c << 4 : c);
          e += double(dq - x) * double(dq - x);
          s2 += double(x) * double(x);
        }
      }
    err[t] = e, ref[t] = s2;
  });
  double e = 0, s2 = 0;
  for (int t = 0; t < nt; ++t) e += err[t], s2 += ref[t];
  q.rel_sq_error = s2 > 0 ? e / s2 : 0;
  return q;
}

Fp4Weight Model::fp4(const Checkpoint& ck, const std::string& prefix) {
  Fp4Weight w;
  const TensorView& t = ck.get(prefix + ".weight");
  const TensorView& s = ck.get(prefix + ".weight_scale");
  if (t.dtype != "U8" || s.dtype != "F8_E4M3") throw std::runtime_error(prefix + ": not NVFP4");
  w.N = static_cast<int>(t.shape[0]);
  w.K = static_cast<int>(t.shape[1] * 2);
  const Fp4Upload u = upload_nvfp4(t, s, w.N, w.K, &bytes_, allocations_);
  w.w = u.w, w.table = u.table, w.scales = u.scales;
  w.scale2 = scalar(ck, prefix + ".weight_scale_2");
  w.in_scale = scalar(ck, prefix + ".input_scale");
  return w;
}

Fp8Weight Model::fp8(const Checkpoint& ck, const std::string& prefix) {
  Fp8Weight w;
  const TensorView& t = ck.get(prefix + ".weight");
  if (t.dtype != "F8_E4M3") throw std::runtime_error(prefix + ": not FP8");
  w.N = static_cast<int>(t.shape[0]);
  w.K = static_cast<int>(t.shape[1]);
  w.w = upload_tiled(t, nullptr, w.N, w.K, &bytes_);
  allocations_.push_back(const_cast<uint8_t*>(w.w));
  w.scale = scalar(ck, prefix + ".weight_scale");
  w.in_scale = scalar(ck, prefix + ".input_scale");
  return w;
}

Bf16Weight Model::bf16(const Checkpoint& ck, const std::string& name) {
  Bf16Weight w;
  const TensorView& t = ck.get(name);
  w.N = static_cast<int>(t.shape[0]);
  w.K = static_cast<int>(t.shape[1]);
  w.w = static_cast<const __nv_bfloat16*>(upload(ck, name, "BF16"));
  return w;
}

}  // namespace ling
