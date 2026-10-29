#include "core/lora.hpp"

#include <limits.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/safetensors.hpp"

namespace ling {

namespace {

uint64_t fnv_bytes(uint64_t h, const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}
constexpr uint64_t kFnvBasis = 1469598103934665603ull;

uint16_t f32_to_bf16(float f) {  // round to nearest even, as torch's .to(bfloat16)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);  // NaN stays NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

float f16_to_f32(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
  float f;
  if (exp == 0) {
    f = std::ldexp(static_cast<float>(man), -24);
  } else if (exp == 31) {
    f = man ? NAN : INFINITY;
  } else {
    f = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
  }
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u |= sign;
  std::memcpy(&f, &u, 4);
  return f;
}

// A tensor as BF16 bits (BF16 as is; F32 and F16 rounded as torch rounds them).
std::vector<uint16_t> to_bf16(const TensorView& t, const std::string& name) {
  const size_t n = static_cast<size_t>(t.numel());
  std::vector<uint16_t> out(n);
  if (t.dtype == "BF16") {
    if (t.bytes != n * 2) throw std::runtime_error(name + ": size does not match its shape");
    std::memcpy(out.data(), t.data, n * 2);
  } else if (t.dtype == "F32") {
    if (t.bytes != n * 4) throw std::runtime_error(name + ": size does not match its shape");
    for (size_t i = 0; i < n; ++i) {
      float f;
      std::memcpy(&f, t.data + i * 4, 4);
      out[i] = f32_to_bf16(f);
    }
  } else if (t.dtype == "F16") {
    if (t.bytes != n * 2) throw std::runtime_error(name + ": size does not match its shape");
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, t.data + i * 2, 2);
      out[i] = f32_to_bf16(f16_to_f32(h));
    }
  } else {
    throw std::runtime_error(name + ": dtype " + t.dtype + " (BF16, F16 or F32 only)");
  }
  return out;
}

LoraSite site_of(LoraProj p) {
  switch (p) {
    case LoraProj::Q:
    case LoraProj::K:
    case LoraProj::V: return LoraSite::Qkv;
    case LoraProj::O: return LoraSite::O;
    case LoraProj::InQkv:
    case LoraProj::InZ: return LoraSite::Gdn;
    case LoraProj::Out: return LoraSite::Out;
  }
  return LoraSite::Qkv;
}

// A module the adapter may name: its projection(s) and, per output segment, which output and which rows.
struct ModuleSpec {
  int order;  // its A block's place in the site (fixed, so a site's layout does not depend on the file)
  std::vector<std::pair<LoraProj, std::pair<int, int>>> parts;  // (projection, rows [n0, n1) of its output)
};

// The output a projection writes within its site.
int out_of(LoraProj p) {
  switch (p) {
    case LoraProj::K: return 1;
    case LoraProj::V: return 2;
    case LoraProj::InZ: return 1;
    default: return 0;
  }
}

ModuleSpec module_spec(const std::string& group, const std::string& mod, int layer, const LoraModelShape& m,
                       const std::string& key) {
  auto rows = [&](LoraProj p) {
    const auto [N, K] = m.shape(layer, p);
    if (N <= 0) throw std::runtime_error(key + ": layer " + std::to_string(layer) + " has no " + mod);
    return N;
  };
  if (group == "mlp")
    throw std::runtime_error(key + ": the MLP is NVFP4, and adapters on it are not served (attention and DeltaNet "
                                   "projections only)");
  if (group == "self_attn") {
    if (mod == "q_proj") return {0, {{LoraProj::Q, {0, rows(LoraProj::Q)}}}};
    if (mod == "k_proj") return {1, {{LoraProj::K, {0, rows(LoraProj::K)}}}};
    if (mod == "v_proj") return {2, {{LoraProj::V, {0, rows(LoraProj::V)}}}};
    if (mod == "o_proj") return {0, {{LoraProj::O, {0, rows(LoraProj::O)}}}};
  } else if (group == "linear_attn") {
    if (mod == "in_proj_qkv") return {0, {{LoraProj::InQkv, {0, rows(LoraProj::InQkv)}}}};
    if (mod == "in_proj_z") return {3, {{LoraProj::InZ, {0, rows(LoraProj::InZ)}}}};
    if (mod == "in_proj_qkvz") return {0, {{LoraProj::InQkv, {0, rows(LoraProj::InQkv)}}, {LoraProj::InZ, {0, rows(LoraProj::InZ)}}}};
    if (mod == "out_proj") return {0, {{LoraProj::Out, {0, rows(LoraProj::Out)}}}};
    if (mod == "in_proj_q" || mod == "in_proj_k" || mod == "in_proj_v") {
      const int N = rows(LoraProj::InQkv);
      if (m.gdn_q <= 0 || m.gdn_q + m.gdn_k + m.gdn_v != N)
        throw std::runtime_error(key + ": the split form needs in_proj_qkv's q, k and v sizes");
      if (mod == "in_proj_q") return {0, {{LoraProj::InQkv, {0, m.gdn_q}}}};
      if (mod == "in_proj_k") return {1, {{LoraProj::InQkv, {m.gdn_q, m.gdn_q + m.gdn_k}}}};
      return {2, {{LoraProj::InQkv, {m.gdn_q + m.gdn_k, N}}}};
    }
    if (mod == "in_proj_a" || mod == "in_proj_b")
      throw std::runtime_error(key + ": in_proj_a and in_proj_b are BF16, and adapters on them are not served");
  }
  throw std::runtime_error(key + ": not a module an adapter may target");
}

}  // namespace

size_t LoraWeights::bytes() const {
  size_t n = 0;
  for (const LoraSiteWeights& s : sites) n += (s.a.size() + s.b.size()) * sizeof(uint16_t);
  return n;
}

const LoraSiteWeights* LoraWeights::find(int layer, LoraSite site) const {
  for (const LoraSiteWeights& s : sites)
    if (s.layer == layer && s.site == site) return &s;
  return nullptr;
}

LoraWeights load_lora_weights(const std::string& dir, const LoraModelShape& model, int max_rank) {
  std::ifstream cf(dir + "/adapter_config.json");
  if (!cf) throw std::runtime_error("no adapter_config.json in the adapter");
  nlohmann::json cfg;
  try {
    cfg = nlohmann::json::parse(cf);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("adapter_config.json: ") + e.what());
  }
  if (!cfg.is_object()) throw std::runtime_error("adapter_config.json is not an object");
  auto flag = [&](const char* k) { return cfg.contains(k) && !cfg[k].is_null() && cfg[k] != false; };
  auto nonempty = [&](const char* k) { return cfg.contains(k) && !cfg[k].is_null() && !cfg[k].empty(); };
  if (cfg.contains("peft_type") && cfg["peft_type"] != "LORA")
    throw std::runtime_error("adapter_config.json: peft_type is not LORA");
  if (!cfg.contains("r") || !cfg["r"].is_number_integer()) throw std::runtime_error("adapter_config.json: no rank r");
  const int r = cfg["r"].get<int>();
  if (r < 1 || r > max_rank)
    throw std::runtime_error("adapter rank " + std::to_string(r) + " (1 to " + std::to_string(max_rank) + ")");
  if (!cfg.contains("lora_alpha") || !cfg["lora_alpha"].is_number())
    throw std::runtime_error("adapter_config.json: no lora_alpha");
  const double alpha = cfg["lora_alpha"].get<double>();
  if (flag("use_dora")) throw std::runtime_error("DoRA adapters are not served");
  if (nonempty("rank_pattern") || nonempty("alpha_pattern"))
    throw std::runtime_error("per-module ranks (rank_pattern, alpha_pattern) are not served");
  if (cfg.contains("bias") && cfg["bias"] != "none") throw std::runtime_error("adapter biases are not served");
  if (nonempty("modules_to_save")) throw std::runtime_error("modules_to_save is not served");
  if (flag("fan_in_fan_out")) throw std::runtime_error("fan_in_fan_out adapters are not served");
  if (cfg.contains("layers_to_transform") && !cfg["layers_to_transform"].is_null())
    throw std::runtime_error("layers_to_transform is not served");

  LoraWeights w;
  w.rank = r;
  w.scale = static_cast<float>(flag("use_rslora") ? alpha / std::sqrt(static_cast<double>(r)) : alpha / r);

  Checkpoint ck(dir, "adapter_model.safetensors");
  static const std::regex key_re(R"(^(.*\.)?layers\.(\d+)\.(self_attn|linear_attn|mlp)\.([a-z_]+)\.lora_(A|B)\.weight$)");
  struct Module {
    const TensorView* a = nullptr;
    const TensorView* b = nullptr;
    std::string name;
  };
  std::map<std::tuple<int, std::string, std::string>, Module> modules;
  std::string prefix;
  bool first = true;
  for (const auto& [key, view] : ck.tensors()) {
    std::smatch mt;
    if (!std::regex_match(key, mt, key_re)) throw std::runtime_error(key + ": not a LoRA weight this engine serves");
    const std::string pre = mt[1].str();
    for (const char* bad : {"visual", "vision", "mtp"})
      if (pre.find(bad) != std::string::npos) throw std::runtime_error(key + ": adapters on the " + bad + " part are not served");
    if (first) prefix = pre;
    else if (pre != prefix) throw std::runtime_error(key + ": tensor names with different prefixes");
    first = false;
    const std::string digits = mt[2].str();
    if (digits.size() > 4) throw std::runtime_error(key + ": no such layer");
    const int layer = std::stoi(digits);
    if (layer >= model.layers) throw std::runtime_error(key + ": no such layer");
    Module& m = modules[{layer, mt[3].str(), mt[4].str()}];
    m.name = key.substr(0, key.size() - std::string(".lora_A.weight").size());
    (mt[5].str() == "A" ? m.a : m.b) = &view;
  }
  if (modules.empty()) throw std::runtime_error("the adapter has no weights");

  // Per (layer, site): its modules in their fixed order.
  std::map<std::pair<int, int>, std::vector<std::pair<ModuleSpec, const Module*>>> sites;
  for (const auto& [k, m] : modules) {
    const auto& [layer, group, mod] = k;
    if (!m.a || !m.b) throw std::runtime_error(m.name + ": lora_A without lora_B or the other way round");
    ModuleSpec spec = module_spec(group, mod, layer, model, m.name);
    const LoraProj p0 = spec.parts[0].first;
    const int K = model.shape(layer, p0).second;
    if (m.a->shape != std::vector<int64_t>{r, K})
      throw std::runtime_error(m.name + ".lora_A: shape is not [" + std::to_string(r) + ", " + std::to_string(K) + "]");
    int64_t N = 0;
    for (const auto& part : spec.parts) N += part.second.second - part.second.first;
    if (m.b->shape != std::vector<int64_t>{N, r})
      throw std::runtime_error(m.name + ".lora_B: shape is not [" + std::to_string(N) + ", " + std::to_string(r) + "]");
    sites[{layer, static_cast<int>(site_of(p0))}].push_back({std::move(spec), &m});
  }
  for (auto& [ls, mods] : sites) {
    std::sort(mods.begin(), mods.end(), [](const auto& x, const auto& y) { return x.first.order < y.first.order; });
    LoraSiteWeights s;
    s.layer = ls.first;
    s.site = static_cast<LoraSite>(ls.second);
    s.K = model.shape(s.layer, mods[0].first.parts[0].first).second;
    s.blocks = static_cast<int>(mods.size());
    for (size_t bi = 0; bi < mods.size(); ++bi) {
      const auto& [spec, m] = mods[bi];
      if (bi > 0 && spec.order == mods[bi - 1].first.order)
        throw std::runtime_error(m->name + ": overlaps another module of layer " + std::to_string(s.layer));
      std::vector<uint16_t> a = to_bf16(*m->a, m->name + ".lora_A");
      s.a.insert(s.a.end(), a.begin(), a.end());
      std::vector<uint16_t> b = to_bf16(*m->b, m->name + ".lora_B");
      size_t row = 0;
      for (const auto& [proj, nr] : spec.parts) {
        LoraSegment g;
        g.out = out_of(proj);
        g.n0 = nr.first;
        g.n1 = nr.second;
        g.block = static_cast<int>(bi);
        g.b_off = s.b.size();
        for (const LoraSegment& o : s.segs)
          if (o.out == g.out && g.n0 < o.n1 && o.n0 < g.n1)
            throw std::runtime_error(m->name + ": overlaps another module of layer " + std::to_string(s.layer));
        const size_t n = size_t(g.n1 - g.n0) * r;
        s.b.insert(s.b.end(), b.begin() + static_cast<ptrdiff_t>(row), b.begin() + static_cast<ptrdiff_t>(row + n));
        row += n;
        s.segs.push_back(g);
      }
    }
    if (s.blocks > kLoraMaxBlocks || s.segs.size() > static_cast<size_t>(kLoraMaxSegs))
      throw std::runtime_error("too many modules in layer " + std::to_string(s.layer));
    w.sites.push_back(std::move(s));
  }
  return w;
}

uint64_t lora_scope(uint64_t scope, const LoraAdapter* adapter) {
  if (!adapter) return scope;
  uint64_t h = fnv_bytes(kFnvBasis, &scope, sizeof scope);
  h = fnv_bytes(h, &adapter->id, sizeof adapter->id);
  return h | 1;
}

bool lora_name_ok(const std::string& name) {
  if (name.empty() || name.size() > 64 || name[0] == '.') return false;
  return std::all_of(name.begin(), name.end(), [](char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '.' ||
           ch == '_' || ch == '-';
  });
}

LoraLibrary::LoraLibrary(std::string root, int max_loaded, LoraModelShape target, LoraModelShape drafter)
    : max_loaded_(max_loaded), target_(std::move(target)), drafter_(std::move(drafter)) {
  if (!root.empty()) {
    char buf[PATH_MAX];
    if (!realpath(root.c_str(), buf)) throw std::runtime_error("--lora-library " + root + ": no such directory");
    root_ = buf;
  }
}

std::shared_ptr<const LoraAdapter> LoraLibrary::load(const std::string& name, const std::string& path) {
  if (root_.empty()) throw std::runtime_error("the engine was started without an adapter library (--lora-library)");
  if (!lora_name_ok(name)) throw std::runtime_error("bad adapter name (1 to 64 of A-Z a-z 0-9 . _ -)");
  char buf[PATH_MAX];
  if (!realpath(path.c_str(), buf)) throw std::runtime_error("the adapter's path does not exist");
  const std::string real = buf;
  if (real.size() <= root_.size() || real.compare(0, root_.size(), root_) != 0 || real[root_.size()] != '/')
    throw std::runtime_error("the adapter's path is not in the adapter library");
  if (find(name)) throw std::runtime_error("adapter " + name + " is already loaded");
  if (static_cast<int>(loaded_.size()) >= max_loaded_)
    throw std::runtime_error("the engine holds " + std::to_string(max_loaded_) + " adapters (max_loaded_loras); unload one first");
  auto a = std::make_shared<LoraAdapter>();
  a->name = name;
  a->dir = real;
  a->target = load_lora_weights(real, target_);
  if (drafter_.layers > 0 && std::ifstream(real + "/drafter/adapter_config.json")) {
    a->drafter = load_lora_weights(real + "/drafter", drafter_);
    a->has_drafter = true;
  }
  uint64_t h = fnv_bytes(kFnvBasis, name.data(), name.size());
  for (const char* f : {"/adapter_model.safetensors", "/drafter/adapter_model.safetensors"}) {
    if (!a->has_drafter && std::strcmp(f, "/adapter_model.safetensors") != 0) continue;
    MappedFile mf(real + f);
    h = fnv_bytes(h, mf.data(), mf.size());
  }
  a->id = h ? h : 1;
  loaded_.push_back(a);
  return a;
}

bool LoraLibrary::unload(const std::string& name) {
  auto it = std::find_if(loaded_.begin(), loaded_.end(), [&](const auto& a) { return a->name == name; });
  if (it == loaded_.end()) return false;
  loaded_.erase(it);
  return true;
}

std::shared_ptr<const LoraAdapter> LoraLibrary::find(const std::string& name) const {
  for (const auto& a : loaded_)
    if (a->name == name) return a;
  return nullptr;
}

std::vector<std::string> LoraLibrary::names() const {
  std::vector<std::string> out;
  for (const auto& a : loaded_) out.push_back(a->name);
  return out;
}

int LoraSlots::acquire(uint64_t id, bool* upload) {
  *upload = false;
  int pick = -1;
  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].id == id) {
      slots_[i].refs += 1;
      slots_[i].used = ++clock_;
      return static_cast<int>(i);
    }
    if (slots_[i].refs > 0) continue;
    // An empty slot first (used 0, the smallest), else the least recently used.
    if (pick < 0 || slots_[i].used < slots_[size_t(pick)].used) pick = static_cast<int>(i);
  }
  if (pick < 0) return -1;
  Slot& s = slots_[size_t(pick)];
  s.id = id;
  s.refs = 1;
  s.used = ++clock_;
  *upload = true;
  return pick;
}

void LoraSlots::release(int slot) {
  Slot& s = slots_[static_cast<size_t>(slot)];
  if (s.refs <= 0) throw std::logic_error("LoraSlots::release without acquire");
  s.refs -= 1;
}

int LoraSlots::slot_of(uint64_t id) const {
  for (size_t i = 0; i < slots_.size(); ++i)
    if (slots_[i].id == id) return static_cast<int>(i);
  return -1;
}

}  // namespace ling
