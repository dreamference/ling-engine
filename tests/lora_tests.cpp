// The LoRA loader, the adapter library and the GPU pool's slots, on the host (src/core/lora.hpp): tiny adapters
// written as PEFT writes them, against a tiny model of two full-attention and two DeltaNet layers.
#include <stdlib.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/lora.hpp"

namespace {

int failures = 0;
#define EXPECT(c)                                                       \
  do {                                                                  \
    if (!(c)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

namespace fs = std::filesystem;
using ling::LoraProj;
using ling::LoraSite;

// The tiny model: hidden 8; layers 0 and 2 DeltaNet (in_qkv 10 rows: q 3, k 3, v 4; in_z 6; out 6 -> 8), layers 1
// and 3 full attention (q 12, k 4, v 4, o 6 -> 8).
ling::LoraModelShape tiny() {
  ling::LoraModelShape m;
  m.layers = 4;
  m.gdn_q = 3;
  m.gdn_k = 3;
  m.gdn_v = 4;
  m.shape = [](int layer, LoraProj p) -> std::pair<int, int> {
    const bool attn = layer % 2 == 1;
    switch (p) {
      case LoraProj::Q: return attn ? std::pair{12, 8} : std::pair{0, 0};
      case LoraProj::K: return attn ? std::pair{4, 8} : std::pair{0, 0};
      case LoraProj::V: return attn ? std::pair{4, 8} : std::pair{0, 0};
      case LoraProj::O: return attn ? std::pair{8, 6} : std::pair{0, 0};
      case LoraProj::InQkv: return attn ? std::pair{0, 0} : std::pair{10, 8};
      case LoraProj::InZ: return attn ? std::pair{0, 0} : std::pair{6, 8};
      case LoraProj::Out: return attn ? std::pair{0, 0} : std::pair{8, 6};
    }
    return {0, 0};
  };
  return m;
}

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> v;  // written as F32 (or BF16 when `bf16`)
  bool bf16 = false;
};

uint16_t bf16_of(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

float f32_of_bf16(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void write_safetensors(const std::string& path, const std::map<std::string, Tensor>& ts) {
  nlohmann::json h = nlohmann::json::object();
  std::string data;
  for (const auto& [name, t] : ts) {
    const size_t off = data.size();
    for (float f : t.v) {
      if (t.bf16) {
        const uint16_t b = bf16_of(f);
        data.append(reinterpret_cast<const char*>(&b), 2);
      } else {
        data.append(reinterpret_cast<const char*>(&f), 4);
      }
    }
    h[name] = {{"dtype", t.bf16 ? "BF16" : "F32"}, {"shape", t.shape}, {"data_offsets", {off, data.size()}}};
  }
  const std::string hs = h.dump();
  const uint64_t n = hs.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  f.write(data.data(), static_cast<std::streamsize>(data.size()));
}

// value(i) for element i of a tensor, distinct per tensor through `seed`.
Tensor filled(std::vector<int64_t> shape, float seed, bool bf16 = false) {
  Tensor t;
  t.shape = shape;
  t.bf16 = bf16;
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  for (int64_t i = 0; i < n; ++i) t.v.push_back(seed + 0.01f * static_cast<float>(i));
  return t;
}

const std::string kPre = "base_model.model.model.language_model.layers.";

void add_module(std::map<std::string, Tensor>& ts, int layer, const std::string& mod, int r, int N, int K, float seed,
                bool bf16 = false) {
  ts[kPre + std::to_string(layer) + "." + mod + ".lora_A.weight"] = filled({r, K}, seed, bf16);
  ts[kPre + std::to_string(layer) + "." + mod + ".lora_B.weight"] = filled({N, r}, seed + 100.f, bf16);
}

std::string write_adapter(const std::string& dir, const std::map<std::string, Tensor>& ts, nlohmann::json cfg) {
  fs::create_directories(dir);
  std::ofstream(dir + "/adapter_config.json") << cfg.dump();
  write_safetensors(dir + "/adapter_model.safetensors", ts);
  return dir;
}

nlohmann::json config(int r = 2, double alpha = 4) {
  return {{"peft_type", "LORA"}, {"r", r}, {"lora_alpha", alpha}, {"bias", "none"}, {"use_dora", false},
          {"rank_pattern", nlohmann::json::object()}, {"alpha_pattern", nlohmann::json::object()},
          {"modules_to_save", nullptr}, {"target_modules", {"q_proj"}}};
}

// The full HF form: every target in layers 0 (DeltaNet) and 1 (attention).
std::map<std::string, Tensor> full_form(int r = 2) {
  std::map<std::string, Tensor> ts;
  add_module(ts, 0, "linear_attn.in_proj_qkv", r, 10, 8, 1.f);
  add_module(ts, 0, "linear_attn.in_proj_z", r, 6, 8, 2.f);
  add_module(ts, 0, "linear_attn.out_proj", r, 8, 6, 3.f);
  add_module(ts, 1, "self_attn.q_proj", r, 12, 8, 4.f);
  add_module(ts, 1, "self_attn.k_proj", r, 4, 8, 5.f);
  add_module(ts, 1, "self_attn.v_proj", r, 4, 8, 6.f);
  add_module(ts, 1, "self_attn.o_proj", r, 8, 6, 7.f);
  return ts;
}

bool throws(const std::function<void()>& f, const std::string& part) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(part) != std::string::npos) return true;
    std::fprintf(stderr, "  threw \"%s\", expected \"%s\"\n", e.what(), part.c_str());
    return false;
  }
  std::fprintf(stderr, "  did not throw, expected \"%s\"\n", part.c_str());
  return false;
}

std::string tmp_root() {
  char t[] = "/tmp/ling-lora-tests-XXXXXX";
  const char* d = mkdtemp(t);
  return d ? std::string(d) : std::string();
}

void test_full_form(const std::string& root) {
  const std::string dir = write_adapter(root + "/lib/full", full_form(), config(2, 4));
  const ling::LoraWeights w = ling::load_lora_weights(dir, tiny());
  EXPECT(w.rank == 2);
  EXPECT(w.scale == 2.f);
  EXPECT(w.sites.size() == 4);
  const ling::LoraSiteWeights* g = w.find(0, LoraSite::Gdn);
  EXPECT(g && g->K == 8 && g->blocks == 2 && g->segs.size() == 2);
  if (g && g->segs.size() == 2) {
    EXPECT(g->segs[0].out == 0 && g->segs[0].n0 == 0 && g->segs[0].n1 == 10 && g->segs[0].block == 0);
    EXPECT(g->segs[1].out == 1 && g->segs[1].n0 == 0 && g->segs[1].n1 == 6 && g->segs[1].block == 1);
    EXPECT(g->a.size() == 2u * 2 * 8 && g->b.size() == (10u + 6) * 2);
    // A block 1 is in_proj_z's A (seed 2); B segment 1 is in_proj_z's B (seed 102), F32 rounded to BF16.
    EXPECT(g->a[16] == bf16_of(2.f) && g->a[17] == bf16_of(2.01f));
    EXPECT(g->b[g->segs[1].b_off] == bf16_of(102.f));
    EXPECT(std::fabs(f32_of_bf16(g->b[g->segs[1].b_off + 1]) - 102.01f) < 0.5f);
  }
  const ling::LoraSiteWeights* q = w.find(1, LoraSite::Qkv);
  EXPECT(q && q->blocks == 3 && q->segs.size() == 3);
  if (q && q->segs.size() == 3) {
    EXPECT(q->segs[0].out == 0 && q->segs[0].n1 == 12 && q->segs[0].block == 0);
    EXPECT(q->segs[1].out == 1 && q->segs[1].n1 == 4 && q->segs[1].block == 1);
    EXPECT(q->segs[2].out == 2 && q->segs[2].n1 == 4 && q->segs[2].block == 2);
    EXPECT(q->a[16] == bf16_of(5.f));  // block 1 is k_proj's A
  }
  const ling::LoraSiteWeights* o = w.find(1, LoraSite::O);
  EXPECT(o && o->K == 6 && o->blocks == 1 && o->segs.size() == 1 && o->segs[0].n1 == 8);
  EXPECT(w.find(0, LoraSite::Out) && w.find(0, LoraSite::Out)->K == 6);
  EXPECT(!w.find(2, LoraSite::Gdn) && !w.find(3, LoraSite::Qkv));
  EXPECT(w.bytes() == 2u * (2 * 2 * 8 + 16 * 2 + 2 * 6 + 8 * 2 + 3 * 2 * 8 + 20 * 2 + 2 * 6 + 8 * 2));
}

void test_split_form(const std::string& root) {
  std::map<std::string, Tensor> ts;
  add_module(ts, 2, "linear_attn.in_proj_v", 2, 4, 8, 3.f, true);
  add_module(ts, 2, "linear_attn.in_proj_q", 2, 3, 8, 1.f, true);
  add_module(ts, 2, "linear_attn.in_proj_k", 2, 3, 8, 2.f, true);
  add_module(ts, 2, "linear_attn.in_proj_z", 2, 6, 8, 4.f, true);
  nlohmann::json c = config(2, 4);
  c["use_rslora"] = true;
  const ling::LoraWeights w = ling::load_lora_weights(write_adapter(root + "/lib/split", ts, c), tiny());
  EXPECT(std::fabs(w.scale - 4.f / std::sqrt(2.f)) < 1e-6f);
  const ling::LoraSiteWeights* g = w.find(2, LoraSite::Gdn);
  EXPECT(g && g->blocks == 4 && g->segs.size() == 4);
  if (g && g->segs.size() == 4) {
    EXPECT(g->segs[0].out == 0 && g->segs[0].n0 == 0 && g->segs[0].n1 == 3 && g->segs[0].block == 0);
    EXPECT(g->segs[1].out == 0 && g->segs[1].n0 == 3 && g->segs[1].n1 == 6 && g->segs[1].block == 1);
    EXPECT(g->segs[2].out == 0 && g->segs[2].n0 == 6 && g->segs[2].n1 == 10 && g->segs[2].block == 2);
    EXPECT(g->segs[3].out == 1 && g->segs[3].n0 == 0 && g->segs[3].n1 == 6 && g->segs[3].block == 3);
    EXPECT(g->a[0] == bf16_of(1.f) && g->a[16] == bf16_of(2.f) && g->a[32] == bf16_of(3.f));  // q, k, v order
    EXPECT(g->b[g->segs[2].b_off] == bf16_of(103.f));
  }
  // The split form beside in_proj_qkv covers the same rows twice.
  add_module(ts, 2, "linear_attn.in_proj_qkv", 2, 10, 8, 9.f);
  EXPECT(throws([&] { ling::load_lora_weights(write_adapter(root + "/lib/both", ts, config()), tiny()); }, "overlaps"));
}

void test_refusals(const std::string& root) {
  int n = 0;
  auto refused = [&](std::map<std::string, Tensor> ts, nlohmann::json c, const std::string& why) {
    const std::string dir = write_adapter(root + "/bad" + std::to_string(n++), ts, c);
    return throws([&] { ling::load_lora_weights(dir, tiny()); }, why);
  };
  std::map<std::string, Tensor> mlp = full_form();
  add_module(mlp, 0, "mlp.gate_proj", 2, 16, 8, 1.f);
  EXPECT(refused(mlp, config(), "MLP"));
  std::map<std::string, Tensor> ab = full_form();
  add_module(ab, 0, "linear_attn.in_proj_a", 2, 4, 8, 1.f);
  EXPECT(refused(ab, config(), "in_proj_a"));
  std::map<std::string, Tensor> wrong_layer;
  add_module(wrong_layer, 0, "self_attn.q_proj", 2, 12, 8, 1.f);  // layer 0 is DeltaNet
  EXPECT(refused(wrong_layer, config(), "has no q_proj"));
  std::map<std::string, Tensor> beyond;
  add_module(beyond, 4, "self_attn.q_proj", 2, 12, 8, 1.f);
  EXPECT(refused(beyond, config(), "no such layer"));
  std::map<std::string, Tensor> shape;
  add_module(shape, 1, "self_attn.q_proj", 2, 11, 8, 1.f);
  EXPECT(refused(shape, config(), "lora_B: shape"));
  std::map<std::string, Tensor> ashape;
  add_module(ashape, 1, "self_attn.o_proj", 2, 8, 8, 1.f);  // o_proj reads 6
  EXPECT(refused(ashape, config(), "lora_A: shape"));
  std::map<std::string, Tensor> rank = full_form(2);
  EXPECT(refused(rank, config(3), "lora_A: shape"));
  std::map<std::string, Tensor> lone = full_form();
  lone.erase(kPre + "1.self_attn.k_proj.lora_B.weight");
  EXPECT(refused(lone, config(), "without lora_B"));
  std::map<std::string, Tensor> vis = full_form();
  vis["base_model.model.model.visual.blocks.layers.0.linear_attn.out_proj.lora_A.weight"] = filled({2, 6}, 1.f);
  EXPECT(refused(vis, config(), "visual"));
  std::map<std::string, Tensor> other = full_form();
  other["base_model.model.lm_head.lora_A.weight"] = filled({2, 8}, 1.f);
  EXPECT(refused(other, config(), "not a LoRA weight"));
  EXPECT(refused({}, config(), "no weights"));
  EXPECT(refused(full_form(17), config(17), "adapter rank 17"));
  nlohmann::json c = config();
  c["use_dora"] = true;
  EXPECT(refused(full_form(), c, "DoRA"));
  c = config();
  c["rank_pattern"] = {{"q_proj", 4}};
  EXPECT(refused(full_form(), c, "rank_pattern"));
  c = config();
  c["bias"] = "all";
  EXPECT(refused(full_form(), c, "biases"));
  c = config();
  c["modules_to_save"] = {"lm_head"};
  EXPECT(refused(full_form(), c, "modules_to_save"));
  c = config();
  c["peft_type"] = "IA3";
  EXPECT(refused(full_form(), c, "peft_type"));
  c = config();
  c.erase("lora_alpha");
  EXPECT(refused(full_form(), c, "lora_alpha"));
  // A truncated weight file.
  const std::string dir = write_adapter(root + "/trunc", full_form(), config());
  fs::resize_file(dir + "/adapter_model.safetensors", fs::file_size(dir + "/adapter_model.safetensors") - 4);
  EXPECT(throws([&] { ling::load_lora_weights(dir, tiny()); }, ""));
  EXPECT(throws([&] { ling::load_lora_weights(root + "/nothing", tiny()); }, "adapter_config.json"));
}

void test_library(const std::string& root) {
  const std::string lib = root + "/lib";
  write_adapter(lib + "/a", full_form(), config());
  std::map<std::string, Tensor> other = full_form();
  other[kPre + "1.self_attn.o_proj.lora_B.weight"].v[0] = 42.f;
  write_adapter(lib + "/a2", other, config());
  write_adapter(lib + "/b", full_form(), config());
  write_adapter(root + "/outside", full_form(), config());
  fs::create_symlink(root + "/outside", lib + "/escape");

  ling::LoraLibrary none("", 8, tiny(), {});
  EXPECT(!none.enabled());
  EXPECT(throws([&] { none.load("a", lib + "/a"); }, "without an adapter library"));

  ling::LoraLibrary l(lib + "/.", 2, tiny(), {});
  EXPECT(l.enabled() && l.root() == fs::canonical(lib).string());
  EXPECT(throws([&] { l.load("x", root + "/outside"); }, "not in the adapter library"));
  EXPECT(throws([&] { l.load("x", lib + "/../outside"); }, "not in the adapter library"));
  EXPECT(throws([&] { l.load("x", lib + "/escape"); }, "not in the adapter library"));
  EXPECT(throws([&] { l.load("x", lib); }, "not in the adapter library"));
  EXPECT(throws([&] { l.load("x", lib + "/missing"); }, "does not exist"));
  EXPECT(throws([&] { l.load("../a", lib + "/a"); }, "bad adapter name"));
  EXPECT(throws([&] { l.load("a b", lib + "/a"); }, "bad adapter name"));

  auto a = l.load("a", lib + "/a");
  EXPECT(a && a->name == "a" && a->id != 0 && !a->has_drafter && l.find("a") == a);
  EXPECT(throws([&] { l.load("a", lib + "/b"); }, "already loaded"));
  auto b = l.load("b", lib + "/b");
  EXPECT(b->id != a->id);  // the same weights under another name
  EXPECT(throws([&] { l.load("c", lib + "/a2"); }, "max_loaded_loras"));
  EXPECT(l.names() == (std::vector<std::string>{"a", "b"}));
  EXPECT(l.unload("a") && !l.unload("a") && !l.find("a"));
  EXPECT(a->target.rank == 2);  // a request holding it keeps it
  auto a2 = l.load("a", lib + "/a2");
  EXPECT(a2->id != a->id);  // other weights under the same name: another id, so no state is shared
  EXPECT(l.names() == (std::vector<std::string>{"b", "a"}));
  EXPECT(ling::lora_scope(7, nullptr) == 7 && ling::lora_scope(0, nullptr) == 0);
  EXPECT(ling::lora_scope(0, a.get()) != 0 && (ling::lora_scope(0, a.get()) & 1));
  EXPECT(ling::lora_scope(0, a.get()) != ling::lora_scope(0, a2.get()));
  EXPECT(ling::lora_scope(0, a.get()) != ling::lora_scope(5, a.get()));
  EXPECT(ling::lora_scope(5, a.get()) == ling::lora_scope(5, a.get()));

  // The drafter's adapter, from drafter/ in the package, checked against the drafter's shape.
  ling::LoraModelShape drafter;
  drafter.layers = 1;
  drafter.shape = [](int, LoraProj p) -> std::pair<int, int> {
    return p == LoraProj::Q ? std::pair{12, 8} : std::pair{0, 0};
  };
  ling::LoraLibrary ld(lib, 8, tiny(), drafter);
  std::map<std::string, Tensor> dt;
  add_module(dt, 0, "self_attn.q_proj", 2, 12, 8, 1.f);
  write_adapter(lib + "/d", full_form(), config());
  write_adapter(lib + "/d/drafter", dt, config(2, 2));
  auto d = ld.load("d", lib + "/d");
  EXPECT(d->has_drafter && d->drafter.scale == 1.f && d->drafter.find(0, LoraSite::Qkv));
  EXPECT(d->bytes() == d->target.bytes() + d->drafter.bytes());
  std::map<std::string, Tensor> bad;
  add_module(bad, 0, "self_attn.k_proj", 2, 12, 8, 1.f);
  write_adapter(lib + "/e", full_form(), config());
  write_adapter(lib + "/e/drafter", bad, config());
  EXPECT(throws([&] { ld.load("e", lib + "/e"); }, "has no k_proj"));
  EXPECT(!ld.find("e"));

  EXPECT(ling::lora_name_ok("my-adapter_1.2") && !ling::lora_name_ok("") && !ling::lora_name_ok(".hidden") &&
         !ling::lora_name_ok("a/b") && !ling::lora_name_ok("a+b") && !ling::lora_name_ok(std::string(65, 'a')) &&
         ling::lora_name_ok(std::string(64, 'a')));
}

void test_slots() {
  ling::LoraSlots s(3);
  bool up = false;
  EXPECT(s.acquire(10, &up) == 0 && up);
  EXPECT(s.acquire(10, &up) == 0 && !up && s.refs(0) == 2);
  EXPECT(s.acquire(20, &up) == 1 && up);
  EXPECT(s.acquire(30, &up) == 2 && up);
  EXPECT(s.acquire(40, &up) == -1);  // every slot in use
  s.release(1);
  s.release(2);
  // 20 (slot 1) was used before 30 (slot 2): it goes first.
  EXPECT(s.acquire(40, &up) == 1 && up && s.id(1) == 40 && s.slot_of(20) == -1);
  EXPECT(s.acquire(30, &up) == 2 && !up);  // still resident: no upload
  s.release(0);
  s.release(0);
  s.release(2);
  EXPECT(s.acquire(30, &up) == 2 && !up);  // resident beats an LRU pick before it
  EXPECT(s.acquire(50, &up) == 0 && up);
  bool threw = false;
  try {
    ling::LoraSlots t(1);
    t.release(0);
  } catch (const std::logic_error&) {
    threw = true;
  }
  EXPECT(threw);
}

}  // namespace

int main() {
  const std::string root = tmp_root();
  if (root.empty()) {
    std::fprintf(stderr, "no temporary directory\n");
    return 1;
  }
  try {
    test_full_form(root);
    test_split_form(root);
    test_refusals(root);
    test_library(root);
    test_slots();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    ++failures;
  }
  fs::remove_all(root);
  if (failures) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("lora tests passed\n");
  return 0;
}
