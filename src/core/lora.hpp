// LoRA adapters (specs/DREAMFERENCE_LING_ENGINE_LORA.md): a PEFT adapter of the target model, read and checked on
// the host, and the bookkeeping of the two levels that hold one (the loaded set, max_loaded_loras, and the GPU pool,
// max_loras_per_batch). This header is host code only, so the loader and both levels are tested on the host
// (tests/lora_tests.cpp); the engine uploads a slot's weights and runs the kernels (kernels.cuh, lora_*).
//
// What an adapter may target: the FP8 projections, which are the projections whose input the engine has in FP32 on
// every path: full attention's q_proj, k_proj, v_proj and o_proj (16 layers), and the Gated DeltaNet's in_proj_qkv
// (or its split form in_proj_q, in_proj_k, in_proj_v), in_proj_z and out_proj (48 layers). The NVFP4 MLP, the BF16
// in_proj_a and in_proj_b, the embedding, the head and the vision tower are refused, by name, at load.
//
// The forward adds B (A x) * scale to the base projection's FP32 output (PEFT's y = W x + B A x * alpha / r, with
// alpha / sqrt(r) for rsLoRA), for the rows of the sequences that use the adapter only: a row without one is never
// touched, so a run without adapters is bit for bit the engine without them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ling {

// The projections an adapter may target.
enum class LoraProj : uint8_t { Q, K, V, O, InQkv, InZ, Out };

// A layer's sites. Each is one shrink (x A^T into the rank space, every A block of the site stacked) and one expand
// (each segment's B into its rows of one output): the modules that read one input go together.
//   Qkv  full attention's q/k/v, from the input norm; outputs 0 q_proj's (q and gate), 1 k_proj's, 2 v_proj's
//   O    o_proj, from the gated attention output; output 0 the residual stream
//   Gdn  the DeltaNet's in_qkv and in_z, from the input norm; outputs 0 in_qkv's (q, k, v), 1 in_z's
//   Out  out_proj, from the gated norm's output; output 0 the residual stream
enum class LoraSite : uint8_t { Qkv = 0, O = 1, Gdn = 2, Out = 3 };
constexpr int kLoraSites = 4;
constexpr int kLoraMaxRank = 16;
constexpr int kLoraMaxBlocks = 4;  // A blocks of one site (the split form: in_proj_q, _k, _v and _z)
constexpr int kLoraMaxSegs = 4;    // B segments of one site
constexpr int kLoraMaxOuts = 3;    // outputs of one site

// One B: rows [n0, n1) of output `out`, from A block `block` (rank rows block * rank .. + rank).
struct LoraSegment {
  int out = 0;
  int n0 = 0, n1 = 0;
  int block = 0;
  size_t b_off = 0;  // its B, [n1 - n0][rank], at this element of the site's `b`
};

// One site of one layer: the stacked A, [blocks * rank][K], and every segment's B, as BF16 bits.
struct LoraSiteWeights {
  int layer = 0;
  LoraSite site = LoraSite::Qkv;
  int K = 0;
  int blocks = 0;
  std::vector<uint16_t> a;
  std::vector<uint16_t> b;
  std::vector<LoraSegment> segs;
};

// An adapter's weights for one model (the target, or the drafter).
struct LoraWeights {
  int rank = 0;
  float scale = 1.f;  // alpha / r, or alpha / sqrt(r) with use_rslora
  std::vector<LoraSiteWeights> sites;  // ordered by (layer, site); a site the adapter does not touch is absent
  size_t bytes() const;
  const LoraSiteWeights* find(int layer, LoraSite site) const;
};

// The shape {N, K} of a base projection; {0, 0}: the layer has no such projection.
using LoraShapeFn = std::function<std::pair<int, int>(int layer, LoraProj p)>;

// What the loader checks an adapter against: the model's layers and projection shapes, and the split of the
// DeltaNet's in_qkv rows (q, k, v; 0 refuses the split form).
struct LoraModelShape {
  int layers = 0;
  LoraShapeFn shape;
  int gdn_q = 0, gdn_k = 0, gdn_v = 0;
};

// Reads dir/adapter_config.json and dir/adapter_model.safetensors and checks them: PEFT's LORA type, one rank at
// most `max_rank` (no rank_pattern, alpha_pattern, DoRA, bias or modules_to_save), every tensor a known module's
// lora_A or lora_B with the base projection's shape, each A with its B. Throws std::runtime_error with the reason.
LoraWeights load_lora_weights(const std::string& dir, const LoraModelShape& model, int max_rank = kLoraMaxRank);

// A loaded adapter: its name, an id that changes with its weights (FNV-1a over the name and the weight file's bytes:
// the share scope its sessions, checkpoints and prefix store entries are keyed by, so a reloaded adapter never
// resumes from the old one's state), and its weights. The drafter's adapter (drafter/ in the package) is optional.
struct LoraAdapter {
  std::string name;
  std::string dir;
  uint64_t id = 0;
  LoraWeights target;
  bool has_drafter = false;
  LoraWeights drafter;
  size_t bytes() const { return target.bytes() + drafter.bytes(); }
};

// The share scope of a request on `adapter` (nullptr: the base) from its X-Ling-Share-Scope's: the base keeps the
// scope as it is; an adapter mixes its id in (low bit set, so never 0, the unscoped base requests' scope).
uint64_t lora_scope(uint64_t scope, const LoraAdapter* adapter);

// Whether `name` is an adapter name the API accepts: 1 to 64 of [A-Za-z0-9._-], not starting with '.'.
bool lora_name_ok(const std::string& name);

// The loaded set (max_loaded_loras): adapters by name, loaded only from directories under the library root (resolved
// with realpath, so neither ".." nor a symlink leaves it). A full set refuses a load (the caller unloads first); an
// unload drops the name at once, and a request already running on it keeps its shared_ptr until it ends.
class LoraLibrary {
 public:
  LoraLibrary(std::string root, int max_loaded, LoraModelShape target, LoraModelShape drafter);

  // Loads `name` from `path` (which must be under the root). Throws std::runtime_error with the reason: no library,
  // a bad name, a path outside the library, the name already loaded, the set full, or the adapter's own fault.
  std::shared_ptr<const LoraAdapter> load(const std::string& name, const std::string& path);
  bool unload(const std::string& name);  // false: not loaded
  std::shared_ptr<const LoraAdapter> find(const std::string& name) const;
  std::vector<std::string> names() const;  // in load order
  int max_loaded() const { return max_loaded_; }
  const std::string& root() const { return root_; }
  bool enabled() const { return !root_.empty(); }

 private:
  std::string root_;  // realpath of the library ("" : none, every load refused)
  int max_loaded_;
  LoraModelShape target_, drafter_;
  std::vector<std::shared_ptr<const LoraAdapter>> loaded_;
};

// The GPU pool's slots (max_loras_per_batch - 1: the base needs none). A slot holds one adapter's weights; a session
// on it holds a reference. acquire() returns the adapter's slot, or the least recently used unreferenced one (then
// `upload` is set: the caller copies the weights in before any forward reads them), or -1 when every slot holds
// another adapter some session uses (the request waits).
class LoraSlots {
 public:
  explicit LoraSlots(int n) : slots_(static_cast<size_t>(n)) {}
  int acquire(uint64_t id, bool* upload);
  void release(int slot);
  // The slot holds no adapter after all (its upload failed): the next acquire of any id may take it, and uploads.
  void forget(int slot) { slots_[static_cast<size_t>(slot)].id = 0; }
  int slot_of(uint64_t id) const;  // -1: not resident
  int refs(int slot) const { return slots_[static_cast<size_t>(slot)].refs; }
  uint64_t id(int slot) const { return slots_[static_cast<size_t>(slot)].id; }
  int size() const { return static_cast<int>(slots_.size()); }

 private:
  struct Slot {
    uint64_t id = 0;  // 0: empty
    int refs = 0;
    uint64_t used = 0;
  };
  std::vector<Slot> slots_;
  uint64_t clock_ = 0;
};

}  // namespace ling
