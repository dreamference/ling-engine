// The engine's LoRA pool (specs/DREAMFERENCE_LING_ENGINE_LORA.md §2 and §3): the GPU slots, each one buffer sized for
// a rank-16 adapter on every site, the descriptor table the kernels read, and a site's shrink and expand for the rows
// of the sequences on an adapter. A batch with no such row launches nothing, so the base path is main's bit for bit.
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/engine.hpp"

namespace ling {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("lora: ") + what + ": " + cudaGetErrorString(e));
}

// Every A and B starts on 256 bytes (128 BF16), so the shrink's paired loads stay aligned.
constexpr size_t kAlign = 128;
size_t aligned(size_t elems) { return (elems + kAlign - 1) / kAlign * kAlign; }

}  // namespace

void Engine::lora_alloc() {
  const ModelConfig& c = model_->config();
  const int R = kLoraMaxRank, H = c.hidden, C = c.lin_conv_channels(), Vs = c.lin_value_size();
  const int qsize = c.heads * c.head_dim, kvsize = c.kv_heads * c.head_dim;
  // The most one adapter can need: rank 16 on every site of every layer (the split DeltaNet form's four A blocks).
  size_t elems = 0;
  for (int li = 0; li < c.layers; ++li) {
    if (c.full_attention[li]) {
      elems += aligned(size_t(3) * R * H) + aligned(size_t(2 * qsize + 2 * kvsize) * R);  // Qkv
      elems += aligned(size_t(R) * qsize) + aligned(size_t(H) * R);                       // O
    } else {
      elems += aligned(size_t(kLoraMaxBlocks) * R * H) + aligned(size_t(C + Vs) * R);  // Gdn
      elems += aligned(size_t(R) * Vs) + aligned(size_t(H) * R);                      // Out
    }
  }
  lora_slot_bytes_ = elems * sizeof(__nv_bfloat16);
  for (int s = 0; s < opts_.lora_slots; ++s) lora_pool_.push_back(alloc<__nv_bfloat16>(elems));
  lora_sites_ = reinterpret_cast<kernels::LoraSiteDev*>(
      alloc<uint8_t>(size_t(opts_.lora_slots) * c.layers * kLoraSites * sizeof(kernels::LoraSiteDev)));
  lora_seq_dev_ = alloc<int>(kernels::kMaxSeqs);
  const std::vector<int> base(kernels::kMaxSeqs, -1);
  check(cudaMemcpy(lora_seq_dev_, base.data(), base.size() * sizeof(int), cudaMemcpyHostToDevice), "slots");
  std::fprintf(stderr, "lora: %d GPU slots of %zu MB (rank %d on every site)\n", opts_.lora_slots,
               lora_slot_bytes_ >> 20, R);
}

namespace {

// An adapter's weights laid out in one slot's buffer: each site's A and B offsets, and the descriptor table of
// `layers` layers. Throws (changing nothing) when the weights do not fit what the kernels or the slot take.
struct SlotLayout {
  std::vector<kernels::LoraSiteDev> table;
  std::vector<size_t> a_at, b_at;
};

SlotLayout lay_out(const LoraWeights& w, int layers, __nv_bfloat16* pool, size_t slot_bytes, const std::string& name) {
  SlotLayout l;
  l.table.resize(size_t(layers) * kLoraSites);
  size_t used = 0;
  for (const LoraSiteWeights& sw : w.sites) {
    if (sw.layer < 0 || sw.layer >= layers || static_cast<int>(sw.site) >= kLoraSites)
      throw std::runtime_error("lora: " + name + ": a site outside the model");
    if (sw.blocks * w.rank > kernels::kLoraRankRows || sw.segs.size() > size_t(kernels::kLoraSegs) || sw.K % 4)
      throw std::runtime_error("lora: " + name + ": a site the kernels cannot take");
    l.a_at.push_back(used);
    used += aligned(sw.a.size());
    l.b_at.push_back(used);
    used += aligned(sw.b.size());
  }
  if (used * sizeof(__nv_bfloat16) > slot_bytes) throw std::runtime_error("lora: " + name + " does not fit a GPU slot");
  for (size_t i = 0; i < w.sites.size(); ++i) {
    const LoraSiteWeights& sw = w.sites[i];
    kernels::LoraSiteDev& d = l.table[size_t(sw.layer) * kLoraSites + static_cast<size_t>(sw.site)];
    d.a = pool + l.a_at[i];
    d.b = pool + l.b_at[i];
    d.K = sw.K;
    d.rank = w.rank;
    d.blocks = sw.blocks;
    d.scale = w.scale;
    d.nsegs = static_cast<int>(sw.segs.size());
    for (int g = 0; g < d.nsegs; ++g) {
      const LoraSegment& s = sw.segs[static_cast<size_t>(g)];
      d.segs[g].out = s.out;
      d.segs[g].n0 = s.n0;
      d.segs[g].n1 = s.n1;
      d.segs[g].block = s.block;
      d.segs[g].b_off = static_cast<long long>(s.b_off);
    }
  }
  return l;
}

void copy_in(const LoraWeights& w, const SlotLayout& l, __nv_bfloat16* pool, kernels::LoraSiteDev* table_dev) {
  for (size_t i = 0; i < w.sites.size(); ++i) {
    const LoraSiteWeights& sw = w.sites[i];
    check(cudaMemcpy(pool + l.a_at[i], sw.a.data(), sw.a.size() * 2, cudaMemcpyHostToDevice), "upload A");
    check(cudaMemcpy(pool + l.b_at[i], sw.b.data(), sw.b.size() * 2, cudaMemcpyHostToDevice), "upload B");
  }
  check(cudaMemcpy(table_dev, l.table.data(), l.table.size() * sizeof(kernels::LoraSiteDev), cudaMemcpyHostToDevice),
        "upload table");
}

}  // namespace

void Engine::lora_alloc_drafter() {
  const DraftConfig& d = draft_->config();
  const int R = kLoraMaxRank, H = d.hidden, dq = d.heads * d.head_dim, dkv = d.kv_heads * d.head_dim;
  size_t elems = 0;
  for (int l = 0; l < d.layers; ++l)
    elems += aligned(size_t(3) * R * H) + aligned(size_t(dq + 2 * dkv) * R) + aligned(size_t(R) * dq) + aligned(size_t(H) * R);
  lora_dslot_bytes_ = elems * sizeof(__nv_bfloat16);
  for (int s = 0; s < opts_.lora_slots; ++s) lora_dpool_.push_back(alloc<__nv_bfloat16>(elems));
  const size_t table = size_t(opts_.lora_slots) * d.layers * kLoraSites * sizeof(kernels::LoraSiteDev);
  lora_dsites_ = reinterpret_cast<kernels::LoraSiteDev*>(alloc<uint8_t>(table));
  check(cudaMemset(lora_dsites_, 0, table), "drafter table");
  std::fprintf(stderr, "lora: the drafter's part of each slot, %zu MB\n", lora_dslot_bytes_ >> 20);
}

void Engine::lora_upload(int slot, const LoraAdapter& adapter) {
  if (slot < 0 || slot >= opts_.lora_slots) throw std::runtime_error("lora: no such GPU slot");
  const ModelConfig& c = model_->config();
  const size_t su = static_cast<size_t>(slot);
  // The layouts first, so an adapter that does not fit changes nothing. The drafter's table is written whatever the
  // adapter carries: an adapter without drafter/ empties it, so the slot's last adapter's drafter part is never read.
  const SlotLayout target = lay_out(adapter.target, c.layers, lora_pool_[su], lora_slot_bytes_, adapter.name);
  const int dlayers = lora_dsites_ ? draft_->config().layers : 0;
  const LoraWeights none;
  const LoraWeights& dw = adapter.has_drafter && dlayers > 0 ? adapter.drafter : none;
  const SlotLayout draft = dlayers > 0 ? lay_out(dw, dlayers, lora_dpool_[su], lora_dslot_bytes_, adapter.name + " (drafter)")
                                       : SlotLayout{};
  // Nothing may read the slot while it is rewritten: every stream idle (a slot is refilled only when no running
  // sequence holds it, so this waits for at most the step in flight).
  side_join();
  check(cudaDeviceSynchronize(), "upload");
  copy_in(adapter.target, target, lora_pool_[su], lora_sites_ + su * c.layers * kLoraSites);
  if (dlayers > 0) copy_in(dw, draft, lora_dpool_[su], lora_dsites_ + su * dlayers * kLoraSites);
}

LoraModelShape Engine::lora_shape() const {
  const ModelConfig c = model_->config();
  LoraModelShape m;
  m.layers = c.layers;
  m.gdn_q = m.gdn_k = c.lin_k_heads * c.lin_k_dim;
  m.gdn_v = c.lin_value_size();
  m.shape = [c](int layer, LoraProj p) -> std::pair<int, int> {
    const int H = c.hidden, qsize = c.heads * c.head_dim, kvsize = c.kv_heads * c.head_dim;
    const bool attn = c.full_attention[layer];
    switch (p) {  // {out, in}; {0, 0}: the layer has no such projection
      case LoraProj::Q: return attn ? std::pair{2 * qsize, H} : std::pair{0, 0};  // q and its gate
      case LoraProj::K:
      case LoraProj::V: return attn ? std::pair{kvsize, H} : std::pair{0, 0};
      case LoraProj::O: return attn ? std::pair{H, qsize} : std::pair{0, 0};
      case LoraProj::InQkv: return attn ? std::pair{0, 0} : std::pair{c.lin_conv_channels(), H};
      case LoraProj::InZ: return attn ? std::pair{0, 0} : std::pair{c.lin_value_size(), H};
      case LoraProj::Out: return attn ? std::pair{0, 0} : std::pair{H, c.lin_value_size()};
    }
    return {0, 0};
  };
  return m;
}

LoraModelShape Engine::lora_draft_shape() const {
  LoraModelShape m;
  if (!draft_ || opts_.lora_slots <= 0) return m;
  const DraftConfig d = draft_->config();
  m.layers = d.layers;
  m.shape = [d](int, LoraProj p) -> std::pair<int, int> {
    const int H = d.hidden, dq = d.heads * d.head_dim, dkv = d.kv_heads * d.head_dim;
    switch (p) {  // every layer is attention, with no output gate
      case LoraProj::Q: return {dq, H};
      case LoraProj::K:
      case LoraProj::V: return {dkv, H};
      case LoraProj::O: return {H, dq};
      default: return {0, 0};
    }
  };
  return m;
}

kernels::LoraRows Engine::lora_rows(std::span<const Seg> segs) const {
  kernels::LoraRows rows;
  if (opts_.lora_slots <= 0) return rows;
  bool any = false;
  const int n = static_cast<int>(std::min<size_t>(segs.size(), kernels::kMaxSeqs));
  for (int i = 0; i < n; ++i) {
    rows.row0[i] = segs[static_cast<size_t>(i)].row0;
    rows.row0[i + 1] = segs[static_cast<size_t>(i)].row0 + segs[static_cast<size_t>(i)].rows;
    rows.slot[i] = segs[static_cast<size_t>(i)].s->lora_slot;
    any = any || rows.slot[i] >= 0;
  }
  if (!any) return rows;  // n == 0: nothing launches
  rows.n = n;
  if (use_dev_pos_) rows.slot_dev = lora_seq_dev_;  // a captured graph: the slots written before each replay
  return rows;
}

void Engine::lora_site(const kernels::LoraRows& rows, int layer, LoraSite site, const float* x, int ldx, int M,
                       const kernels::LoraOuts& outs, int max_n) {
  if (rows.n == 0) return;
  const int stride = model_->config().layers * kLoraSites;
  const kernels::LoraSiteDev* d = lora_sites_ + size_t(layer) * kLoraSites + static_cast<size_t>(site);
  kernels::lora_shrink(x, ldx, M, rows, d, stride, opts_.lora_slots, lora_t_, stream_);
  kernels::lora_expand(lora_t_, M, rows, d, stride, opts_.lora_slots, outs, max_n, stream_);
}

kernels::LoraRows Engine::lora_draft_rows(std::span<Session* const> s, std::span<const int> rows) const {
  kernels::LoraRows r;
  if (!lora_dsites_) return r;
  std::vector<Seg> segs;
  int row0 = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    segs.push_back({s[i], row0, rows[i]});
    row0 += rows[i];
  }
  return lora_rows(segs);
}

void Engine::lora_draft_site(const kernels::LoraRows& rows, int layer, LoraSite site, const float* x, int ldx, int M,
                             const kernels::LoraOuts& outs, int max_n) {
  if (rows.n == 0) return;
  const int stride = draft_->config().layers * kLoraSites;
  const kernels::LoraSiteDev* d = lora_dsites_ + size_t(layer) * kLoraSites + static_cast<size_t>(site);
  kernels::lora_shrink(x, ldx, M, rows, d, stride, opts_.lora_slots, lora_t_, stream_);
  kernels::lora_expand(lora_t_, M, rows, d, stride, opts_.lora_slots, outs, max_n, stream_);
}

}  // namespace ling
