// Admission control on host memory (BATCHING §5.6, phase 0; the owner's decisions of 2026-10-27, after ling-serve
// with eight sessions froze node-b). On GB10 a CUDA allocation is host memory that no cgroup is charged for, so the
// container's --memory and earlyoom do not stop the engine in time; the engine has to refuse work itself.
//
//  - At start, before anything is allocated: the engine's own footprint (weights, the sessions' flat slots at
//    --max-context, the drafter's rings, scratch) must leave --host-reserve-mb of MemAvailable, else ling-serve
//    refuses to start and says by how much. The prefix KV store already sizes itself from what is left above its own
//    reserve (BACKLOG 35) and is not counted here.
//  - While serving: a generation request that arrives while MemAvailable is below --admit-reserve-mb is answered
//    HTTP 503 (code host_memory_low, Retry-After) before anything is streamed or queued.
//
// Neither touches a number the engine computes: a request that is admitted runs exactly as before.
// Host-only arithmetic, tested in ling-api-tests.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "core/model.hpp"

namespace ling::serve {

// The drafter's context ring per session (M4 stage 1): 5 layers x K and V x 8,224 positions x 8 heads x 128 x BF16.
inline constexpr size_t kDraftRingBytes = size_t(5) * 2 * 8224 * 1024 * 2;
// What the engine allocates besides weights and sessions (activations for a 2,048-row chunk and their second set for
// the side prefill, logits, the checkpoint store, graphs), measured on node-b on 2026-10-27: 2.5-3.4 GB at one to
// four sessions; 4 GB with margin.
inline constexpr size_t kScratchBytes = size_t(4) << 30;

// One session's flat slot at `max_context` tokens: the full-attention layers' BF16 KV, the DeltaNet and conv state
// and the end-of-prompt copy of both (snap), the drafter's ring when there is a drafter, and the token history.
inline size_t session_bytes(const ModelConfig& c, int max_context, bool drafter) {
  int full = 0;
  for (bool f : c.full_attention) full += f;
  const int lin = c.layers - full;
  const size_t kv = size_t(full) * 2 * c.kv_heads * c.head_dim * 2 * size_t(max_context);
  const size_t state = size_t(lin) * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim * 4;
  const size_t conv = size_t(lin) * c.lin_conv_channels() * (c.conv_kernel - 1) * 4;
  return kv + 2 * (state + conv) + (drafter ? kDraftRingBytes : 0) + size_t(max_context) * 4;
}

struct MemoryPlan {
  size_t weights = 0, per_session = 0, sessions = 0, scratch = kScratchBytes;
  int session_count = 0;
  size_t total() const { return weights + sessions + scratch; }
};

// The bytes of the *.safetensors files in a checkpoint directory (what the engine uploads, within a few percent).
inline size_t checkpoint_bytes(const std::string& dir) {
  size_t n = 0;
  std::error_code ec;
  if (dir.empty() || !std::filesystem::is_directory(dir, ec)) return 0;
  for (const auto& e : std::filesystem::directory_iterator(dir, ec))
    if (e.path().extension() == ".safetensors") n += std::filesystem::file_size(e.path(), ec);
  return n;
}

inline MemoryPlan plan_memory(const ModelConfig& c, size_t weight_bytes, int sessions, int max_context, bool drafter) {
  MemoryPlan p;
  p.weights = weight_bytes;
  p.per_session = session_bytes(c, max_context, drafter);
  p.session_count = sessions;
  p.sessions = p.per_session * size_t(sessions);
  return p;
}

// MemAvailable in MB from a /proc/meminfo-shaped file; -1 when it cannot be read.
inline long mem_available_mb(const std::string& path = "/proc/meminfo") {
  std::ifstream f(path);
  std::string key;
  long kb = 0;
  std::string unit;
  while (f >> key >> kb) {
    std::getline(f, unit);
    if (key == "MemAvailable:") return kb / 1024;
  }
  return -1;
}

inline std::string gb(size_t bytes) {
  char b[32];
  std::snprintf(b, sizeof b, "%.1f GB", bytes / 1e9);
  return b;
}

// Why the plan cannot start on a host with `avail_mb` available and `reserve_mb` to keep; empty when it can.
inline std::string refuse_start(const MemoryPlan& p, long avail_mb, long reserve_mb) {
  if (avail_mb < 0) return "";  // no /proc/meminfo: nothing to judge by
  const long need_mb = static_cast<long>(p.total() >> 20);
  if (need_mb + reserve_mb <= avail_mb) return "";
  std::ostringstream o;
  o << "ling-serve would need " << gb(p.total()) << " (weights " << gb(p.weights) << ", " << p.session_count
    << " sessions x " << gb(p.per_session) << ", scratch " << gb(p.scratch) << ") and keep " << reserve_mb
    << " MB free, but only " << avail_mb << " MB of host memory is available: short by "
    << (need_mb + reserve_mb - avail_mb) << " MB. Use fewer --sessions, a smaller --max-context, or stop what else "
    << "holds the memory (another model server?).";
  return o.str();
}

// The 503 a generation request gets while the host is below the admission reserve; empty when it may run.
inline std::string refuse_request(long avail_mb, long reserve_mb) {
  if (avail_mb < 0 || avail_mb >= reserve_mb) return "";
  std::ostringstream o;
  o << "host memory low: " << avail_mb << " MB available, below the " << reserve_mb
    << " MB this server keeps free; the request was not started (retry later)";
  return o.str();
}

}  // namespace ling::serve
