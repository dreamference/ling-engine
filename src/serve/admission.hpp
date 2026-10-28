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
//  - Sizing at start (auto-size, the owner's decision of 2026-10-28): what the operator does not give of --sessions
//    and --max-context is chosen from the same plan, before the check above: the model's full context and as many
//    sessions as fit (at most kAutoSessionsCap), the context halved only when one session does not fit.
//
// None of it touches a number the engine computes: a request that is admitted runs exactly as before, and a server
// sized automatically computes what the same --sessions, --batch and --max-context given by hand compute.
//
// P4b (--kv-pages, BATCHING §5.3): a session's full-attention KV lives in the engine's page pool, so the plan counts each
// session's fixed part only (state, snap, drafter ring, history) and the pool once. The pool's size is what MemAvailable
// leaves after the rest of the plan, the prefix KV store's budget and --host-reserve-mb, at most what every session at
// --max-context could map; or --kv-pool-mb, refused at start when the plan with it does not fit.
// Host-only arithmetic, tested in ling-api-tests and ling-kv-pool-tests.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/model.hpp"

namespace ling::serve {

// The drafter's context ring per session (M4 stage 1): 5 layers x K and V x 8,224 positions x 8 heads x 128 x BF16.
inline constexpr size_t kDraftRingBytes = size_t(5) * 2 * 8224 * 1024 * 2;
// What the engine allocates besides weights and sessions (activations for a 2,048-row chunk and their second set for
// the side prefill, logits, the checkpoint store, graphs), measured on node-b on 2026-10-27: 2.5-3.4 GB at one to
// four sessions; 4 GB with margin.
inline constexpr size_t kScratchBytes = size_t(4) << 30;

// The full-attention layers' BF16 K and V of one position.
inline size_t kv_bytes_per_token(const ModelConfig& c) {
  int full = 0;
  for (bool f : c.full_attention) full += f;
  return size_t(full) * 2 * c.kv_heads * c.head_dim * 2;
}
// P4: positions per KV page (kernels::kKvPageTokens) and one page's bytes (4 MiB for Qwen3.8-27B).
inline constexpr int kKvPageTokens = 64;
inline size_t kv_page_bytes(const ModelConfig& c) { return kv_bytes_per_token(c) * kKvPageTokens; }
inline long kv_pages_for(long positions) { return (positions + kKvPageTokens - 1) / kKvPageTokens; }

// One session's flat slot at `max_context` tokens: the full-attention layers' BF16 KV, the DeltaNet and conv state
// and the end-of-prompt copy of both (snap), the drafter's ring when there is a drafter, and the token history.
// kv_pages: the fixed part only (the KV is the pool's, counted once in the plan).
inline size_t session_bytes(const ModelConfig& c, int max_context, bool drafter, bool kv_pages = false) {
  int full = 0;
  for (bool f : c.full_attention) full += f;
  const int lin = c.layers - full;
  const size_t kv = kv_pages ? 0 : kv_bytes_per_token(c) * size_t(max_context);
  const size_t state = size_t(lin) * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim * 4;
  const size_t conv = size_t(lin) * c.lin_conv_channels() * (c.conv_kernel - 1) * 4;
  return kv + 2 * (state + conv) + (drafter ? kDraftRingBytes : 0) + size_t(max_context) * 4;
}

struct MemoryPlan {
  size_t weights = 0, per_session = 0, sessions = 0, scratch = kScratchBytes;
  int session_count = 0;
  // P4b: the KV page pool (its pages and the reserved page 0), and the pages it hands out; 0 without --kv-pages.
  size_t kv_pool = 0;
  long kv_pool_pages = 0;
  size_t total() const { return weights + sessions + scratch + kv_pool; }
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

// `scratch` is kScratchBytes plus what else the start allocates beside weights and sessions (the vision tower's).
inline MemoryPlan plan_memory(const ModelConfig& c, size_t weight_bytes, int sessions, int max_context, bool drafter,
                              bool kv_pages = false, size_t scratch = kScratchBytes) {
  MemoryPlan p;
  p.weights = weight_bytes;
  p.scratch = scratch;
  p.per_session = session_bytes(c, max_context, drafter, kv_pages);
  p.session_count = sessions;
  p.sessions = p.per_session * size_t(sessions);
  return p;
}

// P4b: sizes the plan's KV page pool (p from plan_memory with kv_pages). pool_mb >= 0 (--kv-pool-mb): that many MB of
// pages; else what MemAvailable (avail_mb) leaves after the plan, `other_bytes` (the prefix KV store's budget, which
// sizes itself after the engine) and reserve_mb; with nothing to judge by (avail_mb < 0, or no reserve) every session's
// max_context. Never more pages than sessions x max_context positions need, never fewer than 0. Returns the pages.
inline long plan_kv_pool(MemoryPlan& p, const ModelConfig& c, int max_context, long avail_mb, long reserve_mb,
                         size_t other_bytes, long pool_mb) {
  const size_t page = kv_page_bytes(c);
  const long cap = long(p.session_count) * kv_pages_for(max_context);
  long pages = cap;
  if (pool_mb >= 0) {
    pages = static_cast<long>((size_t(pool_mb) << 20) / page);
  } else if (avail_mb >= 0 && reserve_mb > 0) {
    const size_t have = size_t(avail_mb) << 20, keep = p.total() + other_bytes + (size_t(reserve_mb) << 20) + page;
    pages = have > keep ? static_cast<long>((have - keep) / page) : 0;
  }
  pages = std::max(0L, std::min(pages, cap));
  p.kv_pool_pages = pages;
  p.kv_pool = size_t(pages + 1) * page;  // with the reserved page 0
  return pages;
}

// P4b: why a pool sized from the plan is too small to start with (it cannot hold one conversation at max_context and
// the two pages of growth a request is admitted with); empty when it can. An explicit --kv-pool-mb is the operator's
// choice and is not judged here (a request that needs more pages than the pool has is refused when it arrives).
// A pool that holds every session's max_context (sessions x max_context, the most plan_kv_pool gives) is never too
// small: one session needs no growth pages beyond its own context.
inline long kv_pages_one_conversation(const MemoryPlan& p, int max_context) {
  return std::min(kv_pages_for(max_context) + 2, long(p.session_count) * kv_pages_for(max_context));
}
inline std::string refuse_kv_pool(const MemoryPlan& p, int max_context) {
  const long need = kv_pages_one_conversation(p, max_context);
  if (p.kv_pool_pages >= need) return "";
  std::ostringstream o;
  o << "the KV page pool would hold " << p.kv_pool_pages << " pages (" << p.kv_pool_pages * kKvPageTokens
    << " positions), fewer than one conversation at --max-context " << max_context << " needs (" << need
    << " pages). Use a smaller --max-context, fewer --sessions, or stop what else holds the memory.";
  return o.str();
}

// The start line's description of the pool: GB, pages, positions.
inline std::string describe_kv_pool(const MemoryPlan& p) {
  char b[160];
  std::snprintf(b, sizeof b, "%.1f GB, %ld pages, %ld positions", p.kv_pool / 1e9, p.kv_pool_pages,
                p.kv_pool_pages * long(kKvPageTokens));
  return b;
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
    << " sessions x " << gb(p.per_session) << ", scratch " << gb(p.scratch);
  if (p.kv_pool) o << ", KV page pool " << gb(p.kv_pool);
  o << ") and keep " << reserve_mb
    << " MB free, but only " << avail_mb << " MB of host memory is available: short by "
    << (need_mb + reserve_mb - avail_mb) << " MB. Use fewer --sessions, a smaller --max-context, or stop what else "
    << "holds the memory (another model server?).";
  return o.str();
}

// P4d: what the plan needs to know about the KV page pool. `store_bytes` is the prefix KV store's budget the pool
// leaves (0 when the store sizes itself after the engine), `pool_mb` --kv-pool-mb (< 0: sized from the plan).
struct KvPlanInputs {
  bool on = false;
  size_t store_bytes = 0;
  long pool_mb = -1;
};

// The start's plan and why it is refused (empty when it starts), the one computation the start check, auto-size and
// --plan-only share: plan_memory, with --kv-pages the pool (plan_kv_pool) and its floor (refuse_kv_pool, only for a
// pool sized from a known MemAvailable), then refuse_start. With `reserve_mb` <= 0 nothing is judged.
struct StartPlan {
  MemoryPlan plan;
  std::string refusal;
};
inline StartPlan start_plan(const ModelConfig& c, size_t weight_bytes, size_t scratch, bool drafter, int sessions,
                            int max_context, long avail_mb, long reserve_mb, const KvPlanInputs& kv = {}) {
  StartPlan s{plan_memory(c, weight_bytes, sessions, max_context, drafter, kv.on, scratch), {}};
  if (kv.on) {
    plan_kv_pool(s.plan, c, max_context, reserve_mb > 0 ? avail_mb : -1, reserve_mb, kv.store_bytes, kv.pool_mb);
    if (kv.pool_mb < 0 && reserve_mb > 0 && avail_mb >= 0) s.refusal = refuse_kv_pool(s.plan, max_context);
  }
  if (s.refusal.empty() && reserve_mb > 0) s.refusal = refuse_start(s.plan, avail_mb, reserve_mb);
  return s;
}

// --plan-only's line (read by Mightling's pre-load check): what runs and the plan's bytes, the page pool included.
// sessions, batch, max_context and planned_bytes are the contract; the pool's fields and the reserve are additive.
inline std::string plan_json(int sessions, int batch, int max_context, const MemoryPlan& p, bool kv_pages,
                             long reserve_mb) {
  std::ostringstream o;
  o << "{\"sessions\":" << sessions << ",\"batch\":" << std::min(batch, sessions) << ",\"max_context\":" << max_context
    << ",\"planned_bytes\":" << p.total() << ",\"kv_pages\":" << (kv_pages ? "true" : "false")
    << ",\"kv_pool_bytes\":" << p.kv_pool << ",\"kv_pool_pages\":" << p.kv_pool_pages
    << ",\"host_reserve_bytes\":" << (size_t(std::max(0L, reserve_mb)) << 20) << "}";
  return o.str();
}

// P4d, auto-size with --kv-pages (the coordinator's decision of 2026-10-28): n sessions at max_context fit when the pool
// holds one conversation at max_context (as the start check asks) and n x max_context / kPagedFitShare positions, so
// that n conversations at half the context can be resident at once. More than that is not asked: when n sessions
// would need more pages than the pool has, the scheduler requeues (P4c), it does not refuse the start.
inline constexpr long kPagedFitShare = 2;
inline bool paged_sessions_fit(const MemoryPlan& p, int sessions, int max_context) {
  const long half = long(sessions) * kv_pages_for(max_context) / kPagedFitShare;
  return p.kv_pool_pages >= std::max(kv_pages_one_conversation(p, max_context), half);
}

// Auto-size (BATCHING §5.6): the most sessions chosen without --sessions (the owner's cap on node-b until a larger
// number is approved; also the engine's kMaxSeqs), and the shortest context chosen without --max-context.
inline constexpr int kAutoSessionsCap = 4;
inline constexpr int kAutoContextFloor = 32768;
// What ling-serve runs when it does not size itself (--auto-size 0, or MemAvailable unreadable): today's defaults.
inline constexpr int kDefaultSessions = 2;
inline constexpr int kDefaultContext = 65536;
inline constexpr int kDefaultBatch = 2;  // --batch without sizing; with it, --batch follows the chosen sessions

// The model's full context: `max_position_embeddings` of config.json (under `text_config` when there is one, as
// ModelConfig::from_file reads it); 0 when the file or the key is missing.
inline int model_full_context(const std::string& config_json) {
  std::ifstream f(config_json);
  if (!f) return 0;
  const nlohmann::json root = nlohmann::json::parse(f, nullptr, false);
  if (root.is_discarded() || !root.is_object()) return 0;
  const nlohmann::json& t = root.contains("text_config") ? root["text_config"] : root;
  const auto it = t.find("max_position_embeddings");
  return it != t.end() && it->is_number_integer() ? it->get<int>() : 0;
}

struct SizeChoice {
  int sessions = 0, max_context = 0;
  bool sessions_auto = false, context_auto = false;  // chosen here, not given
  bool sized = false;  // false: auto-size off or MemAvailable unknown, so the defaults fill what was not given
};

// The sessions and context ling-serve starts with: each given value as given, each other one chosen by the start
// check's own plan (start_plan, nothing estimated twice) against `avail_mb` less `reserve_mb`.
// The context first: the full context, halved (to a multiple of 2,048) down to kAutoContextFloor, the first at which
// the given sessions, or one session, fit; then as many sessions as fit at it, up to kAutoSessionsCap. When nothing
// fits, the given values or one session at the floor, which the start check then refuses (exit 3). With `auto_size`
// off, MemAvailable unknown (`avail_mb` < 0) or no reserve to keep (`reserve_mb` <= 0), the defaults fill in.
// With `kv` on (--kv-pages) "fits" is the paged plan's: start_plan does not refuse it and paged_sessions_fit holds.
inline SizeChoice choose_size(const ModelConfig& c, size_t weight_bytes, size_t scratch, bool drafter, long avail_mb,
                              long reserve_mb, int full_context, std::optional<int> sessions,
                              std::optional<int> max_context, bool auto_size = true, const KvPlanInputs& kv = {}) {
  SizeChoice s;
  s.sessions_auto = !sessions;
  s.context_auto = !max_context;
  if (!auto_size || avail_mb < 0 || reserve_mb <= 0 || (sessions && max_context)) {
    s.sessions = sessions.value_or(kDefaultSessions);
    s.max_context = max_context.value_or(kDefaultContext);
    s.sessions_auto = s.context_auto = false;
    s.sized = auto_size && avail_mb >= 0 && reserve_mb > 0;
    return s;
  }
  s.sized = true;
  auto fits = [&](int n, int ctx) {
    const StartPlan p = start_plan(c, weight_bytes, scratch, drafter, n, ctx, avail_mb, reserve_mb, kv);
    return p.refusal.empty() && (!kv.on || paged_sessions_fit(p.plan, n, ctx));
  };
  if (full_context <= 0) full_context = kDefaultContext;
  std::vector<int> contexts;
  if (max_context) {
    contexts.push_back(*max_context);
  } else {
    for (int ctx = full_context; ctx >= kAutoContextFloor; ctx = (ctx / 2) - (ctx / 2) % 2048) contexts.push_back(ctx);
    if (contexts.empty()) contexts.push_back(full_context);  // a model shorter than the floor: its own context
  }
  for (int ctx : contexts) {
    if (!fits(sessions.value_or(1), ctx)) continue;
    s.max_context = ctx;
    s.sessions = sessions.value_or(1);
    if (!sessions)
      for (int n = kAutoSessionsCap; n > 1; --n)
        if (fits(n, ctx)) {
          s.sessions = n;
          break;
        }
    return s;
  }
  s.sessions = sessions.value_or(1);
  s.max_context = contexts.back();
  return s;
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
