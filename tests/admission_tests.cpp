// ling-admission-tests: the start's memory plan with the KV page pool (serve/admission.hpp, P4d), on the host. The one
// plan the start check, auto-size and --plan-only share (start_plan), the line --plan-only prints (plan_json), and
// auto-size in paged mode (choose_size with KvPlanInputs). BATCHING §5.3 and §5.6, with §3.5's node-b figures.
// Exits 0 when everything holds.
#include <cstdio>
#include <nlohmann/json.hpp>

#include "serve/admission.hpp"

using namespace ling::serve;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

ling::ModelConfig qwen() {  // Qwen3.8-27B's shapes (MODEL §5)
  ling::ModelConfig c;
  c.layers = 64;
  c.kv_heads = 4;
  c.head_dim = 256;
  c.lin_k_heads = 16;
  c.lin_v_heads = 48;
  c.lin_k_dim = 128;
  c.lin_v_dim = 128;
  c.conv_kernel = 4;
  for (int i = 0; i < 64; ++i) c.full_attention.push_back(i % 4 == 3);
  return c;
}

// node-b (§3.5): 114.7 GB available, weights 23.47 GB with the drafter, 4 GB of scratch, the 16,384 MiB reserve.
const long kNodeB = static_cast<long>(114.7e9 / (1 << 20)), kReserve = 16384;
const size_t kWeights = size_t(23.47e9);

// --plan-only reports the plan the start runs: with --kv-pages the pool is in planned_bytes and in its own fields.
void plan_only() {
  const ling::ModelConfig c = qwen();
  const KvPlanInputs paged{true, 0, -1};
  const StartPlan p = start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve, paged);
  check(p.refusal.empty(), "paged: four sessions at 262,144 start on node-b");
  check(p.plan.kv_pool > 0 && p.plan.kv_pool_pages > 0, "paged: the plan has a pool");
  const nlohmann::json j = nlohmann::json::parse(plan_json(4, 4, 262144, p.plan, true, kReserve));
  check(j["sessions"] == 4 && j["batch"] == 4 && j["max_context"] == 262144, "what runs");
  check(j["planned_bytes"].get<size_t>() == p.plan.total(), "planned_bytes is the start's plan");
  check(j["kv_pages"] == true && j["kv_pool_bytes"].get<size_t>() == p.plan.kv_pool &&
            j["kv_pool_pages"].get<long>() == p.plan.kv_pool_pages,
        "the pool's fields");
  check(j["planned_bytes"].get<size_t>() >= p.plan.kv_pool + kWeights, "planned_bytes counts the pool");
  check(j["host_reserve_bytes"].get<size_t>() == size_t(kReserve) << 20, "the reserve the pool leaves");
  // The pool takes what the reserve leaves: planned + reserve is MemAvailable within a page (and the reserved page 0).
  const size_t have = size_t(kNodeB) << 20, planned = p.plan.total() + (size_t(kReserve) << 20);
  check(planned <= have && have - planned < 2 * kv_page_bytes(c), "the pool fills what the reserve leaves");
  // Without --kv-pages: the flat plan, no pool, and the same contract fields.
  const StartPlan f = start_plan(c, kWeights, kScratchBytes, true, 2, 65536, kNodeB, kReserve);
  const nlohmann::json g = nlohmann::json::parse(plan_json(2, 2, 65536, f.plan, false, kReserve));
  check(f.refusal.empty() && g["kv_pages"] == false && g["kv_pool_bytes"] == 0 && g["kv_pool_pages"] == 0,
        "flat: no pool");
  check(g["planned_bytes"].get<size_t>() == plan_memory(c, kWeights, 2, 65536, true).total(), "flat: the flat plan");
  check(plan_json(4, 8, 65536, f.plan, false, 0).find("\"batch\":4,") != std::string::npos, "batch never > sessions");
  // The start's refusals, in order: flat four at 262,144 do not fit node-b; a too-small auto pool is refused; an
  // explicit --kv-pool-mb is judged only by refuse_start; with no reserve nothing is judged.
  check(!start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve).refusal.empty(), "flat 4: refused");
  check(start_plan(c, kWeights, kScratchBytes, true, 4, 262144, 50000, kReserve, paged).refusal.find("KV page pool") !=
            std::string::npos,
        "paged on a small host: the pool's refusal");
  check(start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve, {true, 0, 8192}).refusal.empty(),
        "--kv-pool-mb 8192 fits");
  check(!start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve, {true, 0, 80000}).refusal.empty(),
        "--kv-pool-mb 80000 is refused");
  check(start_plan(c, kWeights, kScratchBytes, true, 8, 262144, 1000, 0, paged).refusal.empty(), "no reserve: no check");
}

}  // namespace

int main() {
  plan_only();
  if (failures) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("admission tests passed\n");
  return 0;
}
