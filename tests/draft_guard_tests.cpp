// Host-side checks of the drafter-off guard's policy (core/draft_guard.hpp, BACKLOG 10), no GPU: the warm-up, the
// switch off below the threshold, the probes while off, the hysteresis, the switch back on, the check's flapping
// configuration (a margin of -off_below) and the guard off at 0.
#include <cmath>
#include <cstdio>

#include "core/draft_guard.hpp"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAILED: %s\n", what);
    ++failures;
  }
}

// Plain steps until the guard wants a speculative one (at most `cap`); returns how many.
int plain_until_probe(ling::DraftGuard& g, int cap = 1000) {
  int n = 0;
  while (!g.speculate() && n < cap) {
    g.plain();
    ++n;
  }
  return n;
}

void check_off_at_zero() {
  ling::DraftGuardConfig c;
  c.off_below = 0;
  ling::DraftGuard g;
  bool switched = false;
  for (int i = 0; i < 200; ++i) switched |= g.observe(c, 1.0);
  expect(!switched && g.on() && g.speculate(), "off_below 0 never switches");
}

void check_warmup_and_trip() {
  const ling::DraftGuardConfig c;  // 1.5, back on at 1.75, alpha 1/16, warm-up 16, a probe every 16
  ling::DraftGuard g;
  for (int i = 0; i < c.warmup - 1; ++i) expect(!g.observe(c, 1.0), "no switch during the warm-up");
  expect(g.on(), "on through the warm-up");
  expect(g.observe(c, 1.0), "the warm-up's last step switches off");
  expect(!g.on() && !g.speculate(), "off after the trip, a plain step next");
  expect(plain_until_probe(g) == c.probe_every, "probe_every plain steps before the probe");
  expect(g.speculate() && !g.on(), "the probe is a speculative step, the guard still off");
  expect(!g.observe(c, 1.0), "a losing probe does not switch");
  expect(!g.speculate(), "plain again after a losing probe");
  expect(plain_until_probe(g) == c.probe_every, "and the next probe after another probe_every plain steps");
}

void check_recovery() {
  const ling::DraftGuardConfig c;
  ling::DraftGuard g;
  for (int i = 0; i < c.warmup; ++i) g.observe(c, 1.0);
  expect(!g.on(), "tripped at 1 token per step");
  // The drafter back at 6 tokens per step: the mean is 6 - 5 (15/16)^k after k probes, at least 1.75 from k = 3.
  int probes = 0;
  while (!g.on() && probes < 100) {
    plain_until_probe(g);
    ++probes;
    g.observe(c, 6.0);
  }
  expect(probes == 3, "back on after 3 probes at 6 tokens per step");
  expect(g.speculate(), "speculative after switching back on");
  for (int i = 0; i < 100; ++i) expect(!g.observe(c, 6.0), "stays on at 6 tokens per step");
}

void check_hysteresis() {
  const ling::DraftGuardConfig c;
  ling::DraftGuard on;
  bool switched = false;
  for (int i = 0; i < 500; ++i) switched |= on.observe(c, 1.6);
  expect(!switched && on.on(), "1.6 tokens per step (above 1.5) never trips");
  ling::DraftGuard off;
  for (int i = 0; i < c.warmup; ++i) off.observe(c, 1.0);
  switched = false;
  for (int i = 0; i < 500; ++i) {
    plain_until_probe(off);
    switched |= off.observe(c, 1.6);
  }
  expect(!switched && !off.on(), "probes at 1.6 (below 1.75) never switch back on");
  std::printf("  mean after 500 probes at 1.6: %.3f\n", off.mean());
  expect(std::fabs(off.mean() - 1.6) < 1e-6, "the mean converges to the probes' value");
}

void check_flapping() {
  // ling-run's check: a margin of -off_below puts on_at at 0, so every probe switches back on and the next step off.
  ling::DraftGuardConfig c;
  c.off_below = 100;
  c.on_at = 0;
  ling::DraftGuard g;
  int switches = 0;
  for (int i = 0; i < c.warmup; ++i) switches += g.observe(c, 7.0);
  expect(switches == 1 && !g.on(), "trips at the warm-up's end");
  for (int cycle = 0; cycle < 5; ++cycle) {
    expect(plain_until_probe(g) == c.probe_every, "probe_every plain steps per cycle");
    expect(g.observe(c, 7.0) && g.on(), "the probe switches back on");
    expect(g.speculate(), "the step after it is speculative");
    expect(g.observe(c, 7.0) && !g.on(), "and switches off again");
  }
}

void check_reset() {
  const ling::DraftGuardConfig c;
  ling::DraftGuard g;
  for (int i = 0; i < c.warmup; ++i) g.observe(c, 1.0);
  g.reset();
  expect(g.on() && g.speculate() && g.seen() == 0 && g.mean() < 0, "reset: on, nothing seen");
  for (int i = 0; i < c.warmup - 1; ++i) expect(!g.observe(c, 1.0), "the warm-up again after a reset");
}

}  // namespace

int main() {
  check_off_at_zero();
  check_warmup_and_trip();
  check_recovery();
  check_hysteresis();
  check_flapping();
  check_reset();
  std::printf(failures ? "DRAFT GUARD TESTS FAILED (%d)\n" : "all draft guard tests passed\n", failures);
  return failures ? 1 : 0;
}
