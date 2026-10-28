// The drafter-off guard (BACKLOG §19 item 10, "an automatic switch-off if it ever loses"; SPECULATION §10, "The
// drafter-off guard"): per session, a running mean of the tokens each speculative step produced, and the decision
// whether the session's next step is speculative or one plain row. Host-only policy (no CUDA), so
// tests/draft_guard_tests.cpp runs it without a GPU; the engine (speculate.cpp) applies it.
//
// The measure is tokens per step, the step's anchor plus its accepted drafts (accepted + 1): what a step yields per
// target pass, against the plain step's 1. A speculative step costs 1.11-1.12 plain steps today (95.3 / 84.9 ms at
// 4K, 100.8 / 91.0 at 32K, measured 2026-10-28), at most ~1.24 against a graph-captured one-row pass, so the drafter
// loses only below ~1.12-1.24 tokens per step; agent work runs at 4-7.
//
// The numbers (DraftGuardConfig's defaults):
//   - off_below 1.5 (the owner's default; ling-serve --draft-off-below): a margin of ~0.25-0.4 over the measured
//     break-even, for the contexts where the verify's 16 rows weigh more in attention than at 4K-32K (BACKLOG 10's
//     sweep measures them). Being off when the drafter would win costs far more (at 2 tokens per step, ~45% of the
//     throughput) than being on when it loses (at most ~11%, the step's own overhead), so the default stays low.
//   - on_at 1.75 (off_below + 0.25): hysteresis, so a session near the threshold does not flap on noise. A step's
//     tokens run from 1 to the block, so one value moves the mean by up to (block - 1) / 16.
//   - alpha 1/16, an EMA over roughly the last 16 speculative steps (the sparse head's fallback uses the same rate,
//     speculate.cpp), and warmup 16: no trip before 16 steps have been seen, so a session's first few steps never
//     switch it off.
//   - probe_every 16: while off, one speculative step after every 16 plain ones, whose tokens feed the same mean.
//     A probe costs ~0.12 of a plain step beyond the one row it would have produced at worst, ~0.7% while off; a
//     session whose drafter is back at 4-7 tokens per step is on again after 1-3 probes (16-48 plain steps).
// Switching costs nothing: a plain step feeds the drafter's context ring as a speculative step's commit does
// (Engine::step), so a probe or the first step after switching back on drafts from the context it would have had.
#pragma once

namespace ling {

struct DraftGuardConfig {
  double off_below = 1.5;  // <= 0: the guard is off (every step speculative)
  double on_at = 1.75;
  double alpha = 1.0 / 16;
  int warmup = 16;
  int probe_every = 16;
};

class DraftGuard {
 public:
  // Whether the next step should be speculative: on, or off with its plain steps before the next probe taken.
  bool speculate() const { return on_ || plain_left_ == 0; }
  bool on() const { return on_; }
  double mean() const { return ema_; }
  long seen() const { return seen_; }

  // After a speculative step that produced `tokens` (accepted + 1). Returns true when the guard switched (off, or
  // back on).
  bool observe(const DraftGuardConfig& c, double tokens) {
    ema_ = ema_ < 0 ? tokens : ema_ * (1 - c.alpha) + tokens * c.alpha;
    ++seen_;
    if (c.off_below <= 0) {  // the guard off: it keeps the mean but never switches
      on_ = true;
      return false;
    }
    if (on_) {
      if (seen_ < c.warmup || ema_ >= c.off_below) return false;
      on_ = false;
      plain_left_ = c.probe_every;
      return true;
    }
    if (ema_ >= c.on_at) {  // a probe brought the mean back up
      on_ = true;
      plain_left_ = 0;
      return true;
    }
    plain_left_ = c.probe_every;  // still off: the next probe after another probe_every plain steps
    return false;
  }

  // After a plain step taken because speculate() was false.
  void plain() {
    if (plain_left_ > 0) --plain_left_;
  }

  void reset() { *this = DraftGuard{}; }

 private:
  bool on_ = true;
  double ema_ = -1;  // no step seen yet
  long seen_ = 0;
  int plain_left_ = 0;  // while off: plain steps before the next probe
};

}  // namespace ling
