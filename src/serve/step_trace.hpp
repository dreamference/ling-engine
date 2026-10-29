// The per-step trace (C3): LING_STEP_TRACE=FILE writes one JSON line per scheduler unit, so the time of each
// conversation can be followed pass by pass (bench/agentlong/step_trace.py sums it up). Off when the variable is
// unset; the file is buffered and flushed when no request is running and at exit.
//
// A line: {"t":S,"pass":N,"kind":K,"turn":I,"sessions":[..],"rows":[..],"tokens":[..],"left":[..],"cand":[[..]],"ms":X,
//          "side":B}
//   t        seconds since the trace was opened, at the end of the unit
//   pass     decode passes run so far (a batched step counts once)
//   kind     decode | side_launch | side_finish | chunk | hold | yield
//   turn     the running index whose turn it was
//   sessions the sessions in the unit (several for a batched step)
//   rows     the verify rows of each (decode), 0 for a prefill unit
//   tokens   the tokens each chose in the step (decode), 0 for a prefill unit
//   left     the prompt tokens each has still to prefill (0 once decoding)
//   ms       the unit's wall time
//   side     whether a prefill chunk is in flight on the side stream after the unit
//   cand     decode lines only: every running request as the partner choice saw it, in running order (so cand[turn]
//            is the request whose turn it was), each [session, request, state, recent, last]: request is its
//            admission number (a session holds many requests in turn), state 0 not batchable, 1 batchable, 2
//            batchable but tried and skipped because its KV pages did not fit (P4b), recent its decode passes in
//            the trailing window and last its last decode pass (serve/pairing.hpp), both before the unit.
//            bench/agentlong/fair_gate.py checks fair pairing pass by pass from it.
// Host only: tests/pairing_tests.cpp checks one formatted line, and bench/agentlong/fair_gate_test.py parses the
// same line.
#pragma once

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace ling::serve {

// One running request as the partner choice saw it (the trace's `cand`).
struct StepCandidate {
  int session = 0;
  long request = 0;  // Running::admitted
  int state = 0;     // 0 not batchable, 1 batchable, 2 batchable but skipped for its KV pages
  int recent = 0;    // decode passes in the trailing window, before the unit
  long last = 0;     // its last decode pass (0: none yet), before the unit
};

struct StepRecord {
  double t = 0;
  long pass = 0;
  const char* kind = "decode";
  size_t turn = 0;
  std::vector<int> sessions, rows, tokens, left;
  std::vector<StepCandidate> cand;  // decode lines only (empty: no `cand` key)
  double ms = 0;
  bool side = false;
};

inline std::string format_step(const StepRecord& s) {
  auto list = [](const std::vector<int>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
    return o + "]";
  };
  char head[160], tail[64];
  std::snprintf(head, sizeof head, "{\"t\":%.6f,\"pass\":%ld,\"kind\":\"%s\",\"turn\":%zu,", s.t, s.pass, s.kind,
                s.turn);
  std::snprintf(tail, sizeof tail, ",\"ms\":%.3f,\"side\":%s}", s.ms, s.side ? "true" : "false");
  std::string cand;
  if (!s.cand.empty()) {
    cand = ",\"cand\":[";
    for (size_t i = 0; i < s.cand.size(); ++i) {
      const StepCandidate& c = s.cand[i];
      cand += (i ? ",[" : "[") + std::to_string(c.session) + "," + std::to_string(c.request) + "," +
              std::to_string(c.state) + "," + std::to_string(c.recent) + "," + std::to_string(c.last) + "]";
    }
    cand += "]";
  }
  return std::string(head) + "\"sessions\":" + list(s.sessions) + ",\"rows\":" + list(s.rows) +
         ",\"tokens\":" + list(s.tokens) + ",\"left\":" + list(s.left) + cand + tail;
}

class StepTrace {
 public:
  explicit StepTrace(const char* path) : t0_(std::chrono::steady_clock::now()) {
    if (path && *path) {
      f_ = std::fopen(path, "w");
      if (!f_) std::fprintf(stderr, "ling-serve: cannot open the step trace %s\n", path);
    }
  }
  ~StepTrace() {
    if (f_) std::fclose(f_);
  }
  StepTrace(const StepTrace&) = delete;
  StepTrace& operator=(const StepTrace&) = delete;
  bool on() const { return f_ != nullptr; }
  double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
  void write(const StepRecord& s) {
    if (!f_) return;
    const std::string line = format_step(s);
    std::fwrite(line.data(), 1, line.size(), f_);
    std::fputc('\n', f_);
  }
  void flush() {
    if (f_) std::fflush(f_);
  }

 private:
  std::FILE* f_ = nullptr;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace ling::serve
