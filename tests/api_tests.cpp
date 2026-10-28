// Tests the output parsers and the chat template without a server or a GPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "serve/admission.hpp"
#include "serve/api.hpp"
#include "tokenizer/chat_template.hpp"

using ling::serve::json;
using ling::serve::OutputParser;

static int failures = 0;
#define EXPECT(cond)                                              \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

// A behaviour SGLang or vLLM got wrong and ling-serve still gets wrong (reports/engine-issues-2026-10-09.md):
// reported, not counted as a failure, so the suite stays green until the fix lands; the fix then turns
// the line into an EXPECT.
static int known_gaps = 0;
#define KNOWN(cond)                                                                        \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      std::printf("KNOWN GAP %s:%d: %s\n", __FILE__, __LINE__, #cond);                     \
      ++known_gaps;                                                                        \
    } else {                                                                               \
      std::printf("known gap now passes, make it an EXPECT: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                                      \
  } while (0)

static bool valid_utf8(const std::string& s) {
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k)
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    i += n;
  }
  return true;
}

static int count_of(const std::string& s, const std::string& sub) {
  int n = 0;
  for (size_t p = s.find(sub); p != std::string::npos; p = s.find(sub, p + sub.size())) ++n;
  return n;
}

// Feeds `text` one byte at a time, as a stream would, and collects everything.
static OutputParser::Delta feed(OutputParser& p, const std::string& text) {
  OutputParser::Delta all;
  auto add = [&](const OutputParser::Delta& d) {
    all.reasoning += d.reasoning;
    all.content += d.content;
    all.tool_calls.insert(all.tool_calls.end(), d.tool_calls.begin(), d.tool_calls.end());
  };
  for (char c : text) add(p.push(std::string(1, c)));
  add(p.finish());
  return all;
}

// vLLM's check_sequence_repetition (v1/core/sched/utils.py) on the whole output, the reference the incremental
// RepetitionDetector must match at every token.
static bool vllm_repetition(const std::vector<int>& t, int max_p, int min_p, int min_count) {
  if (min_p <= 0) min_p = 1;
  if (max_p <= 0 || min_count < 2 || min_p > max_p) return false;
  const long n = static_cast<long>(t.size());
  for (int P = min_p; P <= max_p; ++P) {
    if (static_cast<long>(P) * min_count > n) return false;
    bool rep = true;
    for (int k = 1; k <= P && rep; ++k)
      for (int m = 1; m < min_count && rep; ++m) rep = t[n - (static_cast<long>(P) * m + k)] == t[n - k];
    if (rep) return true;
  }
  return false;
}

// API §9: the repetition stop rule, its parsing and its resolution against the server's default.
static void test_repetition() {
  using ling::serve::RepetitionDetection;
  using ling::serve::RepetitionDetector;
  // The incremental test is vLLM's at every token: random outputs over small alphabets (so loops happen), planted
  // loops, every size range.
  unsigned seed = 12345;
  auto rnd = [&] { return seed = seed * 1103515245u + 12345u, (seed >> 16) & 0x7fff; };
  int compared = 0, hits = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const int alphabet = 2 + trial % 4, max_p = 1 + trial % 9, min_p = trial % 3 == 0 ? 0 : 1 + trial % std::max(1, max_p);
    const int min_count = 2 + trial % 4;
    RepetitionDetector d(RepetitionDetection{max_p, std::min(min_p, max_p), min_count, 0});
    std::vector<int> out;
    for (int i = 0; i < 120; ++i) {
      int tok = static_cast<int>(rnd() % alphabet);
      if (out.size() > 8 && rnd() % 3 == 0) tok = out[out.size() - 1 - rnd() % 4];  // echoes: periodic stretches
      out.push_back(tok);
      const bool got = d.push(tok), want = vllm_repetition(out, max_p, std::min(min_p, max_p), min_count);
      EXPECT(got == want);
      ++compared;
      hits += want;
    }
  }
  EXPECT(hits > 100);  // the comparison saw loops, not only their absence
  std::printf("repetition detector: %d prefixes against vLLM's test, %d loops\n", compared, hits);
  {  // a block of 3 repeated 4 times ends the output at its last token, not before
    RepetitionDetector d(RepetitionDetection{8, 2, 4, 0});
    const std::vector<int> t = {9, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2};
    for (int x : t) EXPECT(!d.push(x));
    EXPECT(d.push(3));
  }
  {  // with max_run on, a run of one token is max_run's alone, though it is also a block of 2 repeated (the block
     // test, which would stop it at 2 x 3 = 6, leaves it); a two-token block still stops at its third copy
    RepetitionDetector d(RepetitionDetection{8, 2, 3, 9});
    for (int i = 0; i < 8; ++i) EXPECT(!d.push(7));
    EXPECT(d.push(7));
    RepetitionDetector e(RepetitionDetection{8, 2, 3, 9});
    for (int x : {7, 7, 7, 1, 2, 1, 2, 1}) EXPECT(!e.push(x));
    EXPECT(e.push(2));
  }
  {  // max_run off: vLLM's test as it is, the run stopped as a block of 2
    RepetitionDetector d(RepetitionDetection{8, 2, 3, 0});
    for (int i = 0; i < 5; ++i) EXPECT(!d.push(7));
    EXPECT(d.push(7));
  }
  {  // off
    RepetitionDetector d(RepetitionDetection{0, 0, 0, 0});
    for (int i = 0; i < 1000; ++i) EXPECT(!d.push(i % 2));
  }
  {  // the default: a 40-token paragraph looping, stopped at its twelfth copy; ordinary repetition is not
    RepetitionDetector d(ling::serve::kRepetitionDefault);
    int stopped_at = -1;
    for (int i = 0; i < 40 * 20 && stopped_at < 0; ++i)
      if (d.push(1000 + i % 40)) stopped_at = i;
    EXPECT(stopped_at == 40 * 12 - 1);
    RepetitionDetector e(ling::serve::kRepetitionDefault), z(ling::serve::kRepetitionDefault);
    bool any = false;
    for (int i = 0; i < 4000; ++i) any |= e.push(i % 2 ? 11 : 100 + static_cast<int>(rnd() % 1000));  // "a, b, c, ..."
    for (int i = 0; i < 11 * 2; ++i) any |= z.push(i % 2 ? 11 : 0);  // "0, " eleven times
    RepetitionDetector h(ling::serve::kRepetitionDefault);  // " hello" alone: a run, stopped at 64, not at 24
    for (int i = 0; i < 63; ++i) any |= h.push(42);
    EXPECT(h.push(42));
    EXPECT(!any);
    const bool z0 = z.push(0), z1 = z.push(11);
    EXPECT(!z0 && z1);  // the twelfth "0, "
  }
  // Parsing: absent leaves every field to the server, false switches it off, an object sets its fields.
  const std::string msgs = R"("messages":[{"role":"user","content":"hi"}])";
  auto req = [&](const std::string& extra) { return ling::serve::parse_request(json::parse("{" + msgs + extra + "}"), true); };
  const RepetitionDetection server{256, 2, 12, 64};
  {
    const auto r = ling::serve::resolve_repetition(req(""), server);
    EXPECT(r.max_pattern_size == 256 && r.min_pattern_size == 2 && r.min_count == 12 && r.max_run == 64);
  }
  {
    const auto r = ling::serve::resolve_repetition(req(R"(,"repetition_detection":false)"), server);
    EXPECT(!r.on());
  }
  {
    const auto r = ling::serve::resolve_repetition(req(R"(,"repetition_detection":{"min_count":20,"max_run":0})"), server);
    EXPECT(r.max_pattern_size == 256 && r.min_pattern_size == 2 && r.min_count == 20 && r.max_run == 0);
  }
  {  // max_pattern_size alone below the server's min_pattern_size: the minimum follows it down
    const auto r = ling::serve::resolve_repetition(req(R"(,"repetition_detection":{"max_pattern_size":1})"), server);
    EXPECT(r.max_pattern_size == 1 && r.min_pattern_size == 1);
  }
  auto refused = [&](const std::string& extra) {
    try {
      ling::serve::resolve_repetition(req(extra), server);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  EXPECT(refused(R"(,"repetition_detection":{"min_count":1})"));
  EXPECT(refused(R"(,"repetition_detection":{"min_pattern_size":300})"));
  EXPECT(refused(R"(,"repetition_detection":{"max_run":1})"));
  EXPECT(refused(R"(,"repetition_detection":{"max_pattern_size":-1})"));
  EXPECT(refused(R"(,"repetition_detection":"yes")"));
  EXPECT(!refused(R"(,"repetition_detection":{"max_pattern_size":0,"min_pattern_size":0,"max_run":0})"));
  {  // /v1/responses forwards the field
    const auto r = ling::serve::parse_responses_request(
        json::parse(R"({"input":"hi","repetition_detection":{"min_count":30}})"));
    EXPECT(ling::serve::resolve_repetition(r, server).min_count == 30);
  }
}

// The Responses route's items and events (ResponseItems), the model's text pushed one byte at a time as tokens arrive:
// every text delta names an item that is open (added, not yet done), each item's deltas add up to its final text, every
// item added is done once, in output order, and whitespace after a call opens no item (SGLang's rule). Before
// 2026-10-28 the "\n" between two calls went out as a delta of the message the first call had closed (night 17,
// django-12741: "I'll make the changes now." and four calls to apply_patch, which the tools did not offer).
struct ItemsRun {
  json output;
  int text_deltas = 0, orphan_deltas = 0, added = 0, done = 0;
  bool consistent = true;  // per item: deltas == final text, done once, output_index == position
};

static ItemsRun run_items(const json& tools, bool reasoning, const std::string& text,
                          std::vector<std::string> custom = {}) {
  using ling::serve::ResponseItems;
  ItemsRun r;
  std::map<std::string, std::string> streamed;  // item id -> deltas so far
  std::map<std::string, int> index_of;          // open item id -> output_index
  std::set<std::string> seen;
  auto emit = [&](const std::string& type, json p) {
    if (type == "response.output_item.added") {
      ++r.added;
      const std::string id = p["item"]["id"];
      if (seen.count(id)) r.consistent = false;
      seen.insert(id);
      index_of[id] = p["output_index"];
    } else if (type == "response.output_text.delta" || type == "response.reasoning_text.delta") {
      if (type == "response.output_text.delta") ++r.text_deltas;
      const std::string id = p["item_id"];
      if (!index_of.count(id) || index_of[id] != p["output_index"].get<int>()) ++r.orphan_deltas;
      streamed[id] += p["delta"].get<std::string>();
    } else if (type == "response.output_item.done") {
      ++r.done;
      const json& item = p["item"];
      const std::string id = item["id"];
      if (!index_of.count(id) || index_of[id] != p["output_index"].get<int>()) r.consistent = false;
      if (item["type"] == "message" && item["content"][0]["text"] != streamed[id]) r.consistent = false;
      if (item["type"] == "reasoning" && item["content"][0]["text"] != streamed[id]) r.consistent = false;
      index_of.erase(id);
    }
  };
  OutputParser p(reasoning, tools);
  ResponseItems items(std::move(custom), emit);
  for (char c : text) items.deliver(p.push(std::string(1, c)));
  items.deliver(p.finish());
  items.close();
  r.output = items.output();
  if (!index_of.empty()) r.consistent = false;  // an item added and never done
  for (size_t i = 0; i < r.output.size(); ++i)
    if (!seen.count(r.output[i]["id"].get<std::string>())) r.consistent = false;
  return r;
}

static std::string types_of(const json& output) {
  std::string t;
  for (const json& item : output) t += (t.empty() ? "" : ",") + item["type"].get<std::string>();
  return t;
}

static void test_response_items() {
  const json exec_only = json::parse(R"([{"type":"function","function":{"name":"exec","parameters":{"type":"object",
    "properties":{"input":{"type":"string"}},"required":["input"]}}}])");
  const std::string patch_call =
      "<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** End Patch\n</parameter>\n"
      "</function>\n</tool_call>";
  const std::string exec_call =
      "<tool_call>\n<function=exec>\n<parameter=input>\ntext(1);\n</parameter>\n</function>\n</tool_call>";
  {
    // django-12741's shape: a message, then four calls separated by "\n".
    auto r = run_items(exec_only, false,
                       "I'll make the changes now.\n\n" + patch_call + "\n" + patch_call + "\n" + patch_call + "\n" + patch_call);
    EXPECT(r.orphan_deltas == 0);
    EXPECT(r.consistent);
    EXPECT(types_of(r.output) == "message,function_call,function_call,function_call,function_call");
    EXPECT(r.added == 5 && r.done == 5);
    EXPECT(r.text_deltas == static_cast<int>(std::string("I'll make the changes now.\n\n").size()));
    EXPECT(r.output[0]["content"][0]["text"] == "I'll make the changes now.\n\n");
  }
  {
    // Calls with no text before them: no whitespace-only message between or after them.
    auto r = run_items(exec_only, false, exec_call + "\n" + exec_call + "\n\n");
    EXPECT(r.orphan_deltas == 0 && r.consistent);
    EXPECT(types_of(r.output) == "function_call,function_call");
    EXPECT(r.text_deltas == 0);
  }
  {
    // Text after a call opens a new message item; the whitespace before it is the separator.
    auto r = run_items(exec_only, false, "First.\n\n" + exec_call + "\n\nThen this.");
    EXPECT(r.orphan_deltas == 0 && r.consistent);
    EXPECT(types_of(r.output) == "message,function_call,message");
    if (r.output.size() == 3) {
      EXPECT(r.output[0]["content"][0]["text"] == "First.\n\n");
      EXPECT(r.output[2]["content"][0]["text"] == "Then this.");
      EXPECT(r.output[0]["id"] != r.output[2]["id"]);
    }
    EXPECT(r.text_deltas == static_cast<int>(std::string("First.\n\nThen this.").size()));
  }
  {
    // Reasoning, then text, then a call, then whitespace: reasoning, message, call.
    auto r = run_items(exec_only, true, "Think.\n</think>\n\nNow.\n" + exec_call + "\n");
    EXPECT(r.orphan_deltas == 0 && r.consistent);
    EXPECT(types_of(r.output) == "reasoning,message,function_call");
  }
  {
    // A custom tool's call keeps its events; the "\n" after it opens nothing.
    auto r = run_items(exec_only, false, exec_call + "\n", {"exec"});
    EXPECT(r.orphan_deltas == 0 && r.consistent);
    EXPECT(types_of(r.output) == "custom_tool_call");
  }
  {
    // Plain text only: one message, unchanged.
    auto r = run_items(exec_only, false, "\n\nJust text.\n");
    EXPECT(r.orphan_deltas == 0 && r.consistent);
    EXPECT(types_of(r.output) == "message" && r.output[0]["content"][0]["text"] == "\n\nJust text.\n");
  }
}

int main() {
  test_repetition();
  test_response_items();
  const json tools = json::parse(R"([{"type":"function","function":{"name":"shell","parameters":{"type":"object",
    "properties":{"command":{"type":"array"},"timeout_ms":{"type":"integer"},"workdir":{"type":"string"}}}}}])");
  {
    OutputParser p(true, json::array());
    auto d = feed(p, "Let me think.\n</think>\n\nThe answer is 4.");
    EXPECT(d.reasoning == "Let me think.\n");
    EXPECT(d.content == "The answer is 4.");
  }
  {
    OutputParser p(true, tools);
    auto d = feed(p,
                  "I should list files.\n</think>\n\n<tool_call>\n<function=shell>\n<parameter=command>\n[\"ls\", \"-la\"]\n"
                  "</parameter>\n<parameter=timeout_ms>\n1000\n</parameter>\n<parameter=workdir>\n/tmp\n</parameter>\n"
                  "</function>\n</tool_call>");
    EXPECT(d.reasoning == "I should list files.\n");
    EXPECT(d.content.empty());
    EXPECT(d.tool_calls.size() == 1);
    if (!d.tool_calls.empty()) {
      json args = json::parse(d.tool_calls[0].arguments);
      EXPECT(d.tool_calls[0].name == "shell");
      EXPECT(args["command"] == json::parse(R"(["ls", "-la"])"));
      EXPECT(args["timeout_ms"] == 1000);
      EXPECT(args["workdir"] == "/tmp");
    }
  }
  {
    OutputParser p(false, tools);  // thinking off: everything is content
    auto d = feed(p, "No tools needed: <tool_ca is not a tag.");
    EXPECT(d.content == "No tools needed: <tool_ca is not a tag.");
    EXPECT(d.reasoning.empty());
  }
  {
    json messages = json::parse(R"([{"role":"system","content":"Be brief."},{"role":"user","content":"hi"},
      {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"shell",
       "arguments":"{\"command\": [\"ls\"]}"}}]},{"role":"tool","tool_call_id":"c1","content":"a.txt"}])");
    ling::ChatOptions o;
    std::string out = ling::render_chat(messages, tools, o);
    EXPECT(out.find("<|im_start|>system\n# Tools\n\n") == 0);  // medium effort: no reasoning instructions
    EXPECT(out.find("\n\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n") != std::string::npos);
    EXPECT(out.find("<tool_call>\n<function=shell>\n<parameter=command>\n[\"ls\"]\n</parameter>\n</function>\n</tool_call><|im_end|>\n") !=
           std::string::npos);
    EXPECT(out.find("<|im_start|>user\n<tool_response>\na.txt\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n") !=
           std::string::npos);
    EXPECT(out.find("{\"type\": \"function\", \"function\": {\"name\": \"shell\"") != std::string::npos);
  }
  {
    // The Responses conversion as production does it (bench/render/render_test.py checks it on replayed
    // requests): tools dumped in production's key order, system parts as chunks of their own, assistant
    // list contents merged without a separator.
    json body = json::parse(R"({"instructions":"Base.","tools":[{"type":"function","name":"shell","description":"Run",
      "parameters":{"type":"object"},"strict":false}],"reasoning":{"effort":"none"},
      "input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"A"},{"type":"input_text","text":"B"}]},
               {"type":"message","role":"user","content":[{"type":"input_text","text":"go"}]},
               {"type":"message","role":"assistant","content":[{"type":"output_text","text":"one"}]},
               {"type":"message","role":"assistant","content":[{"type":"output_text","text":"two"}]},
               {"type":"message","role":"user","content":"next"}]})");
    const ling::serve::Request r = ling::serve::parse_responses_request(body);
    const std::string& t = r.prompt_text;
    EXPECT(t.find("{\"type\": \"function\", \"function\": {\"description\": \"Run\", \"name\": \"shell\", "
                  "\"parameters\": {\"type\": \"object\"}, \"strict\": false}, \"defer_loading\": null}") != std::string::npos);
    EXPECT(t.find("</IMPORTANT>\n\nBase.\n\nA\n\nB<|im_end|>") != std::string::npos);
    EXPECT(t.find("<think>\n\n</think>\n\nonetwo<|im_end|>") != std::string::npos);
    const std::string tail = "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    EXPECT(t.size() >= tail.size() && t.compare(t.size() - tail.size(), tail.size(), tail) == 0);
  }
  {
    // M4 §1: prompt_cache_key (Codex's thread id) becomes the request's session key, the hint the session
    // pool looks at first; a request without one has an empty key.
    json body = json::parse(R"({"input":"hi","prompt_cache_key":"thread-1"})");
    EXPECT(ling::serve::parse_responses_request(body).session_key == "thread-1");
    EXPECT(ling::serve::parse_responses_request(json::parse(R"({"input":"hi"})")).session_key.empty());
    EXPECT(ling::serve::parse_request(json::parse(R"({"messages":[{"role":"user","content":"hi"}],
      "prompt_cache_key":"chat-1"})"), true).session_key == "chat-1");
  }
  // Bug classes from SGLang's and vLLM's trackers (reports/engine-issues-2026-10-09.md, specs/DREAMFERENCE_LING_ENGINE_VALIDATION.md §18 "Known
  // pitfalls"). Each block names the upstream issue it guards against.
  const json edit_tools = json::parse(R"([{"type":"function","function":{"name":"edit","parameters":{"type":"object",
    "properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}}}}}])");
  {
    // vllm#48753: string values keep their indentation and trailing newline (only the template's one
    // newline on each side is markup); a string that looks like JSON stays a string.
    OutputParser p(false, edit_tools);
    auto d = feed(p,
                  "<tool_call>\n<function=edit>\n<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=old_string>\n"
                  "    if x:\n        return 1\n\n</parameter>\n<parameter=new_string>\n{\"a\": 1}\n</parameter>\n"
                  "</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1);
    if (d.tool_calls.size() == 1) {
      json args = json::parse(d.tool_calls[0].arguments);
      EXPECT(args["path"] == "src/a.py");
      EXPECT(args["old_string"] == "    if x:\n        return 1\n");
      EXPECT(args["new_string"].is_string() && args["new_string"] == "{\"a\": 1}");
    }
  }
  {
    // vllm#57699: a last parameter the model did not close is kept (ended at </function>).
    OutputParser p(false, tools);
    auto d = feed(p,
                  "<tool_call>\n<function=shell>\n<parameter=command>\n[\"ls\"]\n</parameter>\n<parameter=workdir>\n/tmp\n"
                  "</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1);
    if (d.tool_calls.size() == 1) {
      json args = json::parse(d.tool_calls[0].arguments);
      EXPECT(args["command"] == json::parse(R"(["ls"])"));
      EXPECT(args["workdir"] == "/tmp");
    }
  }
  {
    // vllm#57699, the other half: an unclosed parameter followed by another must not swallow it.
    OutputParser p(false, tools);
    auto d = feed(p,
                  "<tool_call>\n<function=shell>\n<parameter=workdir>\n/tmp\n<parameter=timeout_ms>\n5\n</parameter>\n"
                  "</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1);
    if (d.tool_calls.size() == 1) {
      json args = json::parse(d.tool_calls[0].arguments);
      EXPECT(args["workdir"] == "/tmp" && args.contains("timeout_ms") && args["timeout_ms"] == 5);
    }
  }
  {
    // ... while parameter markup inside a closed value, not at a line start, stays part of the value.
    OutputParser p(false, edit_tools);
    auto d = feed(p,
                  "<tool_call>\n<function=edit>\n<parameter=path>\na.md\n</parameter>\n<parameter=new_string>\n"
                  "write <parameter=x> to name one\n</parameter>\n</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1);
    if (d.tool_calls.size() == 1) {
      json args = json::parse(d.tool_calls[0].arguments);
      EXPECT(args["path"] == "a.md" && args["new_string"] == "write <parameter=x> to name one");
    }
  }
  {
    // vllm#50989: a call without parameters parses to an empty object.
    OutputParser p(false, tools);
    auto d = feed(p, "<tool_call>\n<function=shell>\n</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1);
    if (d.tool_calls.size() == 1) EXPECT(d.tool_calls[0].name == "shell" && d.tool_calls[0].arguments == "{}");
  }
  {
    // vllm#56658: a literal <tool_call> in prose, with no call after it, loses no text.
    const std::string text = "Wrap calls in a `<tool_call>` tag; the rest of this sentence must survive.";
    OutputParser p(false, tools);
    auto d = feed(p, text);
    EXPECT(d.tool_calls.empty());
    EXPECT(d.content == text);
  }
  {
    // vllm#56658: a literal marker followed by a real call keeps the prose between them.
    OutputParser p(false, tools);
    auto d = feed(p,
                  "Wrap calls in `<tool_call>` tags.\n<tool_call>\n<function=shell>\n<parameter=command>\n[\"pwd\"]\n"
                  "</parameter>\n</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1);
    EXPECT(d.content == "Wrap calls in `<tool_call>` tags.\n");
  }
  {
    // vllm#56658, streaming: the text after a literal marker is released as it arrives, not held back
    // until the end of the answer.
    OutputParser p(false, tools);
    std::string seen;
    for (char c : std::string("Use `<tool_call>` here, then more text")) seen += p.push(std::string(1, c)).content;
    EXPECT(seen.find("`<tool_call>` here, then more") != std::string::npos);
  }
  {
    // vllm#57541: an example inside a markdown code fence is text, not a call.
    const std::string text =
        "Example:\n```xml\n<tool_call>\n<function=shell>\n<parameter=command>\n[\"ls\"]\n</parameter>\n</function>\n"
        "</tool_call>\n```\nThat is the format.";
    OutputParser p(false, tools);
    auto d = feed(p, text);
    EXPECT(d.tool_calls.empty() && d.content == text);
  }
  {
    // A tilde fence counts too, and a real call after the fence has closed is still a call.
    OutputParser p(false, tools);
    auto d = feed(p,
                  "~~~\n<tool_call>\n<function=shell>\n</function>\n</tool_call>\n~~~\nNow for real:\n<tool_call>\n"
                  "<function=shell>\n<parameter=command>\n[\"ls\"]\n</parameter>\n</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1);
    EXPECT(d.content.find("~~~\n<tool_call>\n<function=shell>\n</function>\n</tool_call>\n~~~\nNow for real:") == 0);
  }
  {
    // vllm#58147: a call to a function the request did not offer (here the template's own placeholder)
    // is not executed as a tool call.
    OutputParser p(false, tools);
    const std::string text =
        "<tool_call>\n<function=function_name>\n<parameter=parameter_name>\nvalue\n</parameter>\n</function>\n"
        "</tool_call>";
    auto d = feed(p, text);
    EXPECT(d.tool_calls.empty());
    EXPECT(d.content == text);  // handed back as text, nothing lost
  }
  {
    // vllm#58147: tool-call markup quoted inside the reasoning never becomes a call. (The opposite
    // complaint, vllm#39056, is a real call written before </think>; it stays in the reasoning too.)
    OutputParser p(true, tools);
    auto d = feed(p,
                  "The format is <tool_call>\n<function=shell>\n</function>\n</tool_call>, but I will answer directly.\n"
                  "</think>\n\nNo call needed.");
    EXPECT(d.tool_calls.empty());
    EXPECT(d.reasoning.find("<tool_call>") != std::string::npos);
    EXPECT(d.content == "No call needed.");
  }
  {
    // vllm#55495: a replayed function_call whose arguments are not valid JSON must not make every later
    // request of the conversation fail (the Responses path renders it with no parameters).
    json body = json::parse(R"({"tools":[{"type":"function","name":"shell","parameters":{"type":"object",
      "properties":{"command":{"type":"array"}}}}],
      "input":[{"type":"message","role":"user","content":"run it"},
               {"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"command\": [\"ec"},
               {"type":"function_call_output","call_id":"c1","output":"error"}]})");
    bool ok = true;
    std::string prompt;
    try {
      prompt = ling::serve::parse_responses_request(body).prompt_text;
    } catch (const std::exception&) {
      ok = false;
    }
    EXPECT(ok);
    EXPECT(prompt.find("<function=shell>\n</function>") != std::string::npos);
  }
  {
    // vllm#47761: the same on chat completions (Continue and Cline replay their history there).
    // The owner's decision (2026-10-10, API section 3): a string that is not a JSON object is rendered
    // verbatim where the parameter blocks would go, one newline added when it has none; an empty
    // string renders no parameters; an object renders as parameters. Production refuses such a history
    // (SGLang's parse_tool_call_arguments, 400 "Assistant tool call function.arguments must be valid
    // JSON."), so this is the one place ling-serve knowingly renders a prompt production would not.
    auto replay = [](const std::string& arguments) {
      json body = json::parse(R"({"messages":[{"role":"user","content":"run it"},
        {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"shell",
         "arguments":""}}]},
        {"role":"tool","tool_call_id":"c1","content":"error"}]})");
      body["messages"][1]["tool_calls"][0]["function"]["arguments"] = arguments;
      try {
        return ling::serve::parse_request(body, true).prompt_text;
      } catch (const std::exception& e) {
        return std::string("refused: ") + e.what();
      }
    };
    const std::string head = "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=shell>\n";
    const std::string tail =
        "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nerror\n</tool_response><|im_end|>\n";
    EXPECT(replay("{\"command\": [\"ec").find(head + "{\"command\": [\"ec\n" + tail) != std::string::npos);
    EXPECT(replay("{\"command\": [\"ec\n").find(head + "{\"command\": [\"ec\n" + tail) != std::string::npos);
    EXPECT(replay("").find(head + tail) != std::string::npos);
    EXPECT(replay("[\"ec\"]").find(head + "[\"ec\"]\n" + tail) != std::string::npos);
    EXPECT(replay("{\"command\": [\"ec\"]}").find(head + "<parameter=command>\n[\"ec\"]\n</parameter>\n" + tail) !=
           std::string::npos);
  }
  {
    // vllm#37167, sglang#42110: one assistant turn replayed as reasoning, a commentary message and two
    // function calls renders as ONE assistant block (plus the generation prompt).
    json body = json::parse(R"({"input":[{"type":"message","role":"user","content":"fix it"},
      {"type":"reasoning","summary":[{"type":"summary_text","text":"Look first."}]},
      {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Checking."}]},
      {"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"command\": [\"ls\"]}"},
      {"type":"function_call","call_id":"c2","name":"shell","arguments":"{\"command\": [\"pwd\"]}"},
      {"type":"function_call_output","call_id":"c1","output":"a.txt"},
      {"type":"function_call_output","call_id":"c2","output":"/w"}]})");
    const std::string t = ling::serve::parse_responses_request(body).prompt_text;
    EXPECT(count_of(t, "<|im_start|>assistant") == 2);
    EXPECT(count_of(t, "<tool_call>") == 2);
    EXPECT(count_of(t, "<tool_response>") == 2);
    EXPECT(t.find("Look first.") != std::string::npos && t.find("Checking.") != std::string::npos);
  }
  {
    // BACKLOG §19.13: a `type: custom` (freeform) tool is offered to the model as a function with the one
    // string parameter `input`, its grammar appended to its description; the request remembers
    // which names are custom; a replayed custom_tool_call and its output render as that call and its
    // tool response.
    json body = json::parse(R"({"tools":[{"type":"custom","name":"apply_patch","description":"Edit files.",
      "format":{"type":"grammar","syntax":"lark","definition":"start: x"}},
      {"type":"function","name":"shell","description":"Run","parameters":{"type":"object"},"strict":false}],
      "reasoning":{"effort":"none"},
      "input":[{"type":"message","role":"user","content":"fix it"},
      {"type":"custom_tool_call","call_id":"c1","name":"apply_patch","input":"*** Begin Patch\n*** End Patch"},
      {"type":"custom_tool_call_output","call_id":"c1","output":"Done!"}]})");
    const ling::serve::Request r = ling::serve::parse_responses_request(body);
    const std::string& t = r.prompt_text;
    EXPECT(r.custom_tools == std::vector<std::string>{"apply_patch"});
    EXPECT(t.find("{\"type\": \"function\", \"function\": {\"description\": \"Edit files.\\n\\nThe input must follow "
                  "this lark grammar:\\nstart: x\", \"name\": \"apply_patch\", "
                  "\"parameters\": {\"type\": \"object\", \"properties\": {\"input\": {\"type\": \"string\", "
                  "\"description\": \"The tool's input, as free text\"}}, \"required\": [\"input\"]}, "
                  "\"strict\": false}, \"defer_loading\": null}") != std::string::npos);
    EXPECT(t.find("\"name\": \"shell\"") != std::string::npos);
    EXPECT(t.find("\"format\"") == std::string::npos);  // the grammar reaches the model as prose, not as a field
    EXPECT(t.find("<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** End Patch\n</parameter>\n"
                  "</function>\n</tool_call><|im_end|>") != std::string::npos);
    EXPECT(t.find("<tool_response>\nDone!\n</tool_response>") != std::string::npos);
    // The parser hands the call back with `input` as a string, whatever the text looks like, and the
    // custom tool input is that text; a call the model made with some other single parameter, or with
    // no parsable arguments, still yields a usable input.
    OutputParser p(false, r.tools);
    const auto d = p.push("<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** Update File: a.py\n"
                          "@@\n-1\n+2\n*** End Patch\n</parameter>\n</function>\n</tool_call>");
    EXPECT(d.tool_calls.size() == 1 && d.tool_calls[0].name == "apply_patch");
    if (d.tool_calls.size() == 1)
      EXPECT(ling::serve::custom_tool_input(d.tool_calls[0].arguments) ==
             "*** Begin Patch\n*** Update File: a.py\n@@\n-1\n+2\n*** End Patch");
    EXPECT(ling::serve::custom_tool_input("{\"patch\": \"p\"}") == "p");
    EXPECT(ling::serve::custom_tool_input("{\"a\": \"p\", \"b\": \"q\"}") == "{\"a\": \"p\", \"b\": \"q\"}");
    EXPECT(ling::serve::custom_tool_input("not json") == "not json");
    EXPECT(ling::serve::custom_tool_input("{}") == "");
    // A chat tool without a `type`, or with one that is not a string, is still a function tool.
    json untyped = json::parse(R"({"messages":[{"role":"user","content":"hi"}],"tools":[
      {"function":{"name":"shell","parameters":{"type":"object"}}},
      {"type":null,"function":{"name":"edit","parameters":{"type":"object"}}}]})");
    const std::string u = ling::serve::parse_request(untyped, true).prompt_text;
    EXPECT(u.find("\"name\": \"shell\"") != std::string::npos && u.find("\"name\": \"edit\"") != std::string::npos);
    // A function_call_output and a custom_tool_call_output share one wire shape; tool_choice "none"
    // drops custom tools with the rest.
    json none = body;
    none["tool_choice"] = "none";
    EXPECT(ling::serve::parse_responses_request(none).prompt_text.find("# Tools") == std::string::npos);
    // On chat completions (the vLLM-compatible route) a custom tool is dropped, function tools stay.
    json chat = json::parse(R"({"messages":[{"role":"user","content":"hi"}],"tools":[
      {"type":"custom","name":"apply_patch","description":"Edit files.","format":{"type":"grammar","syntax":"lark","definition":"start: x"}},
      {"type":"function","function":{"name":"shell","parameters":{"type":"object"}}}]})");
    const std::string c = ling::serve::parse_request(chat, true).prompt_text;
    EXPECT(c.find("\"name\": \"shell\"") != std::string::npos && c.find("apply_patch") == std::string::npos);
  }
  {
    // The prompt against production's (2026-10-16): a custom tool renders exactly as the function `ling`
    // offers SGLang in its place (ling-rs/tools custom.rs `as_function`), so ling-serve and today's
    // production prefill the same prompt; SGLang drops a custom tool sent as it is, and that prompt is
    // this one less the tool's own line in <tools>. Recorded sessions of 2026-10-07 offered `exec`
    // custom to SGLang: replayed as captured on ling-serve they prefill that line more (991 tokens on
    // M0's main set; bench/replay/replay.py --custom-tools, bench/render/render_test.py).
    const char* input = R"("reasoning":{"effort":"none"},"input":[{"type":"message","role":"user","content":"run it"}])";
    const json custom = json::parse(std::string(R"({"tools":[{"type":"custom","name":"exec","description":"Run JS.",
      "format":{"type":"grammar","syntax":"lark","definition":"start: SOURCE"}},
      {"type":"function","name":"wait","description":"Wait.","strict":false,"parameters":{"type":"object"}}],)") +
                                    input + "}");
    const json function = json::parse(std::string(R"({"tools":[{"type":"function","name":"exec",
      "description":"Run JS.\n\nThe input must follow this lark grammar:\nstart: SOURCE","strict":false,
      "parameters":{"type":"object","properties":{"input":{"type":"string","description":"The tool's input, as free text"}},"required":["input"]}},
      {"type":"function","name":"wait","description":"Wait.","strict":false,"parameters":{"type":"object"}}],)") +
                                      input + "}");
    const json dropped = json::parse(std::string(R"({"tools":[
      {"type":"function","name":"wait","description":"Wait.","strict":false,"parameters":{"type":"object"}}],)") +
                                     input + "}");
    const std::string tc = ling::serve::parse_responses_request(custom).prompt_text;
    const std::string tf = ling::serve::parse_responses_request(function).prompt_text;
    const std::string td = ling::serve::parse_responses_request(dropped).prompt_text;
    EXPECT(tc == tf);
    const std::string line = "\n{\"type\": \"function\", \"function\": {\"description\": \"Run JS.\\n\\nThe input must "
                             "follow this lark grammar:\\nstart: SOURCE\", \"name\": \"exec\", \"parameters\": {\"type\": "
                             "\"object\", \"properties\": {\"input\": {\"type\": \"string\", \"description\": \"The tool's "
                             "input, as free text\"}}, \"required\": [\"input\"]}, \"strict\": false}, \"defer_loading\": null}";
    const size_t at = tc.find(line);
    EXPECT(at != std::string::npos);
    if (at != std::string::npos) EXPECT(std::string(tc).erase(at, line.size()) == td);
  }
  {
    // vllm#53284, vllm#52738: reasoning_effort "none" on chat completions means thinking off, as it
    // already does on the Responses path (it used to be a 400, "Unexpected reasoning effort none").
    json body = json::parse(R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"})");
    bool ok = true, thinking_off = false;
    try {
      thinking_off = !ling::serve::parse_request(body, true).reasoning;
    } catch (const std::exception&) {
      ok = false;
    }
    EXPECT(ok && thinking_off);
    // The same through chat_template_kwargs; and the other efforts are untouched (high still thinks).
    json kw = json::parse(R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"reasoning_effort":"none"}})");
    EXPECT(!ling::serve::parse_request(kw, true).reasoning);
    json high = json::parse(R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"})");
    const ling::serve::Request h = ling::serve::parse_request(high, true);
    EXPECT(h.reasoning && h.prompt_text.find("Reasoning effort is set to xhigh") != std::string::npos);
  }
  {
    // reports/api-compat-checklist.md gap 2: an image part is rendered as the placeholder `[image]` (no
    // vision encoder yet) instead of a 400, which ends the turn in Codex; the text parts stay as they are.
    // On chat completions (Cline's `image_url` part) ...
    json chat = json::parse(R"({"messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}},
      {"type":"text","text":"Be brief."}]}],"chat_template_kwargs":{"enable_thinking":false}})");
    bool ok = true;
    std::string t;
    try {
      t = ling::serve::parse_request(chat, true).prompt_text;
    } catch (const std::exception&) {
      ok = false;
    }
    EXPECT(ok);
    EXPECT(t.find("What is in this image?[image]Be brief.") != std::string::npos);
    EXPECT(t.find("base64") == std::string::npos);
    // ... and on the Responses path (Codex's `input_image` part).
    json resp = json::parse(R"({"reasoning":{"effort":"none"},"input":[{"type":"message","role":"user","content":[
      {"type":"input_text","text":"What is in this image?"},
      {"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}]}]})");
    ok = true;
    t.clear();
    try {
      t = ling::serve::parse_responses_request(resp).prompt_text;
    } catch (const std::exception&) {
      ok = false;
    }
    EXPECT(ok);
    EXPECT(t.find("What is in this image?[image]") != std::string::npos);
    EXPECT(t.find("base64") == std::string::npos);
  }
  {
    // Stop strings while streaming multi-byte text (the sglang#40529 family): the hold-back used to count
    // bytes, split a character, and the chunk's JSON dump then threw and ended the stream. Fed one byte
    // at a time (harsher than the decoder, which hands over whole characters): every piece is valid
    // UTF-8, the pieces add up to the text before the stop string, and the stop string never appears.
    const std::string text = "用中文写三句话。长城很长，很古老。完毕之后再写一句。";
    for (const std::string stop : {"完毕", "END", "很古老", "。"}) {
      ling::serve::StopScanner s({stop});
      std::string out;
      bool pieces_valid = true;
      for (char c : text) {
        const std::string piece = s.push(std::string(1, c));
        pieces_valid = pieces_valid && valid_utf8(piece);
        out += piece;
        if (s.stopped()) break;
      }
      out += s.finish();
      const size_t at = text.find(stop);
      EXPECT(pieces_valid);
      EXPECT(out == (at == std::string::npos ? text : text.substr(0, at)));
      EXPECT(s.stopped() == (at != std::string::npos));
    }
  }
  {
    // The earliest of several stop strings wins; an empty stop string is ignored; no stops: all text.
    ling::serve::StopScanner s({"", "</function", "STOP"});
    std::string out;
    for (const std::string piece : {"call <", "/funct", "ion> and STOP"}) out += s.push(piece);
    EXPECT(s.stopped() && out == "call ");
    ling::serve::StopScanner none({});
    EXPECT(none.push("é") == "é" && none.push("\xE9\x95") == "" && none.push("\xBF") == "\xE9\x95\xBF");
  }
  {
    // vllm#27572 class: a chat stream that fails after its headers still ends with a finish chunk and
    // [DONE] after the error (it used to stop at the error object), and its finish_reason is not "stop".
    const json finish = {{"id", "chatcmpl-1"}, {"object", "chat.completion.chunk"},
                         {"choices", json::array({json{{"index", 0}, {"delta", json::object()},
                                                       {"finish_reason", ling::serve::kStreamErrorFinish}}})}};
    const std::string tail = ling::serve::stream_error_tail(
        json{{"error", {{"message", "bad \xE9"}, {"type", "server_error"}}}}, finish);
    std::vector<std::string> events;
    for (size_t at = 0, end; (end = tail.find("\n\n", at)) != std::string::npos; at = end + 2)
      events.push_back(tail.substr(at, end - at));
    EXPECT(events.size() == 3);
    if (events.size() == 3) {
      EXPECT(events[0].rfind("data: ", 0) == 0 && json::parse(events[0].substr(6)).contains("error"));
      const json chunk = json::parse(events[1].substr(6));
      EXPECT(chunk["choices"][0]["finish_reason"] == "error");
      EXPECT(events[2] == "data: [DONE]");
    }
  }
  {
    // Context overflow in the shape the clients recognise (reports/api-compat-checklist.md §5): code
    // context_length_exceeded, type invalid_request_error, and a message LiteLLM ("maximum context length
    // is") and Cline ("context length", "maximum context") match. A prompt that fills the context leaves
    // no room for one output token, so it overflows too.
    using ling::serve::exceeds_context;
    EXPECT(!exceeds_context(65535, 65536) && exceeds_context(65536, 65536) && exceeds_context(70000, 65536));
    const json e = ling::serve::context_overflow_error(70000, 65536)["error"];
    EXPECT(e["code"] == "context_length_exceeded" && e["type"] == "invalid_request_error");
    const std::string m = e["message"];
    EXPECT(m.rfind("This model's maximum context length is 65536 tokens.", 0) == 0);
    EXPECT(m.find("70000 input tokens") != std::string::npos && m.find("context length") != std::string::npos);
    EXPECT(std::string(ling::serve::ContextOverflow(70000, 65536).what()) == m);
  }
  {
    // A dump never throws on bytes that are not UTF-8 (half a character): they become U+FFFD.
    bool ok = true;
    std::string out;
    try {
      out = ling::serve::dump_json(json{{"delta", std::string("ok \xE9\x95")}});
    } catch (const std::exception&) {
      ok = false;
    }
    EXPECT(ok && valid_utf8(out) && out.find("\xEF\xBF\xBD") != std::string::npos);
  }
  {
    // Logprobs (API §3): the log-softmax of the raw logits, exact values on a small row, in vLLM's order
    // (the chosen token first, then the top k by rank without it; ties to the lower id, as greedy picks).
    using ling::serve::top_logprobs;
    auto near = [](double a, double b) { return std::fabs(a - b) < 1e-5; };
    const std::vector<float> row = {1.f, 3.f, 2.f, 0.f};
    // m = 3; log(e^-2 + 1 + e^-1 + e^-3) = log(1.5530018) = 0.4401897; logprob(i) = row[i] - 3 - 0.4401897.
    const double lse = 0.44018970;
    auto ok = top_logprobs(row, 1, 2);  // the argmax chosen: k entries
    EXPECT(ok.size() == 2);
    if (ok.size() == 2) {
      EXPECT(ok[0].token == 1 && near(ok[0].logprob, -lse));
      EXPECT(ok[1].token == 2 && near(ok[1].logprob, -1 - lse));
      EXPECT(ok[0].logprob <= 0.f);  // the argmax's logprob never rounds above 0
    }
    auto off = top_logprobs(row, 3, 2);  // the chosen token outside the top k: k + 1 entries, chosen first
    EXPECT(off.size() == 3);
    if (off.size() == 3) {
      EXPECT(off[0].token == 3 && near(off[0].logprob, -3 - lse));
      EXPECT(off[1].token == 1 && near(off[1].logprob, -lse));
      EXPECT(off[2].token == 2 && near(off[2].logprob, -1 - lse));
    }
    auto only = top_logprobs(row, 0, 0);  // k = 0: the chosen token alone
    EXPECT(only.size() == 1 && only[0].token == 0 && near(only[0].logprob, -2 - lse));
    auto all = top_logprobs(row, 2, 4);  // k = the vocabulary: every token once, probabilities summing to 1
    double total = 0;
    for (const auto& e : all) total += std::exp(e.logprob);
    EXPECT(all.size() == 4 && near(total, 1.0) && all[0].token == 2 && all[1].token == 1 && all[2].token == 0 && all[3].token == 3);
    const std::vector<float> tie = {2.f, 2.f, 1.f};  // log(2 + e^-1) = 0.8619948
    auto t1 = top_logprobs(tie, 0, 1);
    EXPECT(t1.size() == 1 && t1[0].token == 0 && near(t1[0].logprob, -0.8619948));
    auto t2 = top_logprobs(tie, 1, 1);  // the tie's lower id ranks first; the chosen one still heads the list
    EXPECT(t2.size() == 2 && t2[0].token == 1 && t2[1].token == 0 && near(t2[1].logprob, -0.8619948));
    EXPECT(top_logprobs(std::vector<float>{}, 0, 2).empty());
    // The request fields as vLLM validates them.
    auto parses = [](const char* text, bool chat) {
      try {
        return std::optional<ling::serve::Request>(ling::serve::parse_request(json::parse(text), chat));
      } catch (const std::invalid_argument&) {
        return std::optional<ling::serve::Request>();
      }
    };
    auto c1 = parses(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":3})", true);
    EXPECT(c1 && c1->logprobs && c1->top_logprobs == 3);
    auto c0 = parses(R"({"messages":[{"role":"user","content":"hi"}]})", true);
    EXPECT(c0 && !c0->logprobs && c0->top_logprobs == 0);
    EXPECT(!parses(R"({"messages":[{"role":"user","content":"hi"}],"top_logprobs":2})", true));  // needs logprobs: true
    EXPECT(!parses(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":21})", true));
    EXPECT(!parses(R"({"messages":[{"role":"user","content":"hi"}],"logprobs":2})", true));  // chat: a boolean
    auto p2 = parses(R"({"prompt":"hi","logprobs":2})", false);
    EXPECT(p2 && p2->logprobs && p2->top_logprobs == 2);
    auto p0 = parses(R"({"prompt":"hi","logprobs":0})", false);
    EXPECT(p0 && p0->logprobs && p0->top_logprobs == 0);
    auto pn = parses(R"({"prompt":"hi","logprobs":null})", false);
    EXPECT(pn && !pn->logprobs);
    EXPECT(!parses(R"({"prompt":"hi","logprobs":21})", false) && !parses(R"({"prompt":"hi","logprobs":-1})", false));
  }
  {
    // Nights 12 and 15 (2026-10-21/22): four agent messages that ended a turn as text because the parser kept back a
    // call it would not send, two of them the task's last answer (an empty patch: django-15554, django-13033). The
    // model's own outputs, verbatim. The request offered `exec` (Codex's code mode) and nothing else.
    const json offered = json::parse(R"([{"type":"function","function":{"name":"exec","parameters":{"type":"object",
      "properties":{"input":{"type":"string"}},"required":["input"]}}}])");
    // django-15554: a call to a shell command the model took for a tool. SGLang sends it as a call; the client
    // answers it as unsupported and the turn goes on.
    {
      OutputParser p(false, offered);
      auto d = feed(p, R"LING(<tool_call>
<function=ling-search>
<parameter=--read>
Django FilteredRelation multiple same relation different conditions only one join generated ticket fix
</parameter>
</function>
</tool_call>)LING");
      EXPECT(d.content.empty() && d.tool_calls.size() == 1);
      if (d.tool_calls.size() == 1) {
        EXPECT(d.tool_calls[0].name == "ling-search");
        EXPECT(json::parse(d.tool_calls[0].arguments)["--read"] ==
               "Django FilteredRelation multiple same relation different conditions only one join generated ticket fix");
      }
    }
    // sympy-19040: the function named in a `<parameter=` tag, the offered `exec`.
    {
      OutputParser p(false, offered);
      auto d = feed(p, R"LING(<tool_call>
<parameter=exec>
<parameter=input>
const r = await tools.exec_command({cmd: "cd /testbed && grep -n 'def get_ring' sympy/polys/domains/domain.py sympy/polys/domains/*.py 2>/dev/null; grep -rn 'class PolyRing' sympy/polys/domains/"});
text(r.output);

</parameter>
</function>
</tool_call>)LING");
      EXPECT(d.content.empty() && d.tool_calls.size() == 1);
      if (d.tool_calls.size() == 1) {
        EXPECT(d.tool_calls[0].name == "exec");
        const json a = json::parse(d.tool_calls[0].arguments);
        EXPECT(a.size() == 1 && a["input"].get<std::string>().find("const r = await tools.exec_command(") == 0);
      }
    }
    // sympy-13757: a call to `exec_command` (not offered) that the answer ended after `</function>`, with no
    // `</tool_call>`.
    {
      OutputParser p(false, offered);
      auto d = feed(p, R"LING(<tool_call>
<function=exec_command>
<parameter=cmd>
cd /testbed && python - <<'EOF' 2>&1 | grep -v -i "warn\|collections\|MutableSet\|Callable\|plotting\|OrderedSet\|^  from"
from sympy import Symbol, Poly
from sympy.polys.polytools import Poly as P

orig = P._mul_ground
def traced(self, g):
    print('>> _mul_ground(self=%r, g=%r)' % (self, g))
    try:
        r = orig(self, g)
        print('<< -> %r' % (r,))
        return r
    except Exception as e:
        print('<< raised %r' % (e,))
        raise
P._mul_ground = traced

x = Symbol('x')
print('=== Poly(1,x) * x')
try:
    r = P(1, x) * x
    print('result:', repr(r), type(r))
except Exception as e:
    print('raised', repr(e))
EOF
</parameter>
</function>
)LING");
      EXPECT(d.content.empty() && d.tool_calls.size() == 1);
      if (d.tool_calls.size() == 1) {
        EXPECT(d.tool_calls[0].name == "exec_command");
        EXPECT(json::parse(d.tool_calls[0].arguments)["cmd"].get<std::string>().find("cd /testbed && python - <<") == 0);
      }
    }
    // django-13033: prose, then `apply_patch` with the patch tool off.
    {
      OutputParser p(false, offered);
      auto d = feed(p, R"LING(Applying the patch now.

<tool_call>
<function=apply_patch>
<parameter=files>
[{"filePath": "django/db/models/sql/query.py", "diff": "@@ -1455,6 +1455,27 @@\n             if name == 'pk':\n                 name = opts.pk.name\n \n+            # If the name is the attname of a foreign key whose column lives\n+            # in the current table (e.g. a self-referential foreign key),\n+            # resolve it as the concrete column instead of joining through\n+            # the relation. The column holds the same value, and this avoids\n+            # an unnecessary join.\n+            if (\n+                hasattr(field, 'attname')\n+                and name == field.attname\n+                and pos + 1 == len(names)\n+                and field.many_to_one\n+                and field.model._meta.concrete_model == opts.model\n+            ):\n+                final_field = field.target_field\n+                targets = (final_field,)\n+                cur_names_with_path[1].append(PathInfo(\n+                    from_model=opts.model, to_model=field.related_model,\n+                    direct=True, transform=None, target_fields=(field.target_field,),\n+                    filter_params=field.get_extra_descriptor_filter(),\n+                    path_fields=(),\n+565440000                    join_field=field,\n+                    related_model=field.related_model,\n+                    is_reverse=True,\n+2147483648                    m2m=field.many_to_many,\n+1073741824                    m2m_field=None,\n+33554432                    m2m_reverse_field=None,\n+329985856                    m2m_target_fields=(),\n+                    filtered_relation=None,\n+                ))\n+                names_with_path.append(cur_names_with_path)\n+                break\n+\n             field = None\n             filtered_relation = None\n"}]
</parameter>
</function>
</tool_call>)LING");
      EXPECT(d.content == "Applying the patch now.\n\n" && d.tool_calls.size() == 1);
      if (d.tool_calls.size() == 1) {
        EXPECT(d.tool_calls[0].name == "apply_patch");
        EXPECT(json::parse(d.tool_calls[0].arguments).contains("files"));
      }
    }
    // What stays text: the template's placeholder (vllm#58147, above), a misnamed block whose `<parameter=` name is
    // not offered, a name with spaces, and a call cut off before its `</function>`.
    {
      OutputParser p(false, offered);
      const std::string t = "<tool_call>\n<parameter=shell>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>";
      auto d = feed(p, t);
      EXPECT(d.tool_calls.empty() && d.content == t);
    }
    {
      OutputParser p(false, offered);
      const std::string t = "<tool_call>\n<function=run a command>\n</function>\n</tool_call>";
      auto d = feed(p, t);
      EXPECT(d.tool_calls.empty() && d.content == t);
    }
    {
      OutputParser p(false, offered);
      const std::string t = "<tool_call>\n<function=exec>\n<parameter=input>\ntext(1);";
      auto d = feed(p, t);
      EXPECT(d.tool_calls.empty() && d.content == t);
    }
  }
  // Admission control on host memory (serve/admission.hpp; BATCHING §5.6, phase 0).
  {
    using namespace ling::serve;
    ling::ModelConfig c;  // Qwen3.8-27B's shapes (MODEL §5)
    c.layers = 64;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.lin_k_heads = 16;
    c.lin_v_heads = 48;
    c.lin_k_dim = 128;
    c.lin_v_dim = 128;
    c.conv_kernel = 4;
    for (int i = 0; i < 64; ++i) c.full_attention.push_back(i % 4 == 3);
    // 64 KB of BF16 KV a token; the DeltaNet state 151 MB and the conv window 5.9 MB, twice (snap); the ring 168 MB.
    const size_t s64 = session_bytes(c, 65536, true);
    EXPECT(s64 == size_t(65536) * 65536 + 2 * (size_t(48) * 48 * 128 * 128 * 4 + size_t(48) * 10240 * 3 * 4) +
                      kDraftRingBytes + size_t(65536) * 4);
    EXPECT(s64 > size_t(4.6e9) && s64 < size_t(4.8e9));  // node-b, 2026-10-27: 4.2-4.6 GB a session measured
    EXPECT(session_bytes(c, 262144, true) - s64 == size_t(262144 - 65536) * (65536 + 4));
    const MemoryPlan two = plan_memory(c, size_t(23.47e9), 2, 65536, true);
    EXPECT(two.total() == two.weights + 2 * s64 + kScratchBytes);
    // The start refused when the plan and the reserve exceed what is available (the case that froze node-b would
    // be eight such sessions beside another server), allowed when they fit; and the reason names the shortfall.
    const long need_mb = static_cast<long>(two.total() >> 20);
    EXPECT(refuse_start(two, need_mb + 16384, 16384).empty());
    const std::string why = refuse_start(two, need_mb + 16383, 16384);
    EXPECT(!why.empty() && why.find("short by 1 MB") != std::string::npos);
    EXPECT(refuse_start(plan_memory(c, size_t(23.47e9), 8, 262144, true), 110000, 16384).find("8 sessions") !=
           std::string::npos);
    EXPECT(refuse_start(two, -1, 16384).empty());  // no /proc/meminfo: nothing to judge by
    // A request is refused below the admission reserve only.
    EXPECT(refuse_request(8192, 8192).empty());
    EXPECT(refuse_request(8191, 8192).find("host memory low: 8191 MB") == 0);
    EXPECT(refuse_request(-1, 8192).empty());
    // MemAvailable read from a /proc/meminfo-shaped file.
    const std::string path = "/tmp/ling-api-tests-meminfo";
    if (FILE* f = std::fopen(path.c_str(), "w")) {
      std::fputs("MemTotal:       127535456 kB\nMemFree:         1000 kB\nMemAvailable:   41870336 kB\n", f);
      std::fclose(f);
      EXPECT(mem_available_mb(path) == 40889);
      std::remove(path.c_str());
    }
    EXPECT(mem_available_mb("/nonexistent/meminfo") == -1);

    // Auto-size (BATCHING §5.6, the owner's decision of 2026-10-28): what is not given is chosen by the same plan.
    // node-b with nothing else on the GPU (§3.5): 109,423 MiB available, weights 23.47 GB, the 16,384 MiB reserve.
    const size_t w = size_t(23.47e9), vis = kScratchBytes + (size_t(1100 + 512) << 20);
    const long nodeb = 109423, reserve = 16384;
    auto choose = [&](long avail, size_t scratch, std::optional<int> s, std::optional<int> ctx, bool on = true) {
      return choose_size(c, w, scratch, true, avail, reserve, 262144, s, ctx, on);
    };
    auto fits = [&](int n, int ctx, long avail, size_t scratch) {
      return refuse_start(plan_memory(c, w, n, ctx, true, scratch), avail, reserve).empty();
    };
    for (size_t scratch : {kScratchBytes, vis}) {  // images off (§3.5's table) and on (the default)
      // Four sessions at the full context are refused (§3.5: 1 GB short), three fit: three at 262,144, batch three.
      EXPECT(!fits(4, 262144, nodeb, scratch) && fits(3, 262144, nodeb, scratch));
      const SizeChoice a = choose(nodeb, scratch, {}, {});
      EXPECT(a.sessions == 3 && a.max_context == 262144 && a.sessions_auto && a.context_auto && a.sized);
    }
    // The plan_memory default scratch is the start check's when images are off.
    EXPECT(plan_memory(c, w, 3, 262144, true).total() == plan_memory(c, w, 3, 262144, true, kScratchBytes).total());
    // --sessions 4 given: the longest context four fit, halving from the full one (§3.5: 131,072 passes, 196,608 is
    // not a halving).
    {
      const SizeChoice a = choose(nodeb, kScratchBytes, 4, {});
      EXPECT(a.sessions == 4 && a.max_context == 131072 && !a.sessions_auto && a.context_auto);
      EXPECT(!fits(4, 262144, nodeb, kScratchBytes) && fits(4, 131072, nodeb, kScratchBytes));
    }
    // --max-context given: only the sessions are chosen, up to the cap of four.
    {
      const SizeChoice a = choose(nodeb, vis, {}, 262144);
      EXPECT(a.sessions == 3 && a.max_context == 262144 && a.sessions_auto && !a.context_auto);
      const SizeChoice b = choose(nodeb, kScratchBytes, {}, 65536);
      EXPECT(b.sessions == kAutoSessionsCap && b.max_context == 65536);
      EXPECT(fits(kAutoSessionsCap + 1, 65536, nodeb, kScratchBytes));  // the cap, not the memory, stops it
    }
    // Both given: today's check exactly, nothing chosen (an over-large pair is left to the start check to refuse).
    {
      const SizeChoice a = choose(nodeb, kScratchBytes, 8, 262144);
      EXPECT(a.sessions == 8 && a.max_context == 262144 && !a.sessions_auto && !a.context_auto);
    }
    // A smaller box: one session does not fit at 262,144 or 131,072, so the context halves to 65,536 and the
    // sessions are then as many as fit there (one).
    {
      const SizeChoice a = choose(50000, kScratchBytes, {}, {});
      EXPECT(a.sessions == 1 && a.max_context == 65536);
      EXPECT(!fits(1, 131072, 50000, kScratchBytes) && fits(1, 65536, 50000, kScratchBytes) &&
             !fits(2, 65536, 50000, kScratchBytes));
    }
    // Too small even for one session at the floor: one at the floor, which the start check refuses (exit 3).
    {
      const SizeChoice a = choose(40000, kScratchBytes, {}, {});
      EXPECT(a.sessions == 1 && a.max_context == kAutoContextFloor);
      EXPECT(!refuse_start(plan_memory(c, w, a.sessions, a.max_context, true), 40000, reserve).empty());
      const SizeChoice b = choose(40000, kScratchBytes, 2, {});  // --sessions given: kept, the floor refused
      EXPECT(b.sessions == 2 && b.max_context == kAutoContextFloor);
    }
    // Auto-size off, MemAvailable unknown, or no reserve: today's defaults fill in what was not given.
    for (const SizeChoice& a : {choose(nodeb, kScratchBytes, {}, {}, false), choose(-1, kScratchBytes, {}, {}),
                                choose_size(c, w, kScratchBytes, true, nodeb, 0, 262144, {}, {})}) {
      EXPECT(a.sessions == kDefaultSessions && a.max_context == kDefaultContext && !a.sessions_auto &&
             !a.context_auto && !a.sized);
    }
    {
      const SizeChoice a = choose(nodeb, kScratchBytes, 3, {}, false);
      EXPECT(a.sessions == 3 && a.max_context == kDefaultContext);
    }
    // A model without max_position_embeddings is sized from today's default context, never beyond it.
    {
      const SizeChoice a = choose_size(c, w, kScratchBytes, true, nodeb, reserve, 0, {}, {});
      EXPECT(a.max_context == kDefaultContext && a.sessions == kAutoSessionsCap);
    }
    // The full context read from config.json: under text_config when there is one, else at the top; 0 without.
    const std::string cfg = "/tmp/ling-api-tests-config.json";
    for (auto [text, want] :
         {std::pair<const char*, int>{R"({"text_config": {"max_position_embeddings": 262144}})", 262144},
          {R"({"max_position_embeddings": 131072})", 131072}, {R"({"text_config": {}})", 0}, {"not json", 0}}) {
      if (FILE* f = std::fopen(cfg.c_str(), "w")) {
        std::fputs(text, f);
        std::fclose(f);
        EXPECT(model_full_context(cfg) == want);
        std::remove(cfg.c_str());
      }
    }
    EXPECT(model_full_context("/nonexistent/config.json") == 0);
  }
  if (known_gaps) std::printf("%d known gaps (not failures; see reports/engine-issues-2026-10-09.md)\n", known_gaps);
  std::printf(failures ? "%d API TEST FAILURES\n" : "all api tests passed\n", failures);
  return failures ? 1 : 0;
}
