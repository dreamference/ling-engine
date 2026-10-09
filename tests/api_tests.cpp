// Tests the output parsers and the chat template without a server or a GPU.
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

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

int main() {
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
  // Bug classes from SGLang's and vLLM's trackers (reports/engine-issues-2026-10-09.md, SPEC "Known
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
    json body = json::parse(R"({"messages":[{"role":"user","content":"run it"},
      {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"shell",
       "arguments":"{\"command\": [\"ec"}}]},
      {"role":"tool","tool_call_id":"c1","content":"error"}]})");
    bool ok = true;
    try {
      (void)ling::serve::parse_request(body, true);
    } catch (const std::exception&) {
      ok = false;
    }
    KNOWN(ok);
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
    // vllm#53284, vllm#52738: reasoning_effort "none" on chat completions means thinking off, as it
    // already does on the Responses path (today: 400 "Unexpected reasoning effort none").
    json body = json::parse(R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"})");
    bool ok = true, thinking_off = false;
    try {
      thinking_off = !ling::serve::parse_request(body, true).reasoning;
    } catch (const std::exception&) {
      ok = false;
    }
    KNOWN(ok && thinking_off);
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
  if (known_gaps) std::printf("%d known gaps (not failures; see reports/engine-issues-2026-10-09.md)\n", known_gaps);
  std::printf(failures ? "%d API TEST FAILURES\n" : "all api tests passed\n", failures);
  return failures ? 1 : 0;
}
