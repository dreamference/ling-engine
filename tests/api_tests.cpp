// Tests the output parsers and the chat template without a server or a GPU.
#include <cstdio>
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
  std::printf(failures ? "%d API TEST FAILURES\n" : "all api tests passed\n", failures);
  return failures ? 1 : 0;
}
