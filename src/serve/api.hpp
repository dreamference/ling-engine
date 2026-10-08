// The chat-completions and completions API, independent of the HTTP library: request parsing into a
// generation job, and the Qwen3 output parsers (reasoning split, qwen3_coder tool calls).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/engine.hpp"

namespace ling::serve {

using json = nlohmann::ordered_json;

struct Request {
  bool chat = true;
  bool stream = false;
  bool include_usage = false;
  std::string prompt_text;          // the rendered prompt (chat) or the given prompt (completions)
  std::vector<int> prompt_ids;      // set when a completions request sends token ids
  int max_tokens = 0;               // 0 = up to the context limit
  SamplingParams sampling;
  std::vector<std::string> stop;
  bool reasoning = true;            // the prompt ends inside <think> (chat with thinking on)
  json tools = json::array();
};

// Throws std::invalid_argument with a message for the client.
Request parse_request(const json& body, bool chat);

struct ToolCall {
  std::string name;
  std::string arguments;  // JSON text
};

// Splits generated text into reasoning, content and tool calls as it streams in.
class OutputParser {
 public:
  OutputParser(bool reasoning, const json& tools) : in_reasoning_(reasoning), tools_(tools) {}

  struct Delta {
    std::string reasoning;
    std::string content;
    std::vector<ToolCall> tool_calls;
  };
  Delta push(const std::string& text);
  Delta finish();

 private:
  std::string convert_value(const std::string& function, const std::string& param, const std::string& raw) const;
  std::optional<ToolCall> parse_call(const std::string& block) const;
  Delta drain(bool final);

  bool in_reasoning_;
  json tools_;
  std::string buf_;
};

std::string new_id(const std::string& prefix);

}  // namespace ling::serve
