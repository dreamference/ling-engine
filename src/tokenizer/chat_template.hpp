// The Qwen3.8 chat template (the checkpoint's chat_template.jinja) written out in C++, for text
// messages, tools and tool calls. Images and video are not supported in v0.
#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace ling {

struct ChatOptions {
  bool add_generation_prompt = true;
  std::optional<bool> enable_thinking;          // undefined = thinking on
  std::optional<std::string> reasoning_effort;  // xhigh (default), high, medium, low, minimal
  std::optional<bool> preserve_thinking;        // undefined = keep reasoning of earlier turns
};

// `messages` and `tools` are the request's JSON (chat-completions shape). Throws std::invalid_argument
// with a message for the client on malformed input.
std::string render_chat(const nlohmann::ordered_json& messages, const nlohmann::ordered_json& tools,
                        const ChatOptions& opts);

// Python's json.dumps(value, ensure_ascii=False): ", " and ": " separators, insertion order.
std::string python_json(const nlohmann::ordered_json& v);

}  // namespace ling
