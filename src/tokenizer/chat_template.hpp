// The Qwen3.8 chat template (the checkpoint's chat_template.jinja, with the two patches Mightling's
// production server applies) written out in C++, for text messages, tools and tool calls, and images
// (specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.6); video is not supported.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ling {

struct ChatOptions {
  bool add_generation_prompt = true;
  std::optional<bool> enable_thinking;          // undefined = thinking on
  std::optional<std::string> reasoning_effort;  // max/xhigh/high, medium (default), low/minimal
  std::optional<bool> preserve_thinking;        // undefined = keep reasoning of earlier turns
  // Vision on: an image part renders `<|vision_start|><|image_pad|><|vision_end|>` and its URL is appended here, in
  // the order of the placeholders in the prompt (one <|image_pad|> each, which the server expands). Null: an image
  // part renders as the text `[image]` (no vision tower).
  std::vector<std::string>* images = nullptr;
};

// `messages` and `tools` are the request's JSON (chat-completions shape). Throws std::invalid_argument
// with a message for the client on malformed input.
std::string render_chat(const nlohmann::ordered_json& messages, const nlohmann::ordered_json& tools,
                        const ChatOptions& opts);

// Python's json.dumps(value, ensure_ascii=False): ", " and ": " separators, insertion order.
std::string python_json(const nlohmann::ordered_json& v);

}  // namespace ling
