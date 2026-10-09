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

// The Responses API (what Mightling's agent calls), converted to a chat request the way SGLang does:
// instructions and developer messages become one leading system message, function calls and their
// outputs become assistant tool calls and tool messages, reasoning items become reasoning_content, and
// only `function` tools reach the template.
Request parse_responses_request(const json& body);

struct ToolCall {
  std::string name;
  std::string arguments;  // JSON text
};

// Splits generated text into reasoning, content and tool calls as it streams in.
//
// A `<tool_call>` marker starts a call only when it is followed by `<function=` and the function is one
// the request offered, and never inside a markdown code fence: an example in a fence, the template's own
// placeholder name, or a marker quoted in prose stays text (vllm#57541, vllm#58147, vllm#56658).
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
  // Whether the content seen so far ends inside a markdown code fence: a line that starts (after spaces or
  // tabs) with three or more backticks or tildes opens one; a line holding only a run of the same
  // character, at least as long, closes it. Fed incrementally; a line is judged when its newline arrives.
  class FenceTracker {
   public:
    void feed(const std::string& text);
    bool open() const { return open_; }

   private:
    void end_line();
    bool open_ = false;
    char fence_char_ = 0;
    size_t fence_len_ = 0;
    // The current line: still in its indentation, then its run of one fence character, then whether
    // anything but whitespace followed the run.
    bool indent_ = true;
    char run_char_ = 0;
    size_t run_len_ = 0;
    bool in_run_ = false, tail_ = false;
  };

  std::string convert_value(const std::string& function, const std::string& param, const std::string& raw) const;
  std::optional<ToolCall> parse_call(const std::string& block) const;
  bool offered(const std::string& function) const;
  void emit_content(Delta& d, const std::string& text);
  Delta drain(bool final);

  bool in_reasoning_;
  int skip_newlines_ = 0;
  json tools_;
  std::string buf_;
  FenceTracker fence_;
};

std::string new_id(const std::string& prefix);

}  // namespace ling::serve
