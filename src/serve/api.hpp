// The chat-completions and completions API, independent of the HTTP library: request parsing into a
// generation job, and the Qwen3 output parsers (reasoning split, qwen3_coder tool calls).
#pragma once

#include <optional>
#include <stdexcept>
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
  std::vector<std::string> custom_tools;  // Responses `type: custom` tools, offered as one-parameter functions
};

// The one parameter a custom (freeform) tool is offered with: the model has no freeform format, so the
// tool is rendered as a function taking `input` (a string), and the call comes back as a custom_tool_call
// item whose input is that parameter's text.
constexpr const char* kCustomToolInput = "input";

// The input of a custom tool call from the arguments the parser produced: the `input` parameter; failing
// that the single string parameter the model used instead; an empty input for a call without
// parameters; failing all that, the arguments text itself.
std::string custom_tool_input(const std::string& arguments);

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

// JSON for the wire. Bytes that are not valid UTF-8 (a model can emit part of a character) become
// U+FFFD instead of making the dump throw: a dump error must never end a stream.
std::string dump_json(const json& j);

// A prompt that leaves no room for a single output token, in the shape each client recognises as a
// context overflow (reports/api-compat-checklist.md §5): code context_length_exceeded, and a message
// starting "This model's maximum context length is", which LiteLLM (OpenHands) and Cline match. Chat
// completions and completions answer it as HTTP 400 before any header; the streamed Responses API sends
// it as response.failed, the only shape on which Codex compacts the conversation instead of retrying.
inline constexpr const char* kContextLengthExceeded = "context_length_exceeded";
bool exceeds_context(size_t prompt_tokens, int max_context);
std::string context_overflow_message(size_t prompt_tokens, int max_context);
json context_overflow_error(size_t prompt_tokens, int max_context);  // {"error": {message, type, code}}

// Thrown by the engine worker for a prompt that is too long (behind the HTTP handler's own check).
class ContextOverflow : public std::invalid_argument {
 public:
  ContextOverflow(size_t prompt_tokens, int max_context)
      : std::invalid_argument(context_overflow_message(prompt_tokens, max_context)) {}
};

// The finish_reason of a chat-completions or completions stream that failed after its headers were sent.
// Not "stop": a client must not take a broken answer for a complete one. Clients that do not know the
// value treat it as an unknown reason, and the error object before it says what happened.
inline constexpr const char* kStreamErrorFinish = "error";

// The end of a chat-completions or completions stream that failed after its headers were sent: the error
// object, then `finish_chunk` (a chunk whose finish_reason is kStreamErrorFinish), then [DONE], so the
// stream still ends the way the clients expect (Cline, through its SDK, and LiteLLM wait for them).
std::string stream_error_tail(const json& error, const json& finish_chunk);

// Stop strings over streamed text. Text is emitted once no stop string can still start in it, and only up
// to a UTF-8 character boundary: holding back a number of bytes could split a character, and the
// chunk's JSON dump then failed and ended the stream.
class StopScanner {
 public:
  explicit StopScanner(const std::vector<std::string>& stops);  // empty strings are ignored
  // Appends generated text and returns what can be emitted now. Once a stop string appears, returns the
  // text before it and sets stopped(); nothing after it is ever returned.
  std::string push(const std::string& text);
  // The text still held back, at the end of generation (empty once stopped).
  std::string finish();
  bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  size_t hold_ = 0;      // the longest stop string's length - 1: the bytes that may still start one
  std::string pending_;  // generated, not yet emitted; it starts on a character boundary
  bool stopped_ = false;
};

}  // namespace ling::serve
