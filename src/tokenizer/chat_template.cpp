#include "tokenizer/chat_template.hpp"

#include <stdexcept>

namespace ling {
namespace {

using json = nlohmann::ordered_json;

std::string trim(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return {};
  size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

bool starts_with(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::string render_content(const json& content, bool is_system) {
  if (content.is_null()) return {};
  if (content.is_string()) return content.get<std::string>();
  if (content.is_array()) {
    std::string out;
    for (const json& item : content) {
      const std::string type = item.value("type", "");
      if (item.contains("image") || item.contains("image_url") || type == "image" || type == "image_url" ||
          item.contains("video") || type == "video") {
        throw std::invalid_argument(is_system ? "System message cannot contain images."
                                              : "ling-engine v0 does not accept images or video");
      }
      if (item.contains("text")) out += item["text"].get<std::string>();
      else throw std::invalid_argument("Unexpected item type in content.");
    }
    return out;
  }
  throw std::invalid_argument("Unexpected content type.");
}

void dump(const json& v, std::string& out) {
  switch (v.type()) {
    case json::value_t::object: {
      out += '{';
      bool first = true;
      for (auto it = v.begin(); it != v.end(); ++it) {
        if (!first) out += ", ";
        first = false;
        out += json(it.key()).dump(-1, ' ', false, json::error_handler_t::replace);
        out += ": ";
        dump(it.value(), out);
      }
      out += '}';
      break;
    }
    case json::value_t::array: {
      out += '[';
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ", ";
        dump(v[i], out);
      }
      out += ']';
      break;
    }
    default:
      out += v.dump(-1, ' ', false, json::error_handler_t::replace);
  }
}

}  // namespace

std::string python_json(const json& v) {
  std::string out;
  dump(v, out);
  return out;
}

std::string render_chat(const json& messages, const json& tools, const ChatOptions& opts) {
  if (!messages.is_array() || messages.empty()) throw std::invalid_argument("No messages provided.");
  std::string out;
  std::string reasoning;
  const bool thinking = !opts.enable_thinking.has_value() || *opts.enable_thinking;
  if (thinking) {
    std::string effort = opts.reasoning_effort.value_or("xhigh");
    if (effort == "high") effort = "xhigh";
    if (effort == "minimal") effort = "low";
    if (effort == "xhigh") {
      reasoning =
          "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
          "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
          "answer.";
    } else if (effort == "low") {
      reasoning =
          "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
          "conclusion without unnecessary elaboration.";
    } else if (effort != "medium") {
      throw std::invalid_argument("Unexpected reasoning effort " + effort +
                                  ". Supported types are xhigh (default), high, medium, low and minimal.");
    }
  }
  const bool has_system = messages[0].value("role", "") == "system";
  if (tools.is_array() && !tools.empty()) {
    out += "<|im_start|>system\n";
    if (!reasoning.empty()) out += reasoning + "\n\n";
    out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const json& t : tools) out += "\n" + python_json(t);
    out += "\n</tools>";
    out +=
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
        "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
        "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
        "the specified format: an inner <function=...></function> block must be nested within "
        "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional "
        "reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If "
        "there is no function call available, answer the question like normal with your current knowledge and "
        "do not tell the user about function calls\n</IMPORTANT>";
    if (has_system) {
      const std::string content = trim(render_content(messages[0]["content"], true));
      if (!content.empty()) out += "\n\n" + content;
    }
    out += "<|im_end|>\n";
  } else if (has_system) {
    const std::string content = trim(render_content(messages[0]["content"], true));
    if (!content.empty()) {
      out += "<|im_start|>system\n" + (reasoning.empty() ? std::string() : reasoning + "\n\n") + content +
             "<|im_end|>\n";
    } else if (!reasoning.empty()) {
      out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    }
  } else if (!reasoning.empty()) {
    out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
  }

  // The last user message that is not just tool responses.
  long last_query = -1;
  for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
    const json& m = messages[i];
    if (m.value("role", "") == "user") {
      const std::string c = trim(render_content(m.contains("content") ? m["content"] : json(), false));
      if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
        last_query = i;
        break;
      }
    }
  }
  if (last_query < 0) throw std::invalid_argument("No user query found in messages.");

  for (size_t i = 0; i < messages.size(); ++i) {
    const json& m = messages[i];
    const std::string role = m.value("role", "");
    const std::string content = trim(render_content(m.contains("content") ? m["content"] : json(), false));
    if (role == "system") {
      if (i != 0) throw std::invalid_argument("System message must be at the beginning.");
    } else if (role == "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (role == "assistant") {
      std::string rc;
      if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) rc = trim(m["reasoning_content"]);
      const bool keep = !opts.preserve_thinking.has_value() || *opts.preserve_thinking ||
                        static_cast<long>(i) > last_query;
      if (keep) out += "<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + content;
      else out += "<|im_start|>assistant\n" + content;
      if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
        bool first = true;
        for (json call : m["tool_calls"]) {
          if (call.contains("function")) call = call["function"];
          const std::string name = call.value("name", "");
          if (first) out += (trim(content).empty() ? "" : "\n\n") + std::string("<tool_call>\n<function=") + name + ">\n";
          else out += "\n<tool_call>\n<function=" + name + ">\n";
          first = false;
          json args = call.contains("arguments") ? call["arguments"] : json();
          if (args.is_string()) {
            const std::string s = args.get<std::string>();
            args = s.empty() ? json() : json::parse(s, nullptr, false);
            if (args.is_discarded()) throw std::invalid_argument("tool call arguments are not valid JSON");
          }
          if (args.is_object()) {
            for (auto it = args.begin(); it != args.end(); ++it) {
              out += "<parameter=" + it.key() + ">\n";
              out += it.value().is_string() ? it.value().get<std::string>() : python_json(it.value());
              out += "\n</parameter>\n";
            }
          }
          out += "</function>\n</tool_call>";
        }
      }
      out += "<|im_end|>\n";
    } else if (role == "tool") {
      if (i > 0 && messages[i - 1].value("role", "") != "tool") out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      const bool last = i + 1 == messages.size();
      if (last || messages[i + 1].value("role", "") != "tool") out += "<|im_end|>\n";
    } else {
      throw std::invalid_argument("Unexpected message role.");
    }
  }
  if (opts.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += (opts.enable_thinking.has_value() && !*opts.enable_thinking) ? "<think>\n\n</think>\n\n" : "<think>\n";
  }
  return out;
}

}  // namespace ling
