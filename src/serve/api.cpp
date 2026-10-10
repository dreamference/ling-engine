#include "serve/api.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

#include "tokenizer/chat_template.hpp"

namespace ling::serve {
namespace {

const std::string kThinkEnd = "</think>";
const std::string kCallOpen = "<tool_call>";
const std::string kCallClose = "</tool_call>";
const std::string kFunctionOpen = "<function=";

// Longest suffix of `s` that is a proper prefix of `tag` (text that might still become the tag).
size_t partial_tag(const std::string& s, const std::string& tag) {
  for (size_t n = std::min(s.size(), tag.size() - 1); n > 0; --n)
    if (s.compare(s.size() - n, n, tag, 0, n) == 0) return n;
  return 0;
}

std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return {};
  return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

template <typename T>
T get_or(const json& body, const char* key, T dflt) {
  if (!body.contains(key) || body[key].is_null()) return dflt;
  return body[key].get<T>();
}

// The messages as production's chat endpoint hands them to the template: a tool message whose content is
// a list of text parts becomes their text joined by spaces, and a final assistant message with string
// content is sent as a user message (SGLang's handling when continue_final_message is off).
json production_messages(const json& messages) {
  if (!messages.is_array()) return messages;
  json out = messages;
  for (json& m : out) {
    if (!m.is_object() || m.value("role", "") != "tool" || !m.contains("content") || !m["content"].is_array()) continue;
    bool text_only = true;
    for (const json& p : m["content"]) text_only = text_only && (p.is_string() || (p.is_object() && p.value("type", "") == "text"));
    if (!text_only) continue;
    std::string joined;
    bool first = true;
    for (const json& p : m["content"]) {
      joined += (first ? "" : " ") + (p.is_string() ? p.get<std::string>() : p.contains("text") && p["text"].is_string() ? p["text"].get<std::string>() : "");
      first = false;
    }
    m["content"] = joined;
  }
  if (!out.empty() && out.back().is_object() && out.back().value("role", "") == "assistant" &&
      out.back().contains("content") && out.back()["content"].is_string())
    out.back() = json{{"role", "user"}, {"content", out.back()["content"]}};
  return out;
}

// The tools as production's server hands them to the chat template: each one validated into its
// `Tool`/`Function` models and dumped again, which fixes the keys and their order:
// {"type", "function": {"description", "name", "parameters", "strict"}, "defer_loading"}. A named
// tool_choice keeps only that tool. Only `function` tools are offered on this route: any other type (a
// Responses `custom` tool sent to chat completions) is dropped, as the vLLM-compatible surface has no
// other kind (the Responses route converts custom tools before reaching here).
json canonical_tools(const json& tools, const json& tool_choice) {
  std::string only;
  if (tool_choice.is_object() && tool_choice.contains("function") && tool_choice["function"].is_object())
    only = tool_choice["function"].value("name", "");
  json out = json::array();
  for (const json& t : tools) {
    if (!t.is_object() || t.value("type", "function") != "function") continue;
    const json& f = t.contains("function") && t["function"].is_object() ? t["function"] : t;
    json fn = json::object();
    fn["description"] = f.contains("description") && f["description"].is_string() ? f["description"] : json();
    fn["name"] = f.value("name", "");
    fn["parameters"] = f.contains("parameters") ? f["parameters"] : json();
    fn["strict"] = f.contains("strict") && f["strict"].is_boolean() ? f["strict"].get<bool>() : false;
    const json defer = t.contains("defer_loading") && t["defer_loading"].is_boolean() ? t["defer_loading"] : json();
    if (!defer.is_null() || (f.contains("defer_loading") && f["defer_loading"].is_boolean()))
      fn["defer_loading"] = defer.is_null() ? f["defer_loading"] : defer;
    if (!only.empty() && fn["name"] != only) continue;
    out.push_back(json{{"type", t.value("type", "function")}, {"function", fn}, {"defer_loading", defer}});
  }
  return out;
}

}  // namespace

std::string new_id(const std::string& prefix) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* hex = "0123456789abcdef";
  std::string id = prefix;
  for (int i = 0; i < 24; ++i) id += hex[rng() & 15];
  return id;
}

std::string dump_json(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

bool exceeds_context(size_t prompt_tokens, int max_context) {
  return prompt_tokens >= static_cast<size_t>(std::max(max_context, 0));
}

std::string context_overflow_message(size_t prompt_tokens, int max_context) {
  return "This model's maximum context length is " + std::to_string(max_context) + " tokens. However, your request has " +
         std::to_string(prompt_tokens) + " input tokens. Please reduce the length of the messages.";
}

json context_overflow_error(size_t prompt_tokens, int max_context) {
  return json{{"error",
               {{"message", context_overflow_message(prompt_tokens, max_context)},
                {"type", "invalid_request_error"},
                {"code", kContextLengthExceeded}}}};
}

std::string stream_error_tail(const json& error, const json& finish_chunk) {
  return "data: " + dump_json(error) + "\n\ndata: " + dump_json(finish_chunk) + "\n\ndata: [DONE]\n\n";
}

namespace {

// The largest b <= end such that s[0, b) does not end inside a UTF-8 character (s starts on a boundary).
// An invalid sequence is left as it is; dump_json replaces it.
size_t utf8_cut(const std::string& s, size_t end) {
  size_t lead = end;  // back over continuation bytes to the first byte of the last character
  while (lead > 0 && end - lead < 4 && (static_cast<unsigned char>(s[lead - 1]) & 0xC0) == 0x80) --lead;
  if (lead == 0) return end;
  const unsigned char c = static_cast<unsigned char>(s[lead - 1]);
  const size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
  return end - (lead - 1) >= need ? end : lead - 1;
}

}  // namespace

StopScanner::StopScanner(const std::vector<std::string>& stops) {
  for (const std::string& s : stops) {
    if (s.empty()) continue;  // "" would match at once and end every answer before it starts
    stops_.push_back(s);
    hold_ = std::max(hold_, s.size() - 1);
  }
}

std::string StopScanner::push(const std::string& text) {
  if (stopped_) return {};
  pending_ += text;
  // A stop string never starts in text already emitted: it would have to end in the held-back bytes,
  // and they are fewer than its length. So the search covers pending_ alone.
  size_t stop_at = std::string::npos;
  for (const std::string& s : stops_) stop_at = std::min(stop_at, pending_.find(s));
  if (stop_at != std::string::npos) {
    stopped_ = true;
    std::string out = pending_.substr(0, stop_at);
    pending_.clear();
    return out;
  }
  const size_t safe = utf8_cut(pending_, pending_.size() - std::min(pending_.size(), hold_));
  std::string out = pending_.substr(0, safe);
  pending_.erase(0, safe);
  return out;
}

std::string StopScanner::finish() {
  std::string out;
  if (!stopped_) out.swap(pending_);
  return out;
}

Request parse_request(const json& body, bool chat) {
  if (!body.is_object()) throw std::invalid_argument("the request body must be a JSON object");
  Request r;
  r.chat = chat;
  r.stream = get_or(body, "stream", false);
  if (body.contains("stream_options") && body["stream_options"].is_object())
    r.include_usage = body["stream_options"].value("include_usage", false);
  if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number())
    r.max_tokens = body["max_completion_tokens"];
  else if (body.contains("max_tokens") && body["max_tokens"].is_number())
    r.max_tokens = body["max_tokens"];
  // The checkpoint's generation_config defaults, which the production server also uses.
  r.sampling.temperature = get_or(body, "temperature", 1.0f);
  r.sampling.top_p = get_or(body, "top_p", 0.95f);
  r.sampling.top_k = get_or(body, "top_k", 20);
  r.sampling.min_p = get_or(body, "min_p", 0.0f);
  r.sampling.presence_penalty = get_or(body, "presence_penalty", 0.0f);
  r.sampling.repetition_penalty = get_or(body, "repetition_penalty", 1.0f);
  if (body.contains("seed") && body["seed"].is_number_integer()) r.sampling.seed = body["seed"].get<uint64_t>();
  if (body.contains("stop")) {
    if (body["stop"].is_string()) r.stop.push_back(body["stop"]);
    else if (body["stop"].is_array())
      for (const auto& s : body["stop"]) r.stop.push_back(s);
  }
  if (chat) {
    if (!body.contains("messages")) throw std::invalid_argument("messages is required");
    ChatOptions opts;
    const json kw = body.contains("chat_template_kwargs") ? body["chat_template_kwargs"] : json::object();
    if (kw.contains("enable_thinking")) opts.enable_thinking = kw["enable_thinking"].get<bool>();
    if (body.contains("enable_thinking")) opts.enable_thinking = body["enable_thinking"].get<bool>();
    if (kw.contains("preserve_thinking")) opts.preserve_thinking = kw["preserve_thinking"].get<bool>();
    if (body.contains("preserve_thinking")) opts.preserve_thinking = body["preserve_thinking"].get<bool>();
    if (kw.contains("reasoning_effort")) opts.reasoning_effort = kw["reasoning_effort"].get<std::string>();
    if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string())
      opts.reasoning_effort = body["reasoning_effort"].get<std::string>();
    if (body.contains("tools") && body["tools"].is_array() && body.value("tool_choice", json("auto")) != "none")
      r.tools = canonical_tools(body["tools"], body.contains("tool_choice") ? body["tool_choice"] : json("auto"));
    r.prompt_text = render_chat(production_messages(body["messages"]), r.tools, opts);
    r.reasoning = !opts.enable_thinking.has_value() || *opts.enable_thinking;
  } else {
    if (!body.contains("prompt")) throw std::invalid_argument("prompt is required");
    const json& p = body["prompt"];
    if (p.is_string()) r.prompt_text = p;
    else if (p.is_array() && !p.empty() && p[0].is_number_integer()) r.prompt_ids = p.get<std::vector<int>>();
    else if (p.is_array() && p.size() == 1 && p[0].is_string()) r.prompt_text = p[0];
    else throw std::invalid_argument("prompt must be a string or a list of token ids");
    r.reasoning = false;
  }
  return r;
}

OutputParser::Delta OutputParser::push(const std::string& text) {
  buf_ += text;
  return drain(false);
}

OutputParser::Delta OutputParser::finish() { return drain(true); }

void OutputParser::FenceTracker::feed(const std::string& text) {
  for (char c : text) {
    if (c == '\n') {
      end_line();
      continue;
    }
    if (indent_ && (c == ' ' || c == '\t')) continue;
    if (indent_) {
      indent_ = false;
      if (c == '`' || c == '~') {
        run_char_ = c;
        run_len_ = 1;
        in_run_ = true;
      } else {
        tail_ = true;
      }
      continue;
    }
    if (in_run_ && c == run_char_) {
      ++run_len_;
      continue;
    }
    in_run_ = false;
    if (c != ' ' && c != '\t' && c != '\r') tail_ = true;
  }
}

void OutputParser::FenceTracker::end_line() {
  if (run_len_ >= 3) {
    if (!open_) {  // an opening fence may carry an info string ("```xml")
      open_ = true;
      fence_char_ = run_char_;
      fence_len_ = run_len_;
    } else if (run_char_ == fence_char_ && run_len_ >= fence_len_ && !tail_) {
      open_ = false;
    }
  }
  indent_ = true;
  run_char_ = 0;
  run_len_ = 0;
  in_run_ = tail_ = false;
}

void OutputParser::emit_content(Delta& d, const std::string& text) {
  d.content += text;
  fence_.feed(text);
}

bool OutputParser::offered(const std::string& function) const {
  for (const json& t : tools_) {
    if (!t.is_object()) continue;
    const json& fn = t.contains("function") && t["function"].is_object() ? t["function"] : t;
    if (fn.value("name", "") == function) return true;
  }
  return false;
}

OutputParser::Delta OutputParser::drain(bool final) {
  Delta d;
  while (!buf_.empty()) {
    if (in_reasoning_) {
      size_t end = buf_.find(kThinkEnd);
      if (end != std::string::npos) {
        d.reasoning += buf_.substr(0, end);
        buf_.erase(0, end + kThinkEnd.size());
        in_reasoning_ = false;
        skip_newlines_ = 2;  // the template puts "\n\n" after </think>; it belongs to neither part
        continue;
      }
      const size_t hold = final ? 0 : partial_tag(buf_, kThinkEnd);
      d.reasoning += buf_.substr(0, buf_.size() - hold);
      buf_.erase(0, buf_.size() - hold);
      break;
    }
    while (skip_newlines_ > 0 && !buf_.empty() && buf_[0] == '\n') {
      buf_.erase(0, 1);
      --skip_newlines_;
    }
    if (!buf_.empty()) skip_newlines_ = 0;
    if (buf_.empty()) break;
    if (tools_.is_array() && !tools_.empty()) {
      const size_t open = buf_.find(kCallOpen);
      if (open == std::string::npos) {
        const size_t hold = final ? 0 : partial_tag(buf_, kCallOpen);
        emit_content(d, buf_.substr(0, buf_.size() - hold));
        buf_.erase(0, buf_.size() - hold);
        break;
      }
      emit_content(d, buf_.substr(0, open));
      buf_.erase(0, open);
      // buf_ starts with the marker. It opens a call only outside a code fence and when `<function=`
      // follows it (after whitespace); otherwise it is quoted text, released at once rather than held
      // until the end of the answer.
      const size_t after = buf_.find_first_not_of(" \t\r\n", kCallOpen.size());
      const std::string next = after == std::string::npos ? std::string() : buf_.substr(after, kFunctionOpen.size());
      const bool call_follows = next == kFunctionOpen;
      const bool may_follow = !call_follows && next.size() < kFunctionOpen.size() &&
                              kFunctionOpen.compare(0, next.size(), next) == 0;
      if (fence_.open() || (!call_follows && !(may_follow && !final))) {
        emit_content(d, kCallOpen);
        buf_.erase(0, kCallOpen.size());
        continue;
      }
      if (!call_follows) break;  // too little text yet to tell
      const size_t close = buf_.find(kCallClose);
      if (close == std::string::npos) {
        if (final) {  // an unterminated call: hand it back as text
          emit_content(d, buf_);
          buf_.clear();
        }
        break;
      }
      const std::string block = buf_.substr(kCallOpen.size(), close - kCallOpen.size());
      buf_.erase(0, close + kCallClose.size());
      auto call = parse_call(block);
      if (call && offered(call->name)) d.tool_calls.push_back(*call);
      else emit_content(d, kCallOpen + block + kCallClose);  // not a call to anything the request offered
      continue;
    }
    d.content += buf_;
    buf_.clear();
  }
  if (!d.tool_calls.empty()) {
    // Text between or after calls is whitespace in practice; drop it rather than send stray newlines.
    if (strip(d.content).empty()) d.content.clear();
  }
  return d;
}

std::optional<ToolCall> OutputParser::parse_call(const std::string& block) const {
  const std::string fopen = "<function=";
  size_t f = block.find(fopen);
  if (f == std::string::npos) return std::nullopt;
  size_t fend = block.find('>', f);
  if (fend == std::string::npos) return std::nullopt;
  ToolCall call;
  call.name = strip(block.substr(f + fopen.size(), fend - f - fopen.size()));
  json args = json::object();
  size_t pos = fend + 1;
  const std::string popen = "<parameter=", pclose = "</parameter>";
  while (true) {
    size_t p = block.find(popen, pos);
    if (p == std::string::npos) break;
    size_t pend = block.find('>', p);
    if (pend == std::string::npos) break;
    const std::string name = strip(block.substr(p + popen.size(), pend - p - popen.size()));
    // A value ends at its </parameter>. One the model left open ends where the next parameter starts on
    // a line of its own, as the template writes them (vllm#57699), so it cannot swallow its neighbour;
    // markup inside a value that is not at a line start stays part of the value. A last value left open
    // ends at </function>.
    const size_t close = block.find(pclose, pend);
    const size_t next = block.find("\n" + popen, pend);
    size_t vend = std::min(close, next);
    if (vend == std::string::npos) vend = block.find("</function>", pend);
    if (vend == std::string::npos) vend = block.size();
    std::string value = block.substr(pend + 1, vend - pend - 1);
    if (!value.empty() && value.front() == '\n') value.erase(0, 1);
    if (!value.empty() && value.back() == '\n') value.pop_back();
    const std::string converted = convert_value(call.name, name, value);
    json parsed = json::parse(converted, nullptr, false);
    args[name] = parsed.is_discarded() ? json(value) : parsed;
    pos = vend == close ? close + pclose.size() : vend;
  }
  call.arguments = args.dump();
  return call;
}

std::string OutputParser::convert_value(const std::string& function, const std::string& param,
                                        const std::string& raw) const {
  std::string type = "string";
  for (const json& t : tools_) {
    const json& fn = t.contains("function") ? t["function"] : t;
    if (fn.value("name", "") != function) continue;
    if (fn.contains("parameters") && fn["parameters"].contains("properties") &&
        fn["parameters"]["properties"].contains(param)) {
      const json& prop = fn["parameters"]["properties"][param];
      if (prop.contains("type") && prop["type"].is_string()) type = prop["type"];
      else if (prop.contains("type") && prop["type"].is_array() && !prop["type"].empty()) type = prop["type"][0];
      else type = "object";
    }
  }
  if (type == "string") return json(raw).dump();
  json parsed = json::parse(strip(raw), nullptr, false);
  if (!parsed.is_discarded()) return parsed.dump();
  return json(raw).dump();
}

}  // namespace ling::serve

namespace ling::serve {
namespace {

std::string text_of_parts(const json& parts) {
  std::string out;
  if (parts.is_string()) return parts.get<std::string>();
  if (!parts.is_array()) return out;
  for (const json& p : parts)
    if (p.is_object() && p.contains("text") && p["text"].is_string()) out += p["text"].get<std::string>();
  return out;
}

std::string call_id_of(const json& item) {
  return item.contains("call_id") && item["call_id"].is_string() && !item["call_id"].get<std::string>().empty()
             ? item["call_id"].get<std::string>()
             : item.value("id", "");
}

// One Responses input item as a chat message, or null to drop it.
json normalize_item(const json& item) {
  const std::string type = item.value("type", "message");
  if (type == "custom_tool_call") {
    // Replayed as the one-parameter function call it was offered as, so the turn renders as the model
    // wrote it and the prefix cache still matches.
    json args = json::object();
    args[kCustomToolInput] = item.contains("input") && item["input"].is_string() ? item["input"] : json("");
    return json{{"role", "assistant"},
                {"tool_calls", json::array({json{{"id", call_id_of(item)},
                                                 {"type", "function"},
                                                 {"function", {{"name", item.value("name", "")}, {"arguments", args.dump()}}}}})}};
  }
  if (type == "function_call") {
    std::string args = "{}";
    if (item.contains("arguments")) {
      const json& raw = item["arguments"];
      if (raw.is_string()) {
        json parsed = json::parse(raw.get<std::string>(), nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) args = raw.get<std::string>();
      } else if (raw.is_object()) {
        args = raw.dump();
      }
    }
    return json{{"role", "assistant"},
                {"tool_calls", json::array({json{{"id", call_id_of(item)},
                                                 {"type", "function"},
                                                 {"function", {{"name", item.value("name", "")}, {"arguments", args}}}}})}};
  }
  if (type == "function_call_output" || type == "custom_tool_call_output") {
    const json& out = item.contains("output") ? item["output"] : json("");
    return json{{"role", "tool"}, {"tool_call_id", item.value("call_id", "")}, {"content", text_of_parts(out)}};
  }
  if (type == "reasoning") {
    std::string text;
    auto collect = [&](const char* key) {
      if (!item.contains(key) || !item[key].is_array()) return;
      for (const json& e : item[key]) {
        if (e.is_object() && e.contains("text") && e["text"].is_string() && !e["text"].get<std::string>().empty()) {
          if (!text.empty()) text += "\n";
          text += e["text"].get<std::string>();
        }
      }
    };
    collect("summary");
    if (text.empty()) collect("content");
    if (text.empty()) return json();
    return json{{"role", "assistant"}, {"reasoning_content", text}};
  }
  if (type != "message") return json();  // built-in tool calls this server never offered
  std::string role = item.value("role", "user");
  if (role == "developer") role = "system";
  json msg = {{"role", role}};
  // Content stays a list of parts when it is one (the template concatenates their text): input_text and
  // output_text parts become text parts, anything else is kept for the template to accept or refuse.
  if (item.contains("content") && !item["content"].is_null()) {
    const json& c = item["content"];
    if (!c.is_array()) {
      msg["content"] = c;
    } else {
      json parts = json::array();
      for (const json& p : c) {
        const std::string pt = p.is_object() ? p.value("type", "") : "";
        if (pt == "input_text" || pt == "output_text")
          parts.push_back({{"type", "text"}, {"text", p.contains("text") && p["text"].is_string() ? p["text"] : json("")}});
        else
          parts.push_back(p);
      }
      msg["content"] = parts;
    }
  }
  return msg;
}

bool empty_content(const json& m) {
  return !m.contains("content") || m["content"].is_null() || (m["content"].is_string() && m["content"].get<std::string>().empty());
}

json as_parts(const json& c) {
  if (c.is_array()) return c;
  if (c.is_string() && !c.get<std::string>().empty()) return json::array({json{{"type", "text"}, {"text", c}}});
  return json::array();
}

}  // namespace

Request parse_responses_request(const json& body) {
  if (!body.is_object()) throw std::invalid_argument("the request body must be a JSON object");
  json messages = json::array();
  if (body.contains("instructions") && body["instructions"].is_string() && !body["instructions"].get<std::string>().empty())
    messages.push_back({{"role", "system"}, {"content", body["instructions"]}});
  if (!body.contains("input")) throw std::invalid_argument("input is required");
  if (body["input"].is_string()) {
    messages.push_back({{"role", "user"}, {"content", body["input"]}});
  } else if (body["input"].is_array()) {
    for (const json& item : body["input"]) {
      json m = normalize_item(item);
      if (!m.is_null()) messages.push_back(std::move(m));
    }
  } else {
    throw std::invalid_argument("input must be a string or a list of items");
  }
  // One assistant turn arrives as several items (reasoning, message, function calls): merge them.
  json merged = json::array();
  for (json& m : messages) {
    if (m["role"] == "assistant" && !merged.empty() && merged.back()["role"] == "assistant") {
      json& prev = merged.back();
      // As production merges them: two strings join with a blank line, anything else as parts.
      if (!empty_content(m)) {
        if (empty_content(prev)) {
          prev["content"] = m["content"];
        } else if (prev["content"].is_string() && m["content"].is_string()) {
          prev["content"] = prev["content"].get<std::string>() + "\n\n" + m["content"].get<std::string>();
        } else {
          json parts = as_parts(prev["content"]);
          for (const json& p : as_parts(m["content"])) parts.push_back(p);
          prev["content"] = parts;
        }
      }
      if (m.contains("tool_calls")) {
        if (!prev.contains("tool_calls")) prev["tool_calls"] = json::array();
        for (const json& c : m["tool_calls"]) prev["tool_calls"].push_back(c);
      }
      if (m.contains("reasoning_content")) {
        const std::string nr = m["reasoning_content"];
        prev["reasoning_content"] = prev.contains("reasoning_content")
                                        ? prev["reasoning_content"].get<std::string>() + "\n" + nr
                                        : nr;
      }
      continue;
    }
    merged.push_back(std::move(m));
  }
  // Every system chunk goes into one leading system message, joined by blank lines: a string content is
  // one chunk (even an empty one), a list contributes each part's text as a chunk of its own.
  std::vector<std::string> chunks;
  json others = json::array();
  for (json& m : merged) {
    if (m["role"] == "system") {
      const json c = m.contains("content") ? m["content"] : json();
      if (c.is_string()) chunks.push_back(c.get<std::string>());
      else if (c.is_array())
        for (const json& p : c)
          if (p.is_object() && p.contains("text") && p["text"].is_string()) chunks.push_back(p["text"].get<std::string>());
    } else {
      others.push_back(std::move(m));
    }
  }
  json final_messages = json::array();
  if (!chunks.empty()) {
    std::string system;
    for (size_t i = 0; i < chunks.size(); ++i) system += (i ? "\n\n" : "") + chunks[i];
    final_messages.push_back({{"role", "system"}, {"content", system}});
  }
  for (json& m : others) final_messages.push_back(std::move(m));

  json chat = json::object();
  chat["messages"] = final_messages;
  json tools = json::array();
  std::vector<std::string> custom;
  if (body.contains("tools") && body["tools"].is_array()) {
    for (const json& t : body["tools"]) {
      if (!t.is_object()) continue;
      const std::string type = t.value("type", "");
      if (type == "custom") {
        // A custom (freeform) tool, as Codex offers apply_patch under some model families: the model has
        // no freeform format, so it is offered as a function with the one string parameter `input`. Its
        // description is rendered with the grammar from `format` appended (without it the model wrote a
        // unified diff where apply_patch's syntax was wanted, 2026-10-10); the grammar is not enforced
        // (API §4).
        json fn = {{"name", t.value("name", "")}};
        std::string description = t.contains("description") && t["description"].is_string() ? t["description"] : "";
        if (t.contains("format") && t["format"].is_object() && t["format"].value("type", "") == "grammar" &&
            t["format"].contains("definition") && t["format"]["definition"].is_string()) {
          const std::string syntax = t["format"].value("syntax", "");
          description += (description.empty() ? "" : "\n\n") + std::string("The input must follow this ") +
                         (syntax.empty() ? "" : syntax + " ") + "grammar:\n" + t["format"]["definition"].get<std::string>();
        }
        fn["description"] = description.empty() ? json() : json(description);
        json prop = json::object();
        prop["type"] = "string";
        prop["description"] = "The tool's input, as free text";
        json props = json::object();
        props[kCustomToolInput] = prop;
        fn["parameters"] = json::object();
        fn["parameters"]["type"] = "object";
        fn["parameters"]["properties"] = props;
        fn["parameters"]["required"] = json::array({kCustomToolInput});
        fn["strict"] = json();
        tools.push_back({{"type", "function"}, {"function", fn}});
        custom.push_back(fn["name"].get<std::string>());
        continue;
      }
      if (type != "function") continue;
      json fn = {{"name", t.value("name", "")}};
      fn["description"] = t.contains("description") ? t["description"] : json();
      fn["parameters"] = t.contains("parameters") ? t["parameters"] : json();
      fn["strict"] = t.contains("strict") ? t["strict"] : json();
      tools.push_back({{"type", "function"}, {"function", fn}});
    }
  }
  if (!tools.empty() && body.value("tool_choice", json("auto")) != "none") chat["tools"] = tools;
  for (const char* k : {"stream", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                        "repetition_penalty", "seed", "stop", "chat_template_kwargs"})
    if (body.contains(k)) chat[k] = body[k];
  if (body.contains("max_output_tokens") && body["max_output_tokens"].is_number()) chat["max_tokens"] = body["max_output_tokens"];
  if (body.contains("reasoning") && body["reasoning"].is_object() && body["reasoning"].contains("effort") &&
      body["reasoning"]["effort"].is_string()) {
    const std::string effort = body["reasoning"]["effort"];
    if (effort == "none") {
      if (!chat.contains("chat_template_kwargs")) chat["chat_template_kwargs"] = json::object();
      if (!chat["chat_template_kwargs"].contains("enable_thinking")) chat["chat_template_kwargs"]["enable_thinking"] = false;
    } else {
      chat["reasoning_effort"] = effort;
    }
  }
  Request r = parse_request(chat, true);
  r.custom_tools = std::move(custom);
  return r;
}

std::string custom_tool_input(const std::string& arguments) {
  const json args = json::parse(arguments, nullptr, false);
  if (args.is_object()) {
    if (args.contains(kCustomToolInput) && args[kCustomToolInput].is_string()) return args[kCustomToolInput];
    if (args.size() == 1 && args.begin()->is_string()) return args.begin().value();
  }
  return arguments;
}

}  // namespace ling::serve
