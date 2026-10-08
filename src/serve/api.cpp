#include "serve/api.hpp"

#include <random>
#include <stdexcept>

#include "tokenizer/chat_template.hpp"

namespace ling::serve {
namespace {

const std::string kThinkEnd = "</think>";
const std::string kCallOpen = "<tool_call>";
const std::string kCallClose = "</tool_call>";

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

}  // namespace

std::string new_id(const std::string& prefix) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* hex = "0123456789abcdef";
  std::string id = prefix;
  for (int i = 0; i < 24; ++i) id += hex[rng() & 15];
  return id;
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
      r.tools = body["tools"];
    r.prompt_text = render_chat(body["messages"], r.tools, opts);
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
      size_t open = buf_.find(kCallOpen);
      if (open != std::string::npos) {
        d.content += buf_.substr(0, open);
        size_t close = buf_.find(kCallClose, open);
        if (close == std::string::npos) {
          buf_.erase(0, open);
          if (final) {  // an unterminated call: hand it back as text
            d.content += buf_;
            buf_.clear();
          }
          break;
        }
        const std::string block = buf_.substr(open + kCallOpen.size(), close - open - kCallOpen.size());
        buf_.erase(0, close + kCallClose.size());
        if (auto call = parse_call(block)) d.tool_calls.push_back(*call);
        else d.content += kCallOpen + block + kCallClose;
        continue;
      }
      const size_t hold = final ? 0 : partial_tag(buf_, kCallOpen);
      d.content += buf_.substr(0, buf_.size() - hold);
      buf_.erase(0, buf_.size() - hold);
      break;
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
    size_t vend = block.find(pclose, pend);
    if (vend == std::string::npos) vend = block.find("</function>", pend);
    if (vend == std::string::npos) vend = block.size();
    std::string value = block.substr(pend + 1, vend - pend - 1);
    if (!value.empty() && value.front() == '\n') value.erase(0, 1);
    if (!value.empty() && value.back() == '\n') value.pop_back();
    const std::string converted = convert_value(call.name, name, value);
    json parsed = json::parse(converted, nullptr, false);
    args[name] = parsed.is_discarded() ? json(value) : parsed;
    pos = vend + pclose.size();
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

// One Responses input item as a chat message, or null to drop it.
json normalize_item(const json& item) {
  const std::string type = item.value("type", "message");
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
    const std::string id = item.contains("call_id") && item["call_id"].is_string() ? item["call_id"].get<std::string>()
                                                                                    : item.value("id", "");
    return json{{"role", "assistant"},
                {"tool_calls", json::array({json{{"id", id},
                                                 {"type", "function"},
                                                 {"function", {{"name", item.value("name", "")}, {"arguments", args}}}}})}};
  }
  if (type == "function_call_output") {
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
  if (item.contains("content")) msg["content"] = item["content"].is_string() ? item["content"] : json(text_of_parts(item["content"]));
  return msg;
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
      const std::string nc = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : "";
      if (!nc.empty()) {
        const std::string pc = prev.contains("content") && prev["content"].is_string() ? prev["content"].get<std::string>() : "";
        prev["content"] = pc.empty() ? nc : pc + "\n\n" + nc;
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
  // Every system chunk goes into one leading system message.
  std::string system;
  json others = json::array();
  for (json& m : merged) {
    if (m["role"] == "system") {
      const std::string c = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : "";
      if (!c.empty()) system += (system.empty() ? "" : "\n\n") + c;
    } else {
      others.push_back(std::move(m));
    }
  }
  json final_messages = json::array();
  if (!system.empty()) final_messages.push_back({{"role", "system"}, {"content", system}});
  for (json& m : others) final_messages.push_back(std::move(m));

  json chat = json::object();
  chat["messages"] = final_messages;
  json tools = json::array();
  if (body.contains("tools") && body["tools"].is_array()) {
    for (const json& t : body["tools"]) {
      if (t.value("type", "") != "function") continue;
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
  return parse_request(chat, true);
}

}  // namespace ling::serve
