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
