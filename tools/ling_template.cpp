// ling-template < cases.jsonl: renders each {"messages", "tools", "kwargs"} line with the C++ chat template
// and prints the result as a JSON string (bench/template_test.py compares it with the Jinja original).
//
// ling-template --responses TOKENIZER_JSON < bodies.jsonl: takes each line's "body" (a Responses API
// request, as bench/replay/replay.py --dump-bodies writes it), converts it as ling-serve does and prints
// {"ids": [...], "text": "..."}: the prompt ling-serve would prefill (bench/render/render_test.py compares
// it with production's).
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "serve/api.hpp"
#include "tokenizer/chat_template.hpp"
#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
  using json = nlohmann::ordered_json;
  std::string line;
  if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--responses") {
    // As ling-serve tokenizes by default (production's pre-tokenizer); a fourth argument "checkpoint"
    // uses tokenizer.json's own pattern.
    const bool checkpoint = argc == 4 && std::string(argv[3]) == "checkpoint";
    ling::Tokenizer tok(argv[2], checkpoint ? "" : ling::kProductionPretokenizer);
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const json rec = json::parse(line);
      try {
        const ling::serve::Request r = ling::serve::parse_responses_request(rec.contains("body") ? rec["body"] : rec);
        std::cout << json{{"ids", tok.encode(r.prompt_text)}, {"text", r.prompt_text}}.dump() << "\n";
      } catch (const std::exception& e) {
        std::cout << json{{"error", e.what()}}.dump() << "\n";
      }
    }
    return 0;
  }
  if (argc != 1) {
    std::cerr << "usage: ling-template < cases.jsonl | ling-template --responses tokenizer.json < bodies.jsonl\n";
    return 2;
  }
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    json c = json::parse(line);
    ling::ChatOptions o;
    const json kw = c.value("kwargs", json::object());
    if (kw.contains("enable_thinking")) o.enable_thinking = kw["enable_thinking"].get<bool>();
    if (kw.contains("preserve_thinking")) o.preserve_thinking = kw["preserve_thinking"].get<bool>();
    if (kw.contains("reasoning_effort")) o.reasoning_effort = kw["reasoning_effort"].get<std::string>();
    try {
      std::cout << json(ling::render_chat(c["messages"], c.value("tools", json()), o)).dump() << "\n";
    } catch (const std::exception& e) {
      std::cout << json{{"error", e.what()}}.dump() << "\n";
    }
  }
  return 0;
}
