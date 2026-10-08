// ling-template < cases.jsonl: renders each {"messages", "tools", "kwargs"} line with the C++ chat template
// and prints the result as a JSON string (bench/template_test.py compares it with the Jinja original).
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "tokenizer/chat_template.hpp"

int main() {
  using json = nlohmann::ordered_json;
  std::string line;
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
