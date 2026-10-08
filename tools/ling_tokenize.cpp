// ling-tokenize TOKENIZER_JSON < lines: reads JSON strings, one per line, and prints each one's token
// ids as a JSON array on its own line (the tokenizer test compares them with the reference tokenizer).
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: ling-tokenize tokenizer.json < strings.jsonl\n";
    return 2;
  }
  try {
    ling::Tokenizer tok(argv[1]);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::string text = nlohmann::json::parse(line).get<std::string>();
      std::cout << nlohmann::json(tok.encode(text)).dump() << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
