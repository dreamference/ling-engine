#!/usr/bin/env python3
"""Compares ling-tokenize with the reference tokenizer (Hugging Face `tokenizers`) token for token.

The corpus is built in-process and covers:
- English prose and code in several languages;
- CJK, Arabic, Hindi, Russian and emoji;
- every whitespace shape the pre-tokenizer pattern distinguishes;
- contractions;
- special and added tokens inside text;
- every file of a source tree given with --tree.

    tokenizer_test.py --ling-tokenize build/ling-tokenize --tokenizer DIR/tokenizer.json [--tree PATH]
"""
import argparse
import json
import pathlib
import subprocess
import sys

from tokenizers import Tokenizer

BASE = [
    "Hello, world!", "  leading spaces", "trailing spaces   ", "tabs\tand\nnewlines\r\n\r\nend",
    "I'm sure it's fine, they've said we'll go and you'd agree. DON'T SHOUT", "1234567890 3.14159 1e-9 0x1F",
    "def f(x):\n    return x ** 2  # square\n\n\nprint(f(3))\n",
    "#include <vector>\nint main() { std::vector<int> v{1,2,3}; return v.size(); }\n",
    "fn main() { let s = String::from(\"héllo\"); println!(\"{s}\"); }",
    "SELECT name, COUNT(*) FROM users WHERE age > 30 GROUP BY name;",
    "这是一个测试句子，包含中文标点。", "日本語のテキストとカタカナ、ひらがな。", "한국어 문장입니다.",
    "مرحبا بالعالم، هذا اختبار.", "नमस्ते दुनिया, यह एक परीक्षण है।", "Привет, мир! Ёлка и щука.",
    "Emoji: 😀👍🏽👨‍👩‍👧‍👦 🇺🇸 ✅", "Café naïve résumé — “quotes” ‘single’ …",
    "é (decomposed) vs é (composed)", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>",
    "text<|endoftext|>more<think>inner</think>", "   \n\n   \t\t \n", "a" * 300, "ab " * 200,
    "!!!???...,,,;;;:::", "URL https://example.com/path?q=1&r=two#frag and email a.b@c.org",
    "mixed中文English123数字", "​ zero width ﻿ bom", "line1\n    indented\n        more\n",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ling-tokenize", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tree", action="append", default=[])
    args = ap.parse_args()

    corpus = list(BASE)
    for tree in args.tree:
        for p in sorted(pathlib.Path(tree).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".rs", ".md", ".cpp", ".hpp", ".cu", ".json", ".toml", ".sh", ".txt"}:
                try:
                    text = p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if 0 < len(text) < 200_000:
                    corpus.append(text)
    ref = Tokenizer.from_file(args.tokenizer)
    inp = "\n".join(json.dumps(t, ensure_ascii=False) for t in corpus) + "\n"
    out = subprocess.run([args.ling_tokenize, args.tokenizer], input=inp, capture_output=True, text=True, check=True)
    ours = [json.loads(line) for line in out.stdout.splitlines()]
    assert len(ours) == len(corpus), (len(ours), len(corpus))
    bad = 0
    tokens = 0
    for text, got in zip(corpus, ours):
        want = ref.encode(text, add_special_tokens=False).ids
        tokens += len(want)
        if got != want:
            bad += 1
            if bad <= 5:
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
                print(f"MISMATCH at token {i}: {text[:60]!r}\n  ref  {want[max(0, i - 3):i + 5]}\n  ours {got[max(0, i - 3):i + 5]}")
    print(f"{len(corpus) - bad}/{len(corpus)} texts identical ({tokens} reference tokens)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
