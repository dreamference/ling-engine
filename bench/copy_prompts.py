#!/usr/bin/env python3
"""Copy-heavy prompts for the copy proposer's exactness checks (moonshot P1).

Writes one JSON string per line, each a raw chat prompt as `ling-run --prompts-file` takes it (the format of the
M3/M4 check sets: a user turn, then the assistant turn with an empty think block). Each prompt shows a source file
of this repository and asks for it back with a small edit, so the greedy answer is mostly long verbatim copies of
the context: what `--spec-check --copy 1|2` and `--graph-check --copy 1|2` need to exercise copy steps. The files
are this repository's own; nothing recorded leaves the machine.

    python3 bench/copy_prompts.py > /tmp/copy.jsonl     # 6 prompts, files of 1-10 KB (answers of ~0.3-3K tokens)
    python3 bench/copy_prompts.py --count 2 > ...       # fewer
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (file, the edit asked for). The edits are small, so the answer is the file nearly verbatim.
TASKS = [
    ("src/core/lookup.cpp", "Rename the function lookup_propose to propose_lookup everywhere"),
    ("src/core/accept.cpp", "Add the line `// Reviewed.` as the first line"),
    ("src/core/sampling.cpp", "Rename the parameter `p` of sample_logits to `params` everywhere in that function"),
    ("src/core/copy_proposer.cpp", "Remove the function extend_with_copy"),
    ("src/core/drafter.cpp", "Add the line `// Reviewed.` as the first line"),
    ("src/tokenizer/chat_template.cpp", "Add the line `// Reviewed.` as the first line"),
]


def prompt(path: str, edit: str) -> str:
    body = (ROOT / path).read_text()
    user = (f"Here is the file `{path}`:\n\n```cpp\n{body}```\n\n{edit}. Reply with the complete new file in one "
            "code block and nothing else.")
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-bytes", type=int, default=16000, help="skip files larger than this")
    ap.add_argument("--count", type=int, default=len(TASKS), help="at most this many prompts")
    args = ap.parse_args()
    n = 0
    for path, edit in TASKS:
        if n >= args.count:
            break
        if (ROOT / path).stat().st_size > args.max_bytes:
            continue
        print(json.dumps(prompt(path, edit)))
        n += 1


if __name__ == "__main__":
    main()
