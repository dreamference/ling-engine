#!/usr/bin/env python3
"""Copy-heavy prompts for the copy proposer's exactness checks (moonshot P1).

Writes one JSON string per line, each a raw chat prompt as `ling-run --prompts-file` takes it (the format of the
M3/M4 check sets, ending in the assistant turn with an empty think block). Each prompt shows a source file of this
repository and asks for a small edit, so the greedy answer is mostly long copies of the context: what
`--spec-check --copy 1|2` and `--graph-check` need to exercise copy steps. Every other prompt shows the file as a
tool output (a `cat` tool call and its <tool_response>, as the chat template renders an agent's read), which the
emission views (`--copy-views`) and edit mode (`--copy-edit`) work on; those ask for an apply_patch patch, whose
`@@` hunk headers and ' '/'-'/'+' lines are what edit mode and the diff views copy. The files are this repository's
own; nothing recorded leaves the machine.

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
    ("src/core/accept.cpp", "Add the line `// Reviewed.` before each function"),
    ("src/core/sampling.cpp", "Rename the parameter `p` of sample_logits to `params` everywhere in that function"),
    ("src/core/copy_proposer.cpp", "Remove the function extend_with_copy"),
    ("src/core/drafter.cpp", "Add the line `// Reviewed.` as the first line"),
    ("src/tokenizer/chat_template.cpp", "Add the line `// Reviewed.` before each function"),
]

TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def inline_prompt(path: str, body: str, edit: str) -> str:
    user = (f"Here is the file `{path}`:\n\n```cpp\n{body}```\n\n{edit}. Reply with the complete new file in one "
            "code block and nothing else.")
    return f"<|im_start|>user\n{user}<|im_end|>\n{TAIL}"


def tool_prompt(path: str, body: str, edit: str) -> str:
    # The read as an agent makes it: a tool call, then its output in a <tool_response> (the template's form).
    call = ("<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat " + path +
            "\n</parameter>\n</function>\n</tool_call>")
    ask = (f"In `{path}`: {edit}. Read the file first.")
    then = ("Now write the change as one apply_patch patch: `*** Begin Patch`, `*** Update File: " + path +
            "`, then hunks that each start with an `@@` line naming a line of the file and carry three lines of "
            "unchanged context (prefixed with a space) around the `-` and `+` lines, then `*** End Patch`. Nothing else.")
    return (f"<|im_start|>user\n{ask}<|im_end|>\n<|im_start|>assistant\n{call}<|im_end|>\n"
            f"<|im_start|>user\n<tool_response>\n{body}\n</tool_response>\n{then}<|im_end|>\n{TAIL}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-bytes", type=int, default=16000, help="skip files larger than this")
    ap.add_argument("--count", type=int, default=len(TASKS), help="at most this many prompts")
    args = ap.parse_args()
    n = 0
    for i, (path, edit) in enumerate(TASKS):
        if n >= args.count:
            break
        if (ROOT / path).stat().st_size > args.max_bytes:
            continue
        body = (ROOT / path).read_text()
        print(json.dumps(tool_prompt(path, body, edit) if i % 2 else inline_prompt(path, body, edit)))
        n += 1


if __name__ == "__main__":
    main()
