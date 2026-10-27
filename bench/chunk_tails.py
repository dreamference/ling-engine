"""Text prompts for --chunk-check whose last chunk of 256 is short: one passage padded by 0, 8, ..., 248 words.

Every residue of the token count modulo 256 is hit at a step of 8, so about four of the 32 prompts end in a chunk of at
most 32 rows (the rows path's size) when prefilled in chunks of 256, and in one long chunk when prefilled whole.

    python3 tails.py SOURCE.md > prompts-tails.jsonl
"""

import json
import sys

text = open(sys.argv[1], encoding="utf-8").read()[:4000]
for k in range(0, 256, 8):
    body = "Summarise the following document in five bullet points.\n\n" + text + "\n\nNotes:" + " ok" * k
    print(json.dumps(f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
