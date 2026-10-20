"""One JSON string per model response in Codex rollouts: its assistant message text and its tool calls' arguments, in
order (a response ends at the next tool output or user message). For rep_stats.py, through ling-tokenize."""
import json, sys

for path in sys.argv[1:]:
    cur = []
    def flush():
        if cur:
            print(json.dumps("".join(cur)))
            cur.clear()
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") != "response_item":
            continue
        p = d.get("payload", {})
        t = p.get("type")
        if t == "message" and p.get("role") == "assistant":
            for c in p.get("content", []):
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    cur.append(c["text"])
        elif t in ("function_call", "custom_tool_call"):
            a = p.get("arguments", p.get("input", ""))
            if isinstance(a, str):
                # the model wrote the arguments as JSON text: decode string values so a patch's lines are its own
                try:
                    v = json.loads(a)
                    a = "\n".join(str(x) for x in v.values()) if isinstance(v, dict) else a
                except ValueError:
                    pass
                cur.append(a)
        elif t == "reasoning":
            for c in p.get("content", []) or []:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    cur.append(c["text"])
        else:
            flush()
    flush()
