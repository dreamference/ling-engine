# Code Mode acceptance: custom tools end to end, 2026-10-10

The first live use of the custom (freeform) tool path ([API §4](../specs/DREAMFERENCE_LING_ENGINE_API.md), [BACKLOG §19.13](../specs/DREAMFERENCE_LING_ENGINE_BACKLOG.md), commit d688273). The operations stream ran the same task twice with the Mightling agent `ling exec` in Code Mode, through a recording proxy with stdin closed: once against `ling-serve` d688273 on second-puffin (port 18080, drafter on, 12 draft tokens), once against the frozen M0 reference container on the same machine (production SGLang, port 8000). The task: fix `calc.py` in a scratch workspace.

**Result.** Against ling-serve the model called the custom `exec` tool three times, each answered as a `custom_tool_call` item with its `custom_tool_call_input` delta, and finished in seven turns and 21 s with the file fixed. Against production the same requests, carrying the same custom `exec` tool, never produced a custom call in nine turns: production's conversion keeps function tools only, so the model never saw `exec` and worked through `exec_command` and `write_stdin` instead.

## The requests

Every request offered 34 tools, one of them `{"type": "custom", "name": "exec", "format": {"type": "grammar", "syntax": "lark", …}}` (a 177-byte grammar), the rest function tools; `reasoning: {"effort": "none", "summary": "auto"}`; the launcher's model name. The first request was 16,277 tokens on ling-serve and 17,569 on production (production's own template, with the thinking block as its template renders it).

## Turn by turn

ling-serve (d688273), items in each `response.completed`:

| Turn | Output items | Input tokens | Cached | Output tokens |
|---|---|---|---|---|
| 1 | `custom_tool_call` exec | 16,277 | 0 | 43 |
| 2 | `function_call` exec_command | 16,353 | 16,256 | 28 |
| 3 | `custom_tool_call` exec | 16,448 | 16,352 | 96 |
| 4 | `function_call` exec_command | 16,584 | 16,448 | 65 |
| 5 | `custom_tool_call` exec | 16,844 | 16,576 | 61 |
| 6 | `function_call` exec_command | 16,939 | 16,832 | 75 |
| 7 | `message` | 17,086 | 16,928 | 144 |

Production (M0 reference):

| Turn | Output items | Input tokens | Cached | Output tokens |
|---|---|---|---|---|
| 1 | `function_call` exec_command | 17,569 | 1,792 | 29 |
| 2 | `function_call` exec_command | 17,667 | 17,536 | 44 |
| 3 | `function_call` exec_command | 18,021 | 17,664 | 313 |
| 4 | `message`, `function_call` exec_command | 18,687 | 18,176 | 193 |
| 5 | `function_call` exec_command | 18,957 | 18,688 | 124 |
| 6 | `function_call` exec_command | 19,143 | 18,944 | 65 |
| 7 | `function_call` write_stdin | 19,373 | 19,200 | 42 |
| 8 | `message`, `function_call` exec_command | 19,461 | 19,328 | 115 |
| 9 | `message` | 19,651 | 19,456 | 116 |

Totals: ling-serve 116,531 input tokens of which 99,392 cached, 512 output tokens over 7 turns; production 168,529 input, 150,784 cached, 1,041 output over 9 turns. The two runs are not a speed comparison (different tool paths, one task, one run each); they show that the custom path works and that the prefix cache carried every turn after the first on both servers.

## What the agent did with the calls

The agent alternated `exec` (the Code Mode sandbox) with `exec_command` (the host shell): it noted that the Code Mode sandbox has no filesystem access and made the edit through `exec_command`. That is Codex's host side, not the server; it is on the operations stream's list, not this engine's.

## Where the recordings are

The raw request and response bodies (18 files for ling-serve, 22 for production) are kept outside the repository, in the operations stream's hand-off directory under `~/.local/share/dreamference/handoff/code-mode-acceptance-2026-10-10/`. They contain the agent's full instructions, the workspace paths and an account address, so they stay untracked; this page carries the counts read from them (`output` of each `response.completed`, `usage`, the tool list of the first request).
