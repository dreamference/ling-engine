"""Simulate a DFlash2 + context-lookup hybrid drafter on recorded Mightling agent sessions.

For each model request in a session's rollout, the visible output (assistant text and tool-call
arguments) is replayed token by token. At every verify step two proposals compete:

- DFlash2: its accepted-draft count is drawn from a truncated geometric distribution with the mean
  production measures on this workload (4.87 tokens per step including the bonus token);
- context lookup: the continuation of the most recent earlier occurrence of the last n tokens in the
  session (context and output so far), up to `--lookup` tokens, accepted while it matches.

The step advances by the larger of the two plus the bonus token. The two sources are drawn
independently, while in reality both succeed on the same copyable spans, so the hybrid figure is an
upper bound. Reads rollouts only; needs the checkpoint's tokenizer.json.

Usage: python3 lookup_sim.py [--sessions 60] [--lookup 32] [--ngram 3] [--seed 0]
"""
import argparse, glob, json, os, random
from tokenizers import Tokenizer

RUNS = os.path.expanduser('~/.local/share/dreamference/swe-bench/runs')
TOKENIZER = os.path.expanduser('~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/tokenizer.json')
DFLASH_MEAN_ACCEPTED = 3.87   # drafts accepted per step, production log (accept length 4.87 incl. bonus)
DFLASH_MAX = 16               # production drafts 16 tokens


def text_of(content):
    return ''.join(c.get('text', '') for c in content if isinstance(c, dict))


def dflash_accepted(rng):
    p = 1 / (1 + DFLASH_MEAN_ACCEPTED)
    a = 0
    while a < DFLASH_MAX and rng.random() > p:
        a += 1
    return a


def simulate(files, tok, policy, n, lookup, rng):
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids if s else []
    steps = tokens = 0
    for f in files:
        ctx, idx = [], {}

        def push(t):
            ctx.append(t)
            if len(ctx) > n:
                idx[tuple(ctx[-n - 1:-1])] = len(ctx) - 1

        pending = []
        for line in open(f):
            r = json.loads(line)
            p = r.get('payload') or {}
            if r['type'] == 'response_item':
                kind = p.get('type')
                if kind == 'message' and p.get('role') == 'assistant':
                    pending.append(text_of(p.get('content', [])))
                elif kind == 'function_call':
                    pending.append(p.get('name', '') + ' ' + p.get('arguments', ''))
                elif kind == 'message':
                    for t in enc(text_of(p.get('content', []))):
                        push(t)
                elif kind in ('function_call_output', 'custom_tool_call_output'):
                    o = p.get('output')
                    for t in enc(o if isinstance(o, str) else json.dumps(o)):
                        push(t)
            elif r['type'] == 'token_usage_record':
                out = [t for s in pending for t in enc(s)]
                i = 0
                while i < len(out):
                    prop = []
                    if len(ctx) >= n:
                        pos = idx.get(tuple(ctx[-n:]))
                        if pos is not None:
                            prop = ctx[pos:pos + lookup]
                    a_l = 0
                    while a_l < len(prop) and i + a_l < len(out) and prop[a_l] == out[i + a_l]:
                        a_l += 1
                    a_d = dflash_accepted(rng)
                    a = a_d if policy == 'dflash' else max(a_d, a_l)
                    a = min(a, len(out) - i - 1)
                    for t in out[i:i + a + 1]:
                        push(t)
                    i += a + 1
                    steps += 1
                    tokens += a + 1
                pending = []
    return tokens / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, default=60)
    ap.add_argument('--lookup', type=int, default=32)
    ap.add_argument('--ngram', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    tok = Tokenizer.from_file(glob.glob(TOKENIZER)[0])
    files = sorted(glob.glob(f'{RUNS}/im-*/scratch/*/codex-home/sessions/*/*/*/rollout-*.jsonl'))
    random.Random(a.seed).shuffle(files)
    files = files[:a.sessions]
    print('sessions', len(files))
    for policy in ('dflash', 'hybrid'):
        rng = random.Random(a.seed)
        print(f'{policy:7s} tokens per step', round(simulate(files, tok, policy, a.ngram, a.lookup, rng), 2))


if __name__ == '__main__':
    main()
