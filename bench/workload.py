"""Profile Mightling agent sessions from SWE-bench rollouts: tokens, latency, cache hits, copyable output.

Reads rollouts only. Usage: python3 workload.py [run names...]
"""
import json, glob, os, sys, statistics as st, collections, datetime
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/tokenizer.json'))[0])
RUNS = sys.argv[1:] or ['im-refine','im-index-on','im-index-off','im-index-off-b','im-scip-only','im-test-first']
base = os.path.expanduser('~/.local/share/dreamference/swe-bench/runs')
def ts(s): return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def enc(s): return TOK.encode(s, add_special_tokens=False).ids if s else []

reqs = []          # per request dicts
sessions = []      # per session (wall, model time)
global_out = collections.defaultdict(list)   # cross-session 4-gram -> continuation start (session outputs)
global_seqs = []
def lookup_sim(out, ctx_index, ctx, n=3, k=16, glob_index=None):
    """Greedy prompt-lookup: per verify step propose up to k tokens following the most recent
    earlier occurrence of the last n tokens (context + generated so far). Returns steps, copied."""
    seq = ctx  # list, we append outputs as generated
    i = 0; steps = 0; copied = 0; L = len(out)
    hist = list(seq)
    idx = ctx_index
    while i < L:
        prop = []
        if len(hist) >= n:
            key = tuple(hist[-n:]); pos = idx.get(key)
            if pos is not None:
                prop = hist[pos:pos+k]
        if not prop and glob_index is not None and len(hist) >= 4:
            g = glob_index.get(tuple(hist[-4:]))
            if g: s_, p_ = g; prop = global_seqs[s_][p_:p_+k]
        a = 0
        while a < len(prop) and i + a < L and prop[a] == out[i + a]: a += 1
        step = a + 1 if i + a < L else a  # accepted + bonus token
        step = max(step, 1)
        for t in out[i:i+step]:
            hist.append(t)
            if len(hist) > n: idx[tuple(hist[-n-1:-1])] = len(hist) - 1
        copied += a; i += step; steps += 1
    return steps, copied

for run in RUNS:
    for f in glob.glob(f'{base}/{run}/scratch/*/codex-home/sessions/*/*/*/rollout-*.jsonl'):
        rows = [json.loads(l) for l in open(f)]
        ctx = []; idx = {}
        def add_ctx(text):
            for t in enc(text):
                ctx.append(t)
                if len(ctx) > 3: idx[tuple(ctx[-4:-1])] = len(ctx) - 1
        pending = []; last_t = None; t0 = None; t_end = None; model_s = 0.0
        for r in rows:
            p = r.get('payload', {}) or {}; ty = r['type']; t = ts(r['timestamp'])
            t0 = t0 or t; t_end = t
            if ty == 'response_item':
                pt = p.get('type')
                if pt == 'message':
                    text = ''.join(c.get('text','') for c in p.get('content',[]) if isinstance(c,dict))
                    if p.get('role') == 'assistant': pending.append(('msg', text))
                    else: add_ctx(text); last_t = t
                elif pt == 'function_call':
                    pending.append(('call', p.get('name','') + ' ' + p.get('arguments','')))
                elif pt in ('function_call_output', 'custom_tool_call_output'):
                    o = p.get('output'); o = o if isinstance(o, str) else json.dumps(o)
                    add_ctx(o); last_t = t
                elif pt == 'custom_tool_call':
                    pending.append(('call', p.get('name','') + ' ' + str(p.get('input',''))))
            elif ty == 'token_usage_record':
                u = p['usage']; vis = []
                kinds = collections.Counter()
                for kind, text in pending:
                    e = enc(text); vis += e; kinds[kind] += len(e)
                lat = (t - last_t) if last_t else None
                if lat is not None and lat < 3600: model_s += lat
                steps, copied = lookup_sim(vis, dict(idx), list(ctx)) if vis else (0, 0)
                reqs.append(dict(run=run, inp=u['input_tokens'], cached=u['cached_input_tokens'],
                                 out=u['output_tokens'], vis=len(vis), msg=kinds['msg'], call=kinds['call'],
                                 lat=lat, lk_steps=steps, lk_copied=copied))
                for kind, text in pending: add_ctx(text)
                pending = []; last_t = t
        if t0 and t_end: sessions.append((run, t_end - t0, model_s))

def q(v, p): v = sorted(v); return v[int(p * (len(v) - 1))]
print('requests', len(reqs), 'sessions', len(sessions))
O = [r['out'] for r in reqs]; V = [r['vis'] for r in reqs]
print('output tokens/request: median', q(O,.5), 'mean', round(st.mean(O)), 'p90', q(O,.9), 'total', sum(O))
print('visible tokens/request: median', q(V,.5), 'mean', round(st.mean(V)), ' visible share of output', round(sum(V)/sum(O),3))
print('  of visible: assistant text', round(sum(r['msg'] for r in reqs)/sum(V),3), 'tool calls', round(sum(r['call'] for r in reqs)/sum(V),3))
new = [r['inp'] - r['cached'] for r in reqs]
print('input/request median', q([r['inp'] for r in reqs],.5), 'new (uncached) median', q(new,.5), 'mean', round(st.mean(new)), 'p90', q(new,.9))
print('cache hit share of input', round(sum(r['cached'] for r in reqs)/sum(r['inp'] for r in reqs),3))
L = [r['lat'] for r in reqs if r['lat'] is not None]
print('request latency s: median', round(q(L,.5),2), 'mean', round(st.mean(L),2), 'p90', round(q(L,.9),2))
wall = sum(s[1] for s in sessions); model = sum(s[2] for s in sessions)
print('session wall h', round(wall/3600,1), 'model h', round(model/3600,1), 'model share', round(model/wall,3))
vs = sum(r['vis'] for r in reqs); ls = sum(r['lk_steps'] for r in reqs); lc = sum(r['lk_copied'] for r in reqs)
print('prompt lookup on visible output: tokens/step', round(vs/ls,2), 'copied share', round(lc/vs,3))
# per kind by bucket of output size
for lo, hi in [(0,64),(64,256),(256,1024),(1024,10**9)]:
    sel = [r for r in reqs if lo <= r['out'] < hi]
    if sel: print(f'out {lo}-{hi}: n={len(sel)} share_of_output={sum(r["out"] for r in sel)/sum(O):.2f} visible_share={sum(r["vis"] for r in sel)/max(1,sum(r["out"] for r in sel)):.2f}')
json.dump(reqs, open('reqs.json', 'w'))
