#!/usr/bin/env python3
"""Moonshot P0, the statistics half: how much of the agent's output could be proposed as long copies.

Reads recorded Mightling agent sessions (the SWE-bench runs' Codex rollouts, read only) and, for every
token the model wrote, measures the longest verbatim run that each proposal source of the moonshot
(specs/drafts/MOONSHOT-repository-aware-decoding-2026-10-13.md, section 3.1) could have handed to the
verify at that position:

  S1   the session's own token history before that token (system prompt, every earlier turn and tool
       output, and the output written so far), rendered as the chat template renders it;
  S2   emission views of every tool output in that history, each tokenized as a whole span:
       `num`  the output with line-number prefixes removed (`cat -n`, `grep -n`, `nl`, the code index's
              `code_show`), which is the form the agent re-emits a file's lines in;
       `diff` every line prefixed with ' ' and with '-' (unified diff / apply_patch context and removals);
       `json` the text JSON-escaped (a function-tool argument in JSON form);
  S3p  a proxy for the repository corpus: tool outputs that *other* sessions of the same task (other
       runs, same base commit) saw before their first edit, raw and in the S2 views;
  S4   other sessions' outputs, leave-one-task-out (every session of the same instance excluded),
       over the whole corpus and partitioned by repository.

Matching is exact on token ids. For S1-S3 a run is the longest extension from the most recent or the
first earlier occurrence of the next 16 tokens (an oracle start: the proposer is told where a run
begins); runs shorter than 16 tokens are not counted. For S1+S2 a realistic proposer is also simulated:
it anchors on the last 16 tokens already written, follows the most recent earlier occurrence and
reports how many proposed tokens would have been accepted (no knowledge of the future).

Phases (each writes into --work, outside the repository; only aggregates leave it):
  tokenize   parse and render every finished session, tokenize with the model's tokenizer
  match      S1, S2, S3p and the realistic proposer, per session (worker pool)
  global     S4 (one process, 16-gram index over every session's output)
  report     aggregate: run-length shares per output class, the tokens-per-step model; JSON to stdout

Memory: every worker sets RLIMIT_AS (--mem-gb); files are streamed line by line.

Usage:
  nice -n 19 python3 copy_stats.py --work DIR all [--workers 3] [--mem-gb 4] > result.json
"""
import argparse
import collections
import glob
import itertools
import json
import math
import os
import pickle
import re
import resource
import sys
from array import array
from multiprocessing import Pool

RUNS = os.path.expanduser('~/.local/share/dreamference/swe-bench/runs')
TOKENIZER = os.path.expanduser(
    '~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/tokenizer.json')
FINISHED = {'done', 'timeout', 'empty', 'interrupted', 'error'}   # 'running' sessions are skipped
N = 16           # n-gram that starts (oracle) or anchors (realistic) a copy; runs below N are not counted
LCAP = 4096      # longest run recorded
ACAP = 128       # realistic proposer: longest proposal scored
MCAP = 64        # realistic proposer: backward match measured up to this
SEP = -1         # document separator inside view streams
CLASSES = ['prose', 'call_syntax', 'edit_patch', 'edit_script', 'file_write', 'shell', 'tool_other']
C = {name: i for i, name in enumerate(CLASSES)}
SOURCES = ['raw', 'num', 'diff', 'json', 's3p', 's4repo', 's4all']

GEN_PROMPT = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
NUM_PREFIX = re.compile(r'^(?:[^\s:]+[:-])?\s*\d+(?:\t|:|-|  )')

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('RAYON_NUM_THREADS', '1')
        from tokenizers import Tokenizer
        paths = sorted(glob.glob(TOKENIZER))
        if not paths:
            sys.exit('tokenizer.json of the Qwen3.8 checkpoint not found in the HF cache')
        _tok = Tokenizer.from_file(paths[-1])
    return _tok


def encode(text):
    return array('i', tokenizer().encode(text, add_special_tokens=False).ids) if text else array('i')


def limit_memory(gb):
    if gb:
        b = int(gb * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (b, b))


# --------------------------------------------------------------------------------------------------
# Discovery and parsing

def discover(runs_dir, skip_runs):
    """Every finished session: (run, instance, rollout path). Sessions still running are skipped."""
    out, skipped = [], collections.Counter()
    for run in sorted(os.listdir(runs_dir)):
        if run.startswith('smoke') or run in skip_runs:
            continue
        for jf in sorted(glob.glob(f'{runs_dir}/{run}/instances/*.json')):
            try:
                meta = json.load(open(jf))
            except (OSError, ValueError):
                skipped['unreadable'] += 1
                continue
            iid = meta.get('instance_id') or os.path.basename(jf)[:-5]
            if meta.get('status') not in FINISHED:
                skipped[str(meta.get('status'))] += 1
                continue
            pat = f'{runs_dir}/{run}/scratch/{iid}/codex-home/sessions/**/rollout-*.jsonl'
            for f in sorted(glob.glob(pat, recursive=True)):
                out.append((run, iid, f))
    return out, skipped


def text_of(content):
    if isinstance(content, str):
        return content
    return ''.join(c.get('text', '') for c in content or [] if isinstance(c, dict))


def classify_cmd(cmd):
    """The output class of an exec_command `cmd` (heuristic; see the report)."""
    if '*** Begin Patch' in cmd:
        return 'edit_patch'
    if re.search(r'\bsed\s+(?:-[a-zA-Z]*i\b|--in-place)', cmd) or re.search(r'\bperl\s+-[a-z]*i', cmd):
        return 'edit_script'
    first = next((ln for ln in cmd.split('\n') if '<<' in ln), '')
    if first and (re.search(r'\bcat\b[^|]*>', first) or re.search(r'\btee\b', first)):
        return 'file_write'
    if ('.write(' in cmd or 'write_text(' in cmd) and ('open(' in cmd or 'Path(' in cmd):
        if '.replace(' in cmd or re.search(r'\bold\w*\s*=', cmd):
            return 'edit_script'
        return 'file_write'
    return 'shell'


def arg_class(name, key, value):
    if name == 'exec_command' and key == 'cmd':
        return classify_cmd(value)
    if name in ('apply_patch',) or (isinstance(value, str) and '*** Begin Patch' in value):
        return 'edit_patch'
    if name in ('edit', 'edit_file', 'str_replace', 'replace'):
        return 'edit_script'
    if name in ('write', 'write_file', 'create_file'):
        return 'file_write'
    return 'tool_other'


def render_output(texts, calls):
    """The response as the model emitted it under the checkpoint's chat template: content, then each
    call as <tool_call><function=…><parameter=…>value</parameter>… with string values verbatim.
    Returns [(text, class)]."""
    parts = []
    content = '\n\n'.join(t for t in texts if t).strip()
    if content:
        parts.append((content, 'prose'))
    for k, (name, args) in enumerate(calls):
        if k == 0:
            head = ('\n\n' if content else '') + '<tool_call>\n<function=' + name + '>\n'
        else:
            head = '\n<tool_call>\n<function=' + name + '>\n'
        parts.append((head, 'call_syntax'))
        for key, val in args.items():
            parts.append(('<parameter=' + key + '>\n', 'call_syntax'))
            sval = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            parts.append((sval, arg_class(name, key, sval) if isinstance(val, str) else 'tool_other'))
            parts.append(('\n</parameter>\n', 'call_syntax'))
        parts.append(('</function>\n</tool_call>', 'call_syntax'))
    parts.append(('<|im_end|>', 'call_syntax'))
    return parts


def tokenize_output(parts):
    text = ''.join(p for p, _ in parts)
    enc = tokenizer().encode(text, add_special_tokens=False)
    bounds, pos = [], 0
    for p, cls in parts:
        pos += len(p)
        bounds.append((pos, C[cls]))
    ids, cls = array('i', enc.ids), array('B')
    j = 0
    for (start, _end) in enc.offsets:
        while j < len(bounds) - 1 and start >= bounds[j][0]:
            j += 1
        cls.append(bounds[j][1])
    return ids, cls


def views_of(text):
    """S2's emission views of one tool output, as texts: (num, diff, json). num is None unless most
    lines carry a line-number prefix."""
    lines = text.split('\n')
    if len(lines) < 3:
        return None, None, None
    nonempty = [ln for ln in lines if ln.strip()]
    num = None
    if nonempty and sum(1 for ln in nonempty if NUM_PREFIX.match(ln)) >= 0.6 * len(nonempty):
        lines = [NUM_PREFIX.sub('', ln, count=1) for ln in lines]
        num = '\n'.join(lines)
    base = '\n'.join(lines)
    diff = '\n'.join(' ' + ln for ln in lines) + '\n' + '\n'.join('-' + ln for ln in lines)
    return num, diff, json.dumps(base, ensure_ascii=False)[1:-1]


def parse_session(path):
    """Stream a rollout into events: ('ctx', text) | ('tool', text) | ('out', parts, recorded_tokens)."""
    events, seg = [], []

    def flush(recorded):
        texts = [x[1] for x in seg if x[0] == 'assistant']
        calls = [(x[1], x[2]) for x in seg if x[0] == 'call']
        first_out = next((k for k, x in enumerate(seg) if x[0] in ('assistant', 'call')), None)
        for k, x in enumerate(seg):
            if x[0] == 'msg' and (first_out is None or k < first_out):
                events.append(('ctx', x[1]))
        if texts or calls:
            events.append(('out', render_output(texts, calls), recorded))
        tools = [x[1] for x in seg if x[0] == 'tool']
        for k, t in enumerate(tools):
            pre = '<|im_start|>user' if k == 0 else ''
            post = '<|im_end|>\n' if k == len(tools) - 1 else ''
            events.append(('tool', t, pre + '\n<tool_response>\n', '\n</tool_response>' + post))
        for k, x in enumerate(seg):
            if x[0] == 'msg' and first_out is not None and k > first_out:
                events.append(('ctx', x[1]))
        seg.clear()

    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t, p = r.get('type'), r.get('payload') or {}
            if t == 'session_meta':
                bi = p.get('base_instructions')
                bi = bi.get('text') if isinstance(bi, dict) else bi
                if bi:
                    seg.append(('msg', '<|im_start|>system\n' + bi + '<|im_end|>\n'))
            elif t == 'compacted':
                if p.get('message'):
                    seg.append(('msg', '<|im_start|>user\n' + p['message'] + '<|im_end|>\n'))
            elif t == 'token_usage_record':
                flush((p.get('usage') or {}).get('output_tokens'))
            elif t == 'response_item':
                kind = p.get('type')
                if kind == 'message':
                    role = p.get('role')
                    txt = text_of(p.get('content'))
                    if role == 'assistant':
                        seg.append(('assistant', txt))
                    else:
                        role = 'system' if role == 'developer' else role
                        seg.append(('msg', f'<|im_start|>{role}\n{txt.strip()}<|im_end|>\n'))
                elif kind == 'function_call':
                    try:
                        args = json.loads(p.get('arguments') or '{}')
                    except ValueError:
                        args = {'arguments': p.get('arguments') or ''}
                    if not isinstance(args, dict):
                        args = {'arguments': json.dumps(args)}
                    seg.append(('call', p.get('name') or '', args))
                elif kind == 'custom_tool_call':
                    seg.append(('call', p.get('name') or '', {'input': p.get('input') or ''}))
                elif kind in ('function_call_output', 'custom_tool_call_output'):
                    o = p.get('output')
                    if isinstance(o, dict):
                        o = o.get('content', o.get('output', json.dumps(o)))
                    seg.append(('tool', text_of(o) if not isinstance(o, str) else o))
    if seg:
        flush(None)
    return events


# --------------------------------------------------------------------------------------------------
# Phase 1: tokenize

def tokenize_job(job):
    run, iid, path, outp, mem = job
    limit_memory(mem)
    if os.path.exists(outp):
        return outp
    events = parse_session(path)
    ev, fidelity, first_edit = [], [], None
    edit_classes = {C['edit_patch'], C['edit_script'], C['file_write']}
    for e in events:
        if e[0] == 'ctx':
            ev.append(('ctx', encode(e[1])))
        elif e[0] == 'tool':
            body = e[1]
            views = [encode(v) if v else None for v in views_of(body)]
            ev.append(('tool', encode(e[2]), encode(body), encode(e[3]), views))
        else:
            ids, cls = tokenize_output(e[1])
            if e[2] is not None:
                fidelity.append(len(ids) - int(e[2]))
            if first_edit is None and any(c in edit_classes for c in cls):
                first_edit = len(ev)
            ev.append(('out', ids, cls))
    data = {'run': run, 'iid': iid, 'events': ev, 'fidelity': fidelity,
            'first_edit': first_edit if first_edit is not None else len(ev)}
    tmp = outp + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, outp)
    return outp


# --------------------------------------------------------------------------------------------------
# Matching structures

class Stream:
    """A token stream with a 16-gram index (most recent and first start of each n-gram). Appended
    n-grams are indexed only once the token after them exists, so a lookup never returns the
    occurrence being written."""

    def __init__(self):
        self.t = array('i')
        self.last = {}
        self.first = {}

    def extend(self, toks, sep=False):
        t = self.t
        if sep:
            t.append(SEP)
        old = len(t)
        t.extend(toks)
        last, first = self.last, self.first
        for q in range(max(old, N), len(t)):
            key = hash(tuple(t[q - N:q]))
            last[key] = q - N
            if key not in first:
                first[key] = q - N

    def append1(self, tok):
        t = self.t
        t.append(tok)
        q = len(t) - 1
        if q >= N:
            key = hash(tuple(t[q - N:q]))
            self.last[key] = q - N
            if key not in self.first:
                self.first[key] = q - N


def extend_match(src, s, out, i, lb, cap):
    n = min(len(src) - s, len(out) - i, cap)
    k = lb
    while k < n and src[s + k] == out[i + k]:
        k += 1
    return k


class Oracle:
    """Longest run starting at each output position, from one stream (oracle start)."""

    def __init__(self, stream):
        self.s = stream
        self.cs, self.cl = None, 0

    def reset(self):
        self.cs, self.cl = None, 0

    def at(self, out, i, key):
        st = self.s
        if self.cs is not None and self.cl - 1 >= N:
            s = self.cs + 1
            L = extend_match(st.t, s, out, i, self.cl - 1, LCAP)
        else:
            L, s = 0, None
            if key is not None:
                for cand in (st.last.get(key), st.first.get(key)):
                    if cand is not None and cand != s:
                        l = extend_match(st.t, cand, out, i, 0, LCAP)
                        if l > L:
                            L, s = l, cand
        self.cs, self.cl = s, L
        return L if L >= N else 0


# --------------------------------------------------------------------------------------------------
# Phase 2: per-session matching (S1, S2, S3p, realistic proposer)

def build_pool(paths, cap_tokens):
    """S3 proxy: tool outputs other sessions of the task saw before their first edit, raw and viewed."""
    pool, n = Stream(), 0
    for p in paths:
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
        for k, e in enumerate(d['events']):
            if k >= d['first_edit']:
                break
            if e[0] != 'tool':
                continue
            for toks in [e[2]] + [v for v in e[4] if v is not None]:
                if n >= cap_tokens:
                    return pool
                pool.extend(toks, sep=True)
                n += len(toks)
    return pool


def match_job(job):
    tokp, pool_paths, outp, mem, pool_cap = job
    limit_memory(mem)
    if os.path.exists(outp):
        return outp
    with open(tokp, 'rb') as fh:
        d = pickle.load(fh)
    raw = Stream()
    views = {'num': Stream(), 'diff': Stream(), 'json': Stream()}
    pool = build_pool(pool_paths, pool_cap) if pool_paths else None
    oracles = {'raw': Oracle(raw), **{k: Oracle(v) for k, v in views.items()}}
    if pool is not None:
        oracles['s3p'] = Oracle(pool)
    gen = encode(GEN_PROMPT)
    nl = encode('\n')
    outs = []
    for e in d['events']:
        if e[0] == 'ctx':
            raw.extend(e[1])
        elif e[0] == 'tool':
            raw.extend(e[1])
            raw.extend(e[2])
            raw.extend(e[3])
            for name, toks in zip(('num', 'diff', 'json'), e[4]):
                if toks is not None and len(toks):
                    views[name].extend(toks, sep=True)
        else:
            out, cls = e[1], e[2]
            raw.extend(gen)
            n = len(out)
            Ls = {k: array('H', bytes(2 * n)) for k in oracles}
            m_arr, a_arr = array('B', bytes(n)), array('B', bytes(n))
            for o in oracles.values():
                o.reset()
            # realistic proposer state: (stream, continuation index, accepted lower bound, m)
            cur = None
            for i in range(n):
                key = hash(tuple(out[i:i + N])) if i + N <= n else None
                for name, o in oracles.items():
                    Ls[name][i] = o.at(out, i, key)
                # realistic: anchor on the last N tokens written (context included)
                rt = raw.t
                if cur is not None and cur[2] >= 1:
                    st, c, a, m = cur
                    c += 1
                    a = extend_match(st.t, c, out, i, a - 1, ACAP)
                    m = min(m + 1, MCAP)
                    cur = (st, c, a, m)
                else:
                    cur = None
                    if len(rt) >= N:
                        akey = hash(tuple(rt[len(rt) - N:]))
                        best = None
                        for st in (raw, views['num'], views['diff'], views['json']):
                            s0 = st.last.get(akey)
                            if s0 is None:
                                continue
                            c = s0 + N
                            if st.t[s0:c] != rt[len(rt) - N:]:
                                continue
                            m = N
                            while m < MCAP and s0 - 1 - (m - N) >= 0 and \
                                    st.t[s0 - 1 - (m - N)] == rt[len(rt) - 1 - m]:
                                m += 1
                            if best is None or m > best[3]:
                                best = (st, c, None, m)
                        if best is not None:
                            st, c, _, m = best
                            cur = (st, c, extend_match(st.t, c, out, i, 0, ACAP), m)
                if cur is not None:
                    m_arr[i], a_arr[i] = cur[3], min(cur[2], 255)
                raw.append1(out[i])
            raw.extend(nl)
            outs.append({'cls': cls, 'L': Ls, 'm': m_arr, 'a': a_arr})
    res = {'run': d['run'], 'iid': d['iid'], 'outs': outs, 'fidelity': d['fidelity'],
           'ctx_tokens': len(raw.t), 'view_tokens': {k: len(v.t) for k, v in views.items()},
           'pool_tokens': len(pool.t) if pool is not None else 0}
    tmp = outp + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, outp)
    return outp


# --------------------------------------------------------------------------------------------------
# Phase 3: S4 (other sessions' outputs, leave-one-task-out)

def repo_of(iid):
    return iid.split('__')[0]


def global_phase(tok_paths, work, mem):
    limit_memory(mem)
    sessions = []   # (iid, concatenated outputs, [(start, len)])
    for p in tok_paths:
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
        cat, spans = array('i'), []
        for e in d['events']:
            if e[0] == 'out':
                if cat:
                    cat.append(SEP)
                spans.append((len(cat), len(e[1])))
                cat.extend(e[1])
        sessions.append((d['iid'], cat, spans))
        del d
    task_ids = {}
    tasks = [task_ids.setdefault(s[0], len(task_ids)) for s in sessions]
    SHIFT = 1 << 24
    results = {}
    for scope in ('all', 'repo'):
        groups = collections.defaultdict(list)
        for k, s in enumerate(sessions):
            groups['*' if scope == 'all' else repo_of(s[0])].append(k)
        for g, members in groups.items():
            first, other = {}, {}
            for k in members:
                cat = sessions[k][1]
                for q in range(len(cat) - N + 1):
                    key = hash(tuple(cat[q:q + N]))
                    v = first.get(key)
                    if v is None:
                        first[key] = k * SHIFT + q
                    elif tasks[v // SHIFT] != tasks[k] and key not in other:
                        other[key] = k * SHIFT + q
            for k in members:
                iid, cat, spans = sessions[k]
                arrs = []
                for (start, n) in spans:
                    out = cat[start:start + n]
                    L = array('H', bytes(2 * n))
                    cs, cl = None, 0
                    for i in range(n):
                        if cs is not None and cl - 1 >= N:
                            src_k, s = cs[0], cs[1] + 1
                            l = extend_match(sessions[src_k][1], s, out, i, cl - 1, LCAP)
                            cs, cl = (src_k, s), l
                        else:
                            cs, cl = None, 0
                            if i + N <= n:
                                key = hash(tuple(out[i:i + N]))
                                v = first.get(key)
                                if v is not None and tasks[v // SHIFT] == tasks[k]:
                                    v = other.get(key)
                                if v is not None:
                                    src_k, s = divmod(v, SHIFT)
                                    l = extend_match(sessions[src_k][1], s, out, i, 0, LCAP)
                                    cs, cl = (src_k, s), l
                        L[i] = cl if cl >= N else 0
                    arrs.append(L)
                results.setdefault(k, {})['s4' + scope] = arrs
            del first, other
    for k, p in enumerate(tok_paths):
        outp = os.path.join(work, 'match', os.path.basename(p))
        with open(outp, 'rb') as fh:
            d = pickle.load(fh)
        for j, o in enumerate(d['outs']):
            for name in ('s4all', 's4repo'):
                o['L'][name] = results[k][name][j]
        tmp = outp + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, outp)
    open(os.path.join(work, 'global.done'), 'w').close()


# --------------------------------------------------------------------------------------------------
# Phase 4: report

COMBOS = {
    'S1': ['raw'],
    'S1+num': ['raw', 'num'],
    'S1+diff': ['raw', 'diff'],
    'S1+json': ['raw', 'json'],
    'S1+S2': ['raw', 'num', 'diff', 'json'],
    'S1+S2+S3p': ['raw', 'num', 'diff', 'json', 's3p'],
    'S4 same repo': ['s4repo'],
    'S4 all': ['s4all'],
    'S1+S2+S4': ['raw', 'num', 'diff', 'json', 's4all'],
    'all': ['raw', 'num', 'diff', 'json', 's3p', 's4all'],
}
THRESH = [16, 32, 64, 128]
KS = [15, 32, 64, 128]
R0 = 5.0                      # DFlash2's measured tokens per step on agent sessions (M0/M1: 4.95-5.24)
R_COPY_M1 = 10.89             # M1 section 6: the drafter on 32+-token verbatim matches, 9.89 + 1
R_BUCKET_M1 = [(32, 10.89), (16, 8.23)]   # M1 section 6 buckets (accepted + bonus), block 16
STEP_COSTS = (0.9, 1.0, 1.2, 1.5)          # a copy step's cost relative to a production step


def r_drafter(m, calibrated):
    if not calibrated:
        return R0
    for lo, r in R_BUCKET_M1:
        if m >= lo:
            return r
    return R0


def group_of(run):
    if run.startswith('im100'):
        return 'im100'
    if re.match(r'n\d', run):
        return 'nights 1-3'
    if run.startswith('im-'):
        return 'im-* (M0 replay population)'
    return 'other (ab, acc, idx)'


def runs_of(Lsparse, n, T, bonus):
    """Greedy left-to-right parse: at a position with a run of at least T, take it (plus the
    mismatch token when bonus) and jump past it. Yields (start, covered)."""
    nxt = 0
    for i in sorted(Lsparse):
        L = Lsparse[i]
        if i < nxt or L < T:
            continue
        c = min(L + 1 if bonus else L, n - i)
        yield i, c
        nxt = i + c


def class_counts(cls, i, c):
    seg = cls[i:i + c].tobytes()
    return [seg.count(k) for k in range(len(CLASSES))]


def model(copy_tokens, copy_steps, ntok):
    """Tokens per step against DFlash2 alone, and the step-cost inflation a copy step may carry
    before the gain in tokens per millisecond falls to +20%. The drafter's rate on the copied tokens
    is R0 (naive), M1's 16-31 bucket, or M1's 32+ bucket (the conservative reading)."""
    B = ntok / R0
    out = {}
    for label, r in (('naive', R0), ('m1_16', R_BUCKET_M1[1][1]), ('m1', R_COPY_M1)):
        saved = copy_tokens / r
        new = B - saved + copy_steps
        out[label] = {'tokens_per_step': ntok / new if new > 0 else None,
                      'gain': B / new if new > 0 else None,
                      'gain_at_c': {c: B / (B - saved + copy_steps * c) for c in STEP_COSTS},
                      'breakeven_step_cost_for_20pct':
                          (B / 1.2 - B + saved) / copy_steps if copy_steps else None}
    return out


def report(work, groups_filter=None):
    paths = sorted(glob.glob(os.path.join(work, 'match', '*.pkl')))
    nc = len(CLASSES)
    tot = [0] * nc
    cov = {cb: {T: [0] * nc for T in THRESH} for cb in COMBOS}
    # per combo, per K: per class [tokens covered by copy steps, copy steps]
    orc = {cb: {K: [[0, 0] for _ in range(nc)] for K in KS} for cb in COMBOS}
    real = {(mm, K, gated): {'tokens': [0] * nc, 'steps': [0] * nc, 'saved_naive': [0.0] * nc,
                             'saved_m1': [0.0] * nc, 'zero': 0}
            for mm in (16, 32) for K in KS for gated in (False, True)}
    groups = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # n, cov16, cov64, tok, steps
    fidelity, sessions, outputs, runs = [], 0, 0, collections.Counter()
    hist = collections.Counter()
    for p in paths:
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
        g = group_of(d['run'])
        if groups_filter and g not in groups_filter:
            continue
        sessions += 1
        runs[d['run']] += 1
        fidelity.extend(d['fidelity'])
        for o in d['outs']:
            outputs += 1
            cls, Ls = o['cls'], o['L']
            n = len(cls)
            for k, v in enumerate(class_counts(cls, 0, n)):
                tot[k] += v
            groups[g][0] += n
            sparse = {s: {i: arr[i] for i in itertools.compress(range(n), arr)}
                      for s, arr in Ls.items()}
            for cb, srcs in COMBOS.items():
                Lc = {}
                for s in srcs:
                    for i, L in sparse.get(s, {}).items():
                        if L > Lc.get(i, 0):
                            Lc[i] = L
                for T in THRESH:
                    for i, c in runs_of(Lc, n, T, False):
                        for k, v in enumerate(class_counts(cls, i, c)):
                            cov[cb][T][k] += v
                        if cb == 'S1+S2':
                            if T == 16:
                                hist[1 << int(math.log2(c))] += c
                                groups[g][1] += c
                            elif T == 64:
                                groups[g][2] += c
                for i, c in runs_of(Lc, n, N, True):
                    counts = class_counts(cls, i, c)
                    for K in KS:
                        steps = math.ceil(c / (K + 1))
                        acc = orc[cb][K]
                        # a step that spans classes is shared in proportion to its tokens
                        for k, v in enumerate(counts):
                            if v:
                                acc[k][0] += v
                                acc[k][1] += steps * v / c
                        if cb == 'S1+S2' and K == 64:
                            groups[g][3] += c
                            groups[g][4] += steps
            # realistic proposer over S1+S2: anchored on the last N tokens already written
            m_arr, a_arr = o['m'], o['a']
            anchored = list(itertools.compress(range(n), m_arr))
            for (mm, K, gated), acc in real.items():
                nxt = 0
                for i in anchored:
                    m = m_arr[i]
                    if i < nxt or m < mm:
                        continue
                    a = min(a_arr[i], K)
                    c = min(a + 1, n - i)
                    if gated and c < r_drafter(m, True):
                        continue   # oracle gate: the drafter's block would have done better here
                    k0 = cls[i]
                    acc['tokens'][k0] += c
                    acc['steps'][k0] += 1
                    acc['saved_naive'][k0] += c / R0
                    acc['saved_m1'][k0] += c / r_drafter(m, True)
                    acc['zero'] += a == 0
                    nxt = i + c
        del d
    total = sum(tot)
    res = {'sessions': sessions, 'outputs': outputs, 'runs': dict(runs), 'output_tokens': total,
           'class_tokens': {CLASSES[k]: tot[k] for k in range(nc)},
           'fidelity': fidelity_summary(fidelity),
           'coverage': {cb: {T: {CLASSES[k]: cov[cb][T][k] for k in range(nc)} for T in THRESH}
                        for cb in COMBOS},
           'runlen_hist_S1S2_tokens': dict(sorted(hist.items())),
           'oracle_model': {}, 'realistic_model': {}, 'groups': {}}
    for cb in COMBOS:
        res['oracle_model'][cb] = {}
        for K in KS:
            acc = orc[cb][K]
            ct, cs = sum(a[0] for a in acc), sum(a[1] for a in acc)
            entry = {'all': model(ct, cs, total), 'copy_tokens': ct, 'copy_steps': cs}
            for k in range(nc):
                if tot[k]:
                    entry[CLASSES[k]] = model(acc[k][0], acc[k][1], tot[k])
            res['oracle_model'][cb][K] = entry
    for (mm, K, gated), acc in real.items():
        ct, cs = sum(acc['tokens']), sum(acc['steps'])
        ent = {'copy_tokens': ct, 'copy_steps': cs, 'steps_accepting_nothing': acc['zero'],
               'mean_tokens_per_copy_step': ct / cs if cs else None}
        B = total / R0
        for lab in ('naive', 'm1'):
            saved = sum(acc['saved_' + lab])
            new = B - saved + cs
            ent[lab] = {'gain': B / new, 'tokens_per_step': total / new,
                        'gain_at_c': {c: B / (B - saved + cs * c) for c in STEP_COSTS},
                        'breakeven_step_cost_for_20pct': (B / 1.2 - B + saved) / cs if cs else None}
        ent['per_class'] = {}
        for k in range(nc):
            if tot[k] and acc['steps'][k]:
                Bk = tot[k] / R0
                ent['per_class'][CLASSES[k]] = {
                    'gain_m1': Bk / (Bk - acc['saved_m1'][k] + acc['steps'][k]),
                    'gain_m1_at_c': {c: Bk / (Bk - acc['saved_m1'][k] + acc['steps'][k] * c)
                                     for c in STEP_COSTS},
                    'mean_tokens_per_copy_step': acc['tokens'][k] / acc['steps'][k]}
        res['realistic_model'][f'm>={mm},K={K}' + (',gated' if gated else '')] = ent
    for g, (n, c16, c64, ct, cs) in groups.items():
        B = n / R0
        new = B - ct / R_COPY_M1 + cs
        res['groups'][g] = {'output_tokens': n, 'share_runs_ge16': c16 / n,
                            'share_runs_ge64': c64 / n, 'gain_K64_m1': B / new}
    return res


def fidelity_summary(diffs):
    if not diffs:
        return {}
    s = sorted(diffs)
    return {'responses': len(s), 'median_rendered_minus_recorded': s[len(s) // 2],
            'p05': s[len(s) // 20], 'p95': s[len(s) * 19 // 20],
            'share_exact': sum(1 for x in s if x == 0) / len(s),
            'share_within_2': sum(1 for x in s if abs(x) <= 2) / len(s)}


# --------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['tokenize', 'match', 'global', 'report', 'all'])
    ap.add_argument('--work', required=True, help='scratch directory outside the repository')
    ap.add_argument('--runs-dir', default=RUNS)
    ap.add_argument('--skip-runs', default='', help='comma-separated run names to leave out')
    ap.add_argument('--only-runs', default='', help='comma-separated run names to keep')
    ap.add_argument('--limit', type=int, default=0, help='first N sessions only (testing)')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--mem-gb', type=float, default=4.0, help='RLIMIT_AS per worker')
    ap.add_argument('--global-mem-gb', type=float, default=14.0, help='RLIMIT_AS for the S4 phase')
    ap.add_argument('--pool-sessions', type=int, default=2, help='S3 proxy: other sessions per task')
    ap.add_argument('--pool-cap', type=int, default=600_000, help='S3 proxy: tokens per pool')
    ap.add_argument('--groups', default='', help='report: comma-separated session groups only')
    a = ap.parse_args()
    for sub in ('tok', 'match'):
        os.makedirs(os.path.join(a.work, sub), exist_ok=True)
    sessions, skipped = discover(a.runs_dir, set(filter(None, a.skip_runs.split(','))))
    if a.only_runs:
        keep = set(a.only_runs.split(','))
        sessions = [s for s in sessions if s[0] in keep]
    if a.limit:
        sessions = sessions[:a.limit]
    print(f'{len(sessions)} finished sessions; skipped instances: {dict(skipped)}', file=sys.stderr)
    name = lambda run, path: f'{run}.{os.path.basename(path)[:-6]}.pkl'
    tok_paths = [os.path.join(a.work, 'tok', name(r, f)) for r, _, f in sessions]
    if a.phase in ('tokenize', 'all'):
        jobs = [(r, i, f, tp, a.mem_gb) for (r, i, f), tp in zip(sessions, tok_paths)]
        with Pool(a.workers, maxtasksperchild=20) as pool:
            for k, _ in enumerate(pool.imap_unordered(tokenize_job, jobs, chunksize=1)):
                if k % 100 == 0:
                    print(f'tokenized {k}/{len(jobs)}', file=sys.stderr)
    if a.phase in ('match', 'all'):
        by_task = collections.defaultdict(list)
        for (r, i, f), tp in zip(sessions, tok_paths):
            by_task[i].append((r, tp))
        jobs = []
        for (r, i, f), tp in zip(sessions, tok_paths):
            others = [p for (r2, p) in sorted(by_task[i]) if r2 != r][:a.pool_sessions]
            jobs.append((tp, others, os.path.join(a.work, 'match', os.path.basename(tp)),
                         a.mem_gb, a.pool_cap))
        with Pool(a.workers, maxtasksperchild=10) as pool:
            for k, _ in enumerate(pool.imap_unordered(match_job, jobs, chunksize=1)):
                if k % 50 == 0:
                    print(f'matched {k}/{len(jobs)}', file=sys.stderr)
    if a.phase in ('global', 'all'):
        if not os.path.exists(os.path.join(a.work, 'global.done')):
            global_phase(tok_paths, a.work, a.global_mem_gb)
    if a.phase in ('report', 'all'):
        res = report(a.work, set(filter(None, a.groups.split(','))) or None)
        res['skipped_instances'] = dict(skipped)
        json.dump(res, sys.stdout, indent=1)
        print()


if __name__ == '__main__':
    main()
