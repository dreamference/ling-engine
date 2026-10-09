"""Production side of the agreement study: talks to the reference SGLang server (LING_REF_URL, default
http://127.0.0.1:8000) and does not change its configuration.

  prod.py gen IDS OUT MAXTOK [CONC]     greedy continuation per prompt, with production's own top-20 at each step
  prod.py force IDS GEN OUT CONC        teacher-forced top-20 over prompt + production's continuation, CONC requests at once
  prod.py flush                         empties the prefix cache (POST /flush_cache), for a cold pass

IDS: one {"ids": [...]} per line (ling_force tokenize). Stops at <|im_end|> (id 248046).
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("LING_REF_URL", "http://127.0.0.1:8000")
K = 20


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read() or b"{}")


def gen_one(p, maxtok):
    return post("/generate", {"input_ids": p, "sampling_params": {"temperature": 0, "max_new_tokens": maxtok,
                                                                          "stop_token_ids": [248046]},
                                   "return_logprob": True, "top_logprobs_num": K})


def gen(ids_path, out_path, maxtok, conc=1):
    prompts = [json.loads(l)["ids"] for l in open(ids_path)]
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda p: gen_one(p, maxtok), prompts))
    with open(out_path, "w") as f:
        for r in res:
            m = r["meta_info"]
            cont = [t[1] for t in m["output_token_logprobs"]]
            top = [[[c[1], c[0]] for c in pos] for pos in m["output_top_logprobs"]]
            f.write(json.dumps({"cont": cont, "top": top, "text": r["text"], "cached": m.get("cached_tokens")}) + "\n")


def force_one(p, cont):
    ids = p + cont
    r = post("/generate", {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                           "return_logprob": True, "logprob_start_len": len(p) - 1, "top_logprobs_num": K})
    m = r["meta_info"]
    tops = m["input_top_logprobs"]  # entry i scores ids[len(p) - 1 + i + 1]
    top = [[[c[1], c[0]] for c in tops[j + 1]] for j in range(len(cont))]
    return {"top": top, "cached": m.get("cached_tokens")}


def force(ids_path, gen_path, out_path, conc):
    prompts = [json.loads(l)["ids"] for l in open(ids_path)]
    conts = [json.loads(l)["cont"] for l in open(gen_path)]
    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda a: force_one(*a), zip(prompts, conts)))
    with open(out_path, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"{out_path}: {len(res)} prompts in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen": gen(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif cmd == "force": force(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == "flush":
        req = urllib.request.Request(URL + "/flush_cache", data=b"", method="POST")
        print(urllib.request.urlopen(req, timeout=60).read()[:100])
