"""Decodes the worst-KL positions' contexts (extra.py's worst.json) with the reference tokenizer.
Runs where transformers is installed (production's image): decode_worst.py MODEL_DIR STUDY_DIR"""
import json, os, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
D = sys.argv[2]
w = json.load(open(os.path.join(D, "worst.json")))
for x in w:
    x["context_text"] = tok.decode(x["context"])[-160:]
    x["prod_top3_text"] = [(tok.decode([c[0]]), round(c[1], 3)) for c in x["prod_top3"]]
    x["eng_top3_text"] = [(tok.decode([c[0]]), round(c[1], 3)) for c in x["eng_top3"]]
    del x["context"]
json.dump(w, open(os.path.join(D, "worst-decoded.json"), "w"), indent=1, ensure_ascii=False)
