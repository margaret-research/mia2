# -*- coding: utf-8 -*-
"""検索(TF-IDF / Qwen3-Embedding-0.6B)×読む枚数k で、読む役が実際に何と答えるかを集める。
各質問について上位10枚を1枚ずつ読ませて答えを保存(判定は別: Qwenに見本と照らす)。
出力 data/kb_cov/reader_k{}.jsonl   使い方: python eval_reader_k.py"""
import json, re, os
import numpy as np, torch
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from knowledge_kb import KB
from model import GPT, Config

dev = "cuda"
kb = KB()
n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
idx = {n(c): i for i, c in enumerate(kb.cards)}
m = [json.loads(l) for l in open("data/kb_cov/measure_pilot_f.jsonl")]
Q = []
for r in m:
    if r["v"] == "A" and r["flag"] != "conflict":
        e = n(r["evidence"]); i = idx.get(e)
        if i is None:
            c = [j for c_, j in idx.items() if e in c_]; i = c[0] if c else None
        if i is not None:
            Q.append(dict(q=r["q"], expected=r["expected"], answer=r["answer"], gold=i))
print("questions", len(Q), flush=True)

EMB = "data/kb_emb_qwen3_0.6b.npy"
name = "Qwen/Qwen3-Embedding-0.6B"
st = SentenceTransformer(name, device=dev, model_kwargs={"torch_dtype": "float16"})
st.max_seq_length = 128; st.tokenizer.padding_side = "left"
if os.path.exists(EMB):
    D = np.load(EMB)
else:
    D = st.encode(kb.cards, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    np.save(EMB, D)
TASK = "Given a question from a child, retrieve the fact card that answers it"
E_ = st.encode([f"Instruct: {TASK}\nQuery:{q['q']}" for q in Q], normalize_embeddings=True, convert_to_numpy=True)
S = E_ @ D.T
del st; torch.cuda.empty_cache()

tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
qa = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=256)).to(dev)
qa.load_state_dict(torch.load(os.environ.get("QA","ckpt/readqa_ext_e15_m120k.pt"), map_location=dev)["model"]); qa.eval()
cache = {}


@torch.no_grad()
def read(card, q, max_new=60):
    k = (card, q)
    if k in cache:
        return cache[k]
    ids = [BOS, USER] + tok.encode(f"Facts: {card} Question: {q}").ids + [BOT]
    n0 = len(ids)
    for _ in range(max_new):
        logits, _ = qa(torch.tensor([ids[-256:]], device=dev))
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        ids.append(nxt)
    cache[k] = tok.decode(ids[n0:]).strip()
    return cache[k]



import random
random.seed(7)
out = []
for k, q in enumerate(Q):
    order = [int(i) for i in np.argsort(-S[k])[:10]]
    rec = dict(q, gold_card=kb.cards[q["gold"]])
    for kk in (3, 4, 5):
        rec[f"top{kk}"] = read(" ".join(kb.cards[i] for i in order[:kk]), q["q"])
    dis = [i for i in order if i != q["gold"]][:3]
    cs = [kb.cards[q["gold"]]] + [kb.cards[i] for i in dis]; random.shuffle(cs)
    rec["gold4"] = read(" ".join(cs), q["q"]); rec["gold4_cards"] = cs
    rec["gold1"] = read(kb.cards[q["gold"]], q["q"])
    out.append(rec)
    if (k + 1) % 50 == 0:
        print(k + 1, flush=True)
with open("data/kb_cov/raft_eval{}.jsonl".format(os.environ.get("TAG","")), "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("done")
