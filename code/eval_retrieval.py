# -*- coding: utf-8 -*-
"""カード選びの評価: 根拠カードが上位k枚に入る割合 (data/kb_cov/measure_pilot_f.jsonl の A かつ conflict でない質問)
使い方: python eval_retrieval.py MODEL [--sub]   例: BAAI/bge-small-en-v1.5"""
import json, re, sys
import numpy as np
from sentence_transformers import SentenceTransformer
from knowledge_kb import KB

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
            Q.append((r["q"], i))
name = sys.argv[1]
qp = "Represent this sentence for searching relevant passages: " if "bge" in name and "en-v1.5" in name else ""
q3 = "Qwen3-Embedding" in name
model = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": "float16"} if q3 else {})
if q3:
    model.max_seq_length = 128
    model.tokenizer.padding_side = "left"
docs = [(f"{s}: {c}" if "--sub" in sys.argv else c) for c, s in zip(kb.cards, kb.subjects)]
D = model.encode(docs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
if q3:
    TASK = "Given a question from a child, retrieve the fact card that answers it"
    E_ = model.encode([f"Instruct: {TASK}\nQuery:{q}" for q, _ in Q], normalize_embeddings=True, convert_to_numpy=True)
else:
    E_ = model.encode([qp + q for q, _ in Q], normalize_embeddings=True, convert_to_numpy=True)
S = E_ @ D.T
ranks = [int((S[k] > S[k, i]).sum()) + 1 for k, (_, i) in enumerate(Q)]
N = len(ranks)
print(name, "--sub" if "--sub" in sys.argv else "", f"N={N}", " ".join(f"top{k}={sum(x<=k for x in ranks)/N:.0%}" for k in (1, 3, 5, 10, 20)))
