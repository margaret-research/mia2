# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — 項目の検索方式の比較 (読解テスト50項目の931問、ルーターの出力は logs/card_routed.json)
 A 題名索引のみ(今) / B 本文の埋め込み検索のみ / C 本文の埋め込み+ルーターの題名の手がかり (重みを変えて)
 python で実行 (sentence-transformers)"""
import json, os, sys
import numpy as np
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer
import torch
from card_retrieve import Index, norm
from gen_card_data import mentions
idx = Index(); titles = idx.titles
st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={"torch_dtype": torch.float16}); st.max_seq_length = 512; st.tokenizer.padding_side = "left"
cache = "data/card_emb_entries.npy"
docs = [f"{t}. {idx.ents[t]['entry']}" for t in titles]
if os.path.exists(cache): E = np.load(cache)
else: E = st.encode(docs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False); np.save(cache, E)
R = json.load(open("logs/card_routed.json"))
TASK = "Given a question from a child, retrieve the encyclopedia entry that answers it"
Q = st.encode([f"Instruct: {TASK}\nQuery:{r['q']}" for r in R], normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
S = Q @ E.T                                              # 質問 × 項目
ti = {t: i for i, t in enumerate(titles)}
def acc(pick): return sum(pick(i) == r["gold"] for i, r in enumerate(R)) / len(R)
print("A 題名索引のみ(今)    ", acc(lambda i: idx.find(R[i]["routed"][4:].strip(), R[i]["q"]) if R[i]["routed"].startswith("card") else None))
print("B 本文の埋め込みのみ   ", acc(lambda i: titles[int(S[i].argmax())]))
# 題名の手がかり: ルーターの出力と完全一致する題名は +w、質問に題名が出ていれば +m
bonus = np.zeros_like(S)
for i, r in enumerate(R):
    if r["routed"].startswith("card"):
        n = norm(r["routed"][4:].strip())
        if n in idx.by_norm: bonus[i, ti[idx.by_norm[n]]] = 1.0
        for j, t in enumerate(titles):
            if mentions(t, r["q"]): bonus[i, j] += 0.5
for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
    print(f"C 埋め込み+題名の手がかり w={w:<4}", acc(lambda i: titles[int((S[i] + w * bonus[i]).argmax())]))
top3 = np.argsort(-S, axis=1)[:, :3]
print("参考: 埋め込みのみ 上位3枚に正解", sum(R[i]["gold"] in [titles[j] for j in top3[i]] for i in range(len(R))) / len(R))
