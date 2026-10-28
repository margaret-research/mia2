# -*- coding: utf-8 -*-
"""Mia の単語埋め込みの初期値を Qwen3-Embedding-0.6B から作る。
語トークン(Ġ+英字)は語そのものをQwenに通し、1024次元をPCAで384次元に縮める。語以外(断片・記号・特殊)は初期値なし(乱数のまま)。
出力: ckpt/emb_init_v3.pt {ids, vec, norm}  python で実行"""
import torch, numpy as np
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import os
D = int(os.environ.get("MIA_NEMBD", 384))
OUT = "ckpt/emb_init_v3.pt" if D == 384 else f"ckpt/emb_init_v3_{D}.pt"
tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
V = tok.get_vocab_size()
ids, words = [], []
for t, i in sorted(tok.get_vocab().items(), key=lambda x: x[1]):
    if t.startswith("Ġ") and len(t) > 1 and t[1:].isalpha():
        ids.append(i); words.append(t[1:])
print("語トークン", len(ids), "/ 全", V, flush=True)
m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
m.half()
X = m.encode(words, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
X = torch.tensor(X, dtype=torch.float32)
X = X - X.mean(0, keepdim=True)
U, S, Vh = torch.linalg.svd(X, full_matrices=False)
P = X @ Vh[:D].T
print(f"PCA {D}次元で残る分散", float((S[:D] ** 2).sum() / (S ** 2).sum()), flush=True)
P = P / P.norm(dim=1, keepdim=True) * 0.4      # 乱数初期値(std0.02)のノルム約0.39にそろえる
torch.save({"ids": torch.tensor(ids), "vec": P, "words": words}, OUT)
# 確認: 近い語
Pn = torch.nn.functional.normalize(P, dim=1)
for w in ["duck", "cow", "quack", "woodpecker", "vapor", "liter"]:
    if w in words:
        j = words.index(w); s = Pn @ Pn[j]; print(w, [words[k] for k in s.topk(7).indices.tolist()[1:]])
