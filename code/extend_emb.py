# -*- coding: utf-8 -*-
"""基礎モデル(pre_stories_B1024.pt)の埋め込みに、新トークナイザー(ts_v4)で足した行を足す(オーナー承認 2026-10-25)
 語トークン: build_emb_init.py と同じ PCA(旧の語トークンで作った平均・基底)で、Qwen3-Embedding の埋め込みを投影し、ノルム0.4にそろえる。
 中間片: 旧モデルで語以外のトークンに入っている乱数と同じ分布(平均0・標準偏差0.02)の乱数(固定シード)。
 旧の行は変えない。まず旧の語トークンで PCA をやり直し、保存済みの値(ckpt/emb_init_v3_512.pt)と一致するか確かめる。
出力: ckpt/pre_stories_B1024_v4.pt  python で実行"""
import json, torch
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
D = 512
ck = torch.load("ckpt/pre_stories_B1024.pt", map_location="cpu")
W = ck["model"]["wte.weight"].float(); V0 = W.shape[0]
saved = torch.load("ckpt/emb_init_v3_512.pt")
t4 = Tokenizer.from_file("data/ts_v4/tokenizer_ext.json"); assert t4.get_vocab_size() > V0
new = json.load(open("data/ts_v4/new_ids.json"))
m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda"); m.half()
enc = lambda ws: torch.tensor(m.encode(ws, batch_size=256, normalize_embeddings=True, show_progress_bar=False), dtype=torch.float32)
X = enc(saved["words"]); mu = X.mean(0, keepdim=True)
U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
P = (X - mu) @ Vh[:D].T; P = P / P.norm(dim=1, keepdim=True) * 0.4
cos = torch.nn.functional.cosine_similarity(P, saved["vec"], dim=1)
print("PCAやり直しと保存値の一致 cos 最小", float(cos.min()), "平均", float(cos.mean()))
# 実際のモデルの行とも照合
r = W[saved["ids"]]; print("モデルの行との cos 最小", float(torch.nn.functional.cosine_similarity(r, saved["vec"], dim=1).min()))
rest = torch.ones(V0, dtype=torch.bool); rest[saved["ids"]] = False
print("乱数行の標準偏差", float(W[rest].std()), "ノルム平均", float(W[rest].norm(dim=1).mean()))
g = torch.Generator().manual_seed(0)
rows = torch.randn(t4.get_vocab_size() - V0, D, generator=g) * float(W[rest].std())
ids_w = [int(i) for i, (tk, kind) in new.items() if kind == "word"]
words = [new[str(i)][0][1:] for i in ids_w]
Pn = (enc(words) - mu) @ Vh[:D].T; Pn = Pn / Pn.norm(dim=1, keepdim=True) * 0.4
for i, v in zip(ids_w, Pn): rows[i - V0] = v
print("語トークン", len(ids_w), "中間片", len(new) - len(ids_w))
Pa = torch.nn.functional.normalize(P, dim=1)
for w in ["nougat", "bonsai", "caribou", "kneecap"]:
    v = torch.nn.functional.normalize(Pn[words.index(w)], dim=0); s = Pa @ v
    print(w, "に近い旧語:", [saved["words"][k] for k in s.topk(6).indices.tolist()])
ck["model"]["wte.weight"] = torch.cat([W.to(ck["model"]["wte.weight"].dtype), rows.to(ck["model"]["wte.weight"].dtype)])
if "head.weight" in ck["model"]: ck["model"]["head.weight"] = ck["model"]["wte.weight"]
torch.save(ck, "ckpt/pre_stories_B1024_v4.pt")
print("saved", ck["model"]["wte.weight"].shape, [k for k in ck["model"] if "head" in k])
