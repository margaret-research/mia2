# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — ルーターMia用: data/router/{train,test}.jsonl を sft_chat.py が読める npz にする
並び: [BOS] <user> h0 <bot> h1 <eos> ... <user> 今の発言 <bot> TARGET <eos>   損失は TARGET と <eos> だけ
(履歴のBot返事の後ろの <eos> は bot_v3.py の履歴と同じ形)。検証は学習から1,000件を除いたもの。
使い方: MIA_TOK=data/ts_v4/tokenizer_ext.json python build_router_sft.py"""
import json, os, random
import numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.environ["MIA_TOK"])
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
enc = lambda s: tok.encode(s).ids

def seq(r):
    ids, w = [BOS], [0]
    h = r["hist"]
    for i, t in enumerate(h):
        e = enc(t)
        if i % 2 == 0: ids += [USER] + e; w += [0] * (1 + len(e))
        else: ids += [BOT] + e + [EOS]; w += [0] * (2 + len(e))
    e = enc(r["user"]); ids += [USER] + e + [BOT]; w += [0] * (len(e) + 2)
    t = enc(r["target"].replace("CALC", "calc", 1).replace("CHAT", "chat", 1).replace("CARD", "card", 1)) + [EOS]; ids += t; w += [1] * len(t)
    return np.array(ids, dtype=np.int32), np.array(w, dtype=np.int8)

def save(rows, d, name):
    os.makedirs(d, exist_ok=True)
    S = [seq(r) for r in rows]
    off = np.cumsum([0] + [len(i) for i, _ in S])
    np.savez(f"{d}/sft_{name}.npz", ids=np.concatenate([i for i, _ in S]), w=np.concatenate([w for _, w in S]), off=off)
    print(name, len(S), "最長", max(len(i) for i, _ in S))

SFX = ("_" + os.environ["RV"]) if os.environ.get("RV") else ""
if os.environ.get("AUG"): SFX += "aug"
rows = [json.loads(l) for l in open(f"data/router/train{SFX}.jsonl")]
random.seed(0); random.shuffle(rows)
save(rows[1000:], f"data/router_sft{SFX}", "train"); save(rows[:1000], f"data/router_sft{SFX}", "val")
