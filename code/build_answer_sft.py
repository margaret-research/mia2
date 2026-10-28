# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — 答える側のSFTデータ: data/router/answer_train.jsonl を sft_chat.py が読める npz にする
並び: [BOS] 履歴 <user> Facts: 式 = 結果. Question: 発言 <bot> 返事 <eos>  損失は返事+<eos>だけ。
REPLAY=N なら read_mix_v5 / chat_mix_v4 の既存SFTデータからN件を混ぜる (現行Miaへ混ぜる案=基準テストを落とさないための復習)。
使い方: MIA_TOK=... python build_answer_sft.py 出力ディレクトリ [REPLAY件数]"""
import json, os, random, sys
import numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.environ["MIA_TOK"])
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
enc = lambda s: tok.encode(s).ids

def seq(r):
    ids, w = [BOS], [0]
    for i, t in enumerate(r["hist"]):
        e = enc(t)
        if i % 2 == 0: ids += [USER] + e; w += [0] * (1 + len(e))
        else: ids += [BOT] + e + [EOS]; w += [0] * (2 + len(e))
    e = enc(f"Facts: {r['expr']} = {r['result']}. Question: {r['user']}"); ids += [USER] + e + [BOT]; w += [0] * (len(e) + 2)
    t = enc(r["reply"]) + [EOS]; ids += t; w += [1] * len(t)
    return np.array(ids, dtype=np.int32), np.array(w, dtype=np.int8)

def read_npz(d, split):
    z = np.load(f"{d}/sft_{split}.npz"); ids, w, off = z["ids"], z["w"], z["off"]
    return [(ids[off[i]:off[i + 1]], w[off[i]:off[i + 1]]) for i in range(len(off) - 1)]

def save(S, d, name):
    os.makedirs(d, exist_ok=True)
    off = np.cumsum([0] + [len(i) for i, _ in S])
    np.savez(f"{d}/sft_{name}.npz", ids=np.concatenate([i for i, _ in S]).astype(np.int32), w=np.concatenate([w for _, w in S]).astype(np.int8), off=off)
    print(name, len(S))

out, n_rep = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = [json.loads(l) for l in open("data/router/answer_train.jsonl")]
random.seed(0); random.shuffle(rows)
val, tr = [seq(r) for r in rows[:500]], [seq(r) for r in rows[500:]]
if n_rep:
    rep = read_npz("data/read_mix_v5", "train"); random.shuffle(rep); tr += rep[:n_rep]
    print("復習", n_rep, "(読解+会話の混合)")
save(tr, out, "train"); save(val, out, "val")
