# -*- coding: utf-8 -*-
"""MIA2: 会話SFT用に、既存の短い会話(chat_clean_*.jsonl)と新しい長い会話(data/long_chat/run/*.jsonl)を混ぜて整形する

短い会話は build_chat_sft.py で整形済みのものをそのまま使う(train 326,115 / val 3,000)。
長い会話は、短い会話と同じ直し(Botの発言の句読点補い)をして、ASCII外・markdown記号・同じ発言の繰り返し・連続を除く。
長い会話の検証用は、ランダムに1,000会話(学習から除く)。
出力: data/chat_mix/sft_{train,val}.npz (形式は build_chat_sft.py と同じ: ids / w / off)、data/chat_mix/long_val.jsonl (評価用の長い会話)
使い方: python build_chat_mix.py
"""
import glob
import json
import os
import random
import re

import numpy as np
from tokenizers import Tokenizer

from build_chat_sft import clean, encode  # noqa  (clean: ラベル除去・句読点補い / encode: ids と損失の重み)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", os.environ.get("MIA_MIX", "chat_mix"))
N_LONG_VAL = 1000
os.makedirs(OUT, exist_ok=True)
random.seed(0)


def ok(T):
    if any(re.search(r"[^\x00-\x7f]|[*#_`\n]|^\s*[-•]\s", t) for t in T):
        return False
    if len(set(T)) < len(T) or any(a.lower() == b.lower() for a, b in zip(T, T[1:])):
        return False
    return True


short_tr = [json.loads(l)["turns"] for l in open(os.path.join(HERE, "data/chat_qa/chat_clean_train.jsonl"), encoding="utf-8")]
short_va = [json.loads(l)["turns"] for l in open(os.path.join(HERE, "data/chat_qa/chat_clean_val.jsonl"), encoding="utf-8")]
long_all, drop = [], 0
for f in sorted(glob.glob(os.path.join(HERE, "data/long_chat/run/raw_lf*.jsonl"))) + [os.path.join(HERE, "data/long_chat/run/restored_dup.jsonl")]:
    for l in open(f, encoding="utf-8"):
        T = clean({"turns": json.loads(l)["turns"]})
        if ok(T):
            long_all.append(T)
        else:
            drop += 1
random.shuffle(long_all)
long_va, long_tr = long_all[:N_LONG_VAL], long_all[N_LONG_VAL:]
print(f"短い会話 train {len(short_tr)} val {len(short_va)} / 長い会話 採用 {len(long_all)} (除外 {drop}) → train {len(long_tr)} val {len(long_va)}")


def write(name, convs):
    ids, w, off = [], [], [0]
    for T in convs:
        i, ww = encode(T)
        ids += i; w += ww; off.append(len(ids))
    np.savez(os.path.join(OUT, f"sft_{name}.npz"), ids=np.array(ids, np.uint16), w=np.array(w, np.uint8), off=np.array(off, np.int64))
    print(f"{name}: {len(convs)} 会話 {len(ids) / 1e6:.2f}Mトークン (損失対象 {sum(w) / 1e6:.2f}M)")


write("train", short_tr + long_tr)
write("val", short_va + long_va)
with open(os.path.join(OUT, "long_val.jsonl"), "w", encoding="utf-8") as fo:
    for T in long_va:
        fo.write(json.dumps({"turns": T}, ensure_ascii=False) + "\n")
