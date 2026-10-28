# -*- coding: utf-8 -*-
"""事前学習に混ぜる知識テキスト: 題材ごとにカードを並べた短い文章(<bos>...<eos>)を、順序を変えて COPIES 回ぶん作り train.bin の後ろに足す。
 新しい語(例: electricity, solid, mammals)を土台の学習で見せるため。評価用の題材とカード(281問の正解カード)は入れない。
使い方: python -I build_knowledge_text.py <data/ts_v3>   (train.bin を作ったあとに1回だけ実行)"""
import collections, json, random, re, sys
import numpy as np
from tokenizers import Tokenizer
D = sys.argv[1]; COPIES = 10
tok = Tokenizer.from_file(f"{D}/tokenizer_ext.json")
BOS, EOS = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
held = {json.loads(l)["subject"] for l in open("data/kb_cov/measure_pilot_f.jsonl", encoding="utf-8")}
gold = {n(json.loads(l)["gold_card"]) for l in open("data/kb_cov/reader_k.jsonl", encoding="utf-8")}
by = collections.defaultdict(list)
for l in open("data/kb_lowgrade/facts_final.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["subject"] in held or n(r["fact"]) in gold:
        continue
    by[r["subject"]].append(r["fact"])
rnd = random.Random(0); docs = []
for c in range(COPIES):
    for s, fs in by.items():
        fs = fs[:]; rnd.shuffle(fs)
        for i in range(0, len(fs), 8):
            docs.append(" ".join(fs[i:i + 8]))
rnd.shuffle(docs)
arr = np.concatenate([np.array([BOS] + e.ids + [EOS], dtype=np.uint16) for e in tok.encode_batch(docs)])
with open(f"{D}/train.bin", "ab") as f:
    arr.tofile(f)
print(f"題材 {len(by)} / 文書 {len(docs)} / 追加 {len(arr)/1e6:.1f}Mトークン")
