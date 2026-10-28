# -*- coding: utf-8 -*-
"""読解データにだけ出る「割れる語」を探す: 読解データ(full, full2)の語形のうち、トークナイザーで2片以上に割れ、
基礎学習データ(ts_v3・simplestories の train.bin)と会話データ(chat_mix の train)に、その片の並びが1回も出ないもの。
出力: data/reading_entry/readonly_words.json  {語: [読解での出現数, 片数]}"""
import collections, glob, json, re
import numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file("data/ts_v3/tokenizer_ext.json")
cnt = collections.Counter()
for f in glob.glob("data/reading_entry/full*/raw3_*.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        for t in [r["entry"]] + [q["q"] + " " + q["a"] for q in r["qas"]]:
            cnt.update(re.findall(r"[A-Za-z]+", t))
seqs = {w: tuple(tok.encode(" " + w).ids) for w in cnt}
split = {w: s for w, s in seqs.items() if len(s) >= 2}
print("語形", len(cnt), "割れる語形", len(split), flush=True)
B1, B2 = np.uint64(1000003), np.uint64(0)
def h(seq):
    x = np.uint64(0)
    for t in seq: x = x * B1 + np.uint64(t + 1)
    return x
bylen = collections.defaultdict(dict)
for w, s in split.items(): bylen[len(s)][int(h(s))] = w
found = set()
def scan(a):
    a = a.astype(np.uint64)
    for L, d in bylen.items():
        if len(a) < L: continue
        x = np.zeros(len(a) - L + 1, np.uint64)
        for k in range(L): x = x * B1 + (a[k:len(a) - L + 1 + k] + np.uint64(1))
        keys = np.array(list(d), np.uint64)
        hit = np.unique(x[np.isin(x, keys)])
        found.update(d[int(k)] for k in hit)
for p in ["data/ts_v3/train.bin", "data/simplestories/train.bin"]:
    mm = np.memmap(p, dtype=np.uint16, mode="r")
    for i in range(0, len(mm), 30_000_000):
        scan(np.asarray(mm[i:i + 30_000_000]))
    print(p, "済", len(found), flush=True)
z = np.load("data/chat_mix/sft_train.npz"); scan(z["ids"])
only = {w: [cnt[w], len(split[w])] for w in split if w not in found}
json.dump(only, open("data/reading_entry/readonly_words.json", "w"), ensure_ascii=False)
print("読解にだけ出る割れる語", len(only), "出現延べ", sum(v[0] for v in only.values()))
for n in (2, 3, 4, 5):
    print(f"片数>={n}: {sum(v[1] >= n for v in only.values())} 語、出現延べ {sum(v[0] for v in only.values() if v[1] >= n)}")
