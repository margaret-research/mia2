# -*- coding: utf-8 -*-
"""T-026 v3: 拡張トークナイザーに「カード・言い換えQAに多く出るのに分解される語」を足す (旧 9,488語 → +N語)
 ・数える対象は大文字小文字を区別した語形(文頭の "Electricity" と "electricity" は別トークン)。
 ・語を1トークンにするため、BPEのmergeを末尾に足す(既存のmergeは不変、既存IDも不変、新IDは 9488 から)。
   1回のパスで「まだ分解されている語」の先頭2片をmergeし、全部1トークンになるまで繰り返す。途中の片(中間トークン)も語彙に入る。
使い方: python build_tokenizer_v3.py [N=4000]   出力: data/ts/tokenizer_v3.json, data/ts/tokenizer_v3_added.json"""
import collections, json, re, sys, copy
from tokenizers import Tokenizer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SRC, OUT = "data/ts/tokenizer_ext.json", "data/ts/tokenizer_v3.json"
base = json.load(open(SRC))
tok = Tokenizer.from_str(json.dumps(base))
cnt = collections.Counter()
def feed(text):
    for w in re.findall(r"[A-Za-z]+", text):
        cnt[w] += 1
for l in open("data/kb_lowgrade/facts_final.jsonl", encoding="utf-8"):
    feed(json.loads(l)["fact"])
for l in open("data/kb_para/qa_raw.jsonl", encoding="utf-8"):
    r = json.loads(l); feed(r["q"]); feed(r["a"])
import glob
for f in glob.glob("data/reading_qa/raw2_s*.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("named"):
            feed(r["card"]); feed(r["q"]); feed(r["a"])
pieces = {w: len(tok.encode(" " + w).ids) for w in cnt}
split = {w: c for w, c in cnt.items() if pieces[w] > 1}
order = sorted(split, key=lambda w: -split[w])
targets = order[:N]
tot = sum(cnt.values()); spt = sum(split.values())
print(f"語形 {len(cnt)} / 分解される語形 {len(split)} (出現 {spt}/{tot} = {spt/tot:.1%}) / 足す語 {len(targets)} (出現 {sum(split[w] for w in targets)/spt:.0%}を覆う)")

d = copy.deepcopy(base)
vocab, merges = d["model"]["vocab"], d["model"]["merges"]
have = {tuple(m) for m in merges}
nxt = max(vocab.values()) + 1
remaining = list(targets)
for p in range(12):
    t = Tokenizer.from_str(json.dumps(d))
    new, nxt_pass = [], []
    for w in remaining:
        toks = t.encode(" " + w).tokens
        if len(toks) == 1:
            continue
        nxt_pass.append(w)
        pair = (toks[0], toks[1])
        if pair not in have and pair not in new:
            new.append(pair)
    if not nxt_pass:
        break
    for a, b in new:
        merges.append([a, b]); have.add((a, b))
        if a + b not in vocab:
            vocab[a + b] = nxt; nxt += 1
    remaining = nxt_pass
    print(f"  パス{p+1}: 未完了 {len(nxt_pass)} 語 / 追加merge {len(new)}", flush=True)
json.dump(d, open(OUT, "w"), ensure_ascii=False)
t = Tokenizer.from_file(OUT)
bad = [w for w in targets if len(t.encode(" " + w).ids) != 1]
print(f"語彙 {t.get_vocab_size()} (旧 {tok.get_vocab_size()} + 目標語 {len(targets) - len(bad)} + 中間片 {t.get_vocab_size() - tok.get_vocab_size() - (len(targets) - len(bad))}) / 1トークンにならなかった語 {len(bad)} {bad[:10]}")
json.dump(targets, open("data/ts/tokenizer_v3_added.json", "w"))
