# -*- coding: utf-8 -*-
"""MIA2 読解用に、「読解データにだけ出る割れる語」を語彙に足す(オーナー承認 2026-10-25)
対象: data/reading_entry/readonly_words.json のうち出現5回以上、小文字だけの語で WordNet にあるものだけ(大文字語・固有名・断片を除く)、複数形は単数が対象にあれば除く、大文字形は小文字形が対象にあれば除く。
足し方は build_tokenizer_v3.py と同じ(既存のmerge・IDは不変、末尾にmergeを足し、新IDは旧語彙の続き)。
出力: data/ts_v4/tokenizer_ext.json、data/ts_v4/added_words.json、data/ts_v4/new_ids.json(新ID→種類 word|piece)"""
import copy, json, os, sys
from tokenizers import Tokenizer
import nltk
nltk.data.path.insert(0, os.environ.get("NLTK_PATH", ""))
from nltk.corpus import wordnet as wn
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SRC, OUT = "data/ts_v3/tokenizer_ext.json", "data/ts_v4/tokenizer_ext.json"
ro = json.load(open("data/reading_entry/readonly_words.json"))
cand = {w: v[0] for w, v in ro.items() if v[0] >= MIN}
def proper(w): return w[0].isupper() and not wn.synsets(w.lower()) and w.lower() not in cand
def sing(w): return [w[:-1]] + ([w[:-2]] if w.endswith("es") else [])
targets = [w for w in sorted(cand, key=lambda w: -cand[w])
           if w.islower() and (wn.synsets(w) or wn.morphy(w)) and not proper(w) and not (w[0].isupper() and w.lower() in cand) and not (w.endswith("s") and any(s in cand for s in sing(w)))]
dropped = len(cand) - len(targets)
print(f"出現{MIN}回以上 {len(cand)} 語 → 大文字語・辞書にない語・固有名・複数形を除いて {len(targets)} 語")
base = json.load(open(SRC)); tok = Tokenizer.from_str(json.dumps(base))
d = copy.deepcopy(base); vocab, merges = d["model"]["vocab"], d["model"]["merges"]
have = {tuple(m) for m in merges}; nxt = max(vocab.values()) + 1; v0 = len(vocab)
remaining = list(targets)
for p in range(12):
    t = Tokenizer.from_str(json.dumps(d)); new, nxt_pass = [], []
    for w in remaining:
        toks = t.encode(" " + w).tokens
        if len(toks) == 1: continue
        nxt_pass.append(w); pair = (toks[0], toks[1])
        if pair not in have and pair not in new: new.append(pair)
    if not nxt_pass: break
    for a, b in new:
        merges.append([a, b]); have.add((a, b))
        if a + b not in vocab: vocab[a + b] = nxt; nxt += 1
    remaining = nxt_pass
json.dump(d, open(OUT, "w"), ensure_ascii=False)
t = Tokenizer.from_file(OUT)
bad = [w for w in targets if len(t.encode(" " + w).ids) != 1]
ids = {int(i): tk for tk, i in t.get_vocab().items() if i >= v0}
kind = {}
tw = {"Ġ" + w for w in targets}
for i, tk in ids.items(): kind[i] = "word" if tk in tw else "piece"
json.dump(targets, open("data/ts_v4/added_words.json", "w")); json.dump({str(i): [ids[i], kind[i]] for i in ids}, open("data/ts_v4/new_ids.json", "w"))
print(f"語彙 {v0} → {t.get_vocab_size()} (語 {sum(k=='word' for k in kind.values())} + 中間片 {sum(k=='piece' for k in kind.values())}) / 1トークンにならなかった語 {len(bad)} {bad[:10]}")
