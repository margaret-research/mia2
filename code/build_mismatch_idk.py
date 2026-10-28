# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — T-026 会話Bot用: 「検索で来たカードに答えがない」質問を IDK にする学習データを作る

会話Botでは、検索で質問と関係の薄いカードが来ても、読解モデルがIDKと言わず答えを作る問題が出た(設計メモ§26)。
そこで、答えのある質問 q (元のカード A) を、検索(TF-IDF)で q に近い「別のカード B」と組み合わせ、B に答えの内容語がほぼ無い場合に
「Facts: B Question: q -> I do not know.」を作る。B が実は答えを含んでいる誤りを避けるため、答えの内容語のうち B にある割合が 1/3 以下のものだけ採用。
学習側は評価題材のカード/質問を使わない。評価側 (test_mismatch) は評価題材の質問 × 検索で来た別カード(学習側のカードでもよい)。
出力: data/reading_qa/m120k_train.jsonl (= s100k_train + 学習側の不一致IDK), m_test_mismatch.jsonl
使い方: python build_mismatch_idk.py [学習側の追加件数=20000]
"""
import json
import random
import re
import sys

from knowledge_kb import DROP, KB, stem

D = "data/reading_qa/"
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rnd = random.Random(31)
S = lambda t: {stem(w) for w in re.findall(r"[a-z]+", t.lower())}
DROPS = {stem(w) for w in DROP}
kb = KB()
held = {json.loads(l)["subject"] for l in open(D + "s10k_test_newsubj.jsonl", encoding="utf-8")}
rows = [json.loads(l) for f in ("raw2_s10k", "raw2_s20k", "raw2_s100k") for l in open(D + f + ".jsonl", encoding="utf-8")]
rows = [r for r in rows if not r["named"] and not r["idk"] and r["type"] in ("wh", "yes", "no", "two")]
idx = {c: i for i, c in enumerate(kb.cards)}


def negatives(r, allow_held, k=6):
    """r の質問に近い別カードのうち、答えの内容語がほぼ無いもの(1つ)。"""
    key = S(r["a"]) - S(r["q"]) - DROPS
    if not key:
        return None
    for sc, card, sub in kb.search(r["q"], k):
        if card == r["card"] or (not allow_held and sub in held) or sc < 0.1:
            continue
        if len(key & S(card)) / len(key) <= 1 / 3:
            return card, sc
    return None


def make(r, card, sc):
    return {"prompt": f"Facts: {card} Question: {r['q']}", "answer": "I do not know.", "idk": True, "type": "mis", "domain": r["domain"],
            "subject": r["subject"], "name": "", "q": r["q"], "card": card, "score": round(sc, 3)}


tr_rows = [r for r in rows if r["subject"] not in held]
te_rows = [r for r in rows if r["subject"] in held]
rnd.shuffle(tr_rows); rnd.shuffle(te_rows)
neg_tr, seen = [], set()
for r in tr_rows:
    if len(neg_tr) >= N_TRAIN:
        break
    if (r["q"], r["card"]) in seen:
        continue
    n = negatives(r, allow_held=False)
    if n:
        seen.add((r["q"], r["card"]))
        neg_tr.append(make(r, *n))
neg_te = []
for r in te_rows:
    if len(neg_te) >= 300:
        break
    n = negatives(r, allow_held=True)
    if n:
        neg_te.append(make(r, *n))
base = [json.loads(l) for l in open(D + "s100k_train.jsonl", encoding="utf-8")]
train = base + neg_tr
rnd.shuffle(train)
wr = lambda n, d: open(D + n, "w", encoding="utf-8").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in d))
wr("m120k_train.jsonl", train); wr("m_test_mismatch.jsonl", neg_te)
print(f"学習: 元 {len(base)} + 不一致IDK {len(neg_tr)} = {len(train)} (IDK {sum(x['idk'] for x in train)} = {sum(x['idk'] for x in train) / len(train):.1%})")
print(f"評価 m_test_mismatch: {len(neg_te)} 件 (評価題材の質問 × 別カード)")
print("スコア平均 学習", round(sum(x['score'] for x in neg_tr) / len(neg_tr), 3), "評価", round(sum(x['score'] for x in neg_te) / len(neg_te), 3))
for x in rnd.sample(neg_tr, 8):
    print(" Q:", x["q"], "\n   Card:", x["card"][:130], f"[{x['score']}]")
