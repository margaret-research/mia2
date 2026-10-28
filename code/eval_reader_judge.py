# -*- coding: utf-8 -*-
"""reader_k.jsonl の読む役の答えをQwenで正誤判定し、検索方式×読む枚数kごとの正答率を出す。
Botの動作: 上位k枚を順に読み、"I do not know." でない最初の答えを返す。全部IDKなら IDK。
判定 C=正しい(見本または正解カードと同じ意味、質問に答えている) / W=誤りまたは質問に答えていない"""
import json, re
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local
IDK = "I do not know."
R = [json.loads(l) for l in open("data/kb_cov/reader_k{}.jsonl".format(__import__("os").environ.get("TAG","")), encoding="utf-8")]
P = """Grade the candidate answers of a children's question-answering bot.
Question: {q}
Correct answer (reference): {exp}
Card with the true fact: {gold}
For each numbered candidate, reply C if it is a correct answer to the question (same meaning as the reference or the card, wording may differ; extra true words are fine), W if it is wrong, nonsense, or does not answer the question.
Candidates:
{items}
Output JSON Lines only: {{"i": 1, "v": "C"}}"""


def judge(r):
    cands = []
    for key in ("emb", "tfidf"):
        for h in r[key]:
            if h["a"] and h["a"] != IDK and h["a"] not in cands:
                cands.append(h["a"])
    if not cands:
        return {}
    t = local.ask(P.format(q=r["q"], exp=r["expected"], gold=r["gold_card"], items="\n".join(f"{i}. {c}" for i, c in enumerate(cands, 1))), max_tokens=300, temperature=0)
    v = {int(i): c.upper() for i, c in re.findall(r'"i"\s*:\s*(\d+)\s*,\s*"v"\s*:\s*"([A-Za-z])', t)}
    return {c: v.get(i, "?") for i, c in enumerate(cands, 1)}


with ThreadPoolExecutor(local.WORKERS) as ex:
    J = list(ex.map(judge, R))
N = len(R)
print(f"N={N}")
print("方式   k  | 正しい  誤答  IDK")
res = {}
for key, nm in (("tfidf", "TF-IDF"), ("emb", "Qwen3-Emb")):
    for k in (1, 3, 5, 10):
        c = w = d = 0
        for r, j in zip(R, J):
            ans = next((h["a"] for h in r[key][:k] if h["a"] and h["a"] != IDK), None)
            if ans is None: d += 1
            elif j.get(ans) == "C": c += 1
            else: w += 1
        print(f"{nm:10s} {k:2d} | {c/N:5.0%} {w/N:5.0%} {d/N:5.0%}")
json.dump([dict(q=r["q"], j=j) for r, j in zip(R, J)], open("data/kb_cov/reader_k_judge{}.json".format(__import__("os").environ.get("TAG","")), "w"), ensure_ascii=False)
# 正解カード1枚だけ読ませたとき(読む役の上限)は、gold が上位10に入った質問だけ別に見る
