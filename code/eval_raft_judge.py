import json, re, os
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local
IDK = "I do not know."
T = os.environ.get("TAG", "_raft")
R = [json.loads(l) for l in open(f"data/kb_cov/raft_eval{T}.jsonl")]
KEYS = ("top3", "top4", "top5", "gold4", "gold1")
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
    for k in KEYS:
        a = r[k]
        if a and a != IDK and a not in cands: cands.append(a)
    if not cands: return {}
    t = local.ask(P.format(q=r["q"], exp=r["expected"], gold=r["gold_card"], items="\n".join(f"{i}. {c}" for i, c in enumerate(cands, 1))), max_tokens=300, temperature=0)
    v = {int(i): c.upper() for i, c in re.findall(r'"i"\s*:\s*(\d+)\s*,\s*"v"\s*:\s*"([A-Za-z])', t)}
    return {c: v.get(i, "?") for i, c in enumerate(cands, 1)}
with ThreadPoolExecutor(local.WORKERS) as ex: J = list(ex.map(judge, R))
N = len(R); print("N", N)
for k in KEYS:
    c = w = d = 0
    for r, j in zip(R, J):
        a = r[k]
        if a == IDK or not a: d += 1
        elif j.get(a) == "C": c += 1
        else: w += 1
    print(f"{k:6s} 正しい {c} ({c/N:.0%}) 誤答 {w} ({w/N:.0%}) わからない {d} ({d/N:.0%})")
json.dump([dict(q=r["q"], j=j) for r, j in zip(R, J)], open(f"data/kb_cov/raft_eval_judge{T}.json", "w"), ensure_ascii=False)
