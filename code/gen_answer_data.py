# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — 答える側の学習データ: 電卓の結果を受けた「自然な答えの文」をQwenに書かせる (ローカル、費用なし)
入力: data/router/train_v2aug.jsonl の計算行 (履歴つき) / 出力: data/router/answer_train.jsonl
  {"hist","user","expr","result","reply","type"}  reply = Mia の答え (1〜2文)
検査: 結果の数をちょうど1回含む / 質問と履歴にない新しい数を言わない / ASCII / 25語以内 / 質問に答える形。
約1割は固定文 ("The answer is X." など) も混ぜる。使い方: python gen_answer_data.py  ※先にvLLM(Qwen)起動"""
import json, os, random, re, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local
from gen_router_para import nums_of

PROMPT = """You are Mia, a kind helper for children in grade 1-2. A calculator already worked out the answer. Write how you tell the answer to the child.
{ctx}Child says: {u}
Calculation: {e} = {r}
Write {k} different replies. Rules:
- Each reply is 1 or 2 short sentences, easy words a 7-year-old knows, under 20 words, plain ASCII.
- Say the answer {r} exactly once, written exactly like that, and say what it is about (for example "You have {r} pencils now." or "{r} stickers are left." or "7 times 8 is {r}.").
- Do not say any other new number. Do not show the calculation steps. Do not ask a question back. No emoji, no markdown.
- Vary the wording between replies (sometimes start with "Let me see." or "Good question." but not always).
Output {k} lines only, one reply per line, no numbering."""
FIXED = ["The answer is {r}.", "It is {r}.", "The answer is {r}.", "That is {r}."]


def ok_reply(line, r, allowed):
    if not line or not line.isascii() or len(line.split()) > 25 or "?" in line or "=" in line:
        return False
    got = Counter(k.replace(',', '') for k in nums_of(line).elements())
    want = r.replace(",", "")
    if got.get(want, 0) != 1 or len(re.findall(re.escape(r).replace(",", ",?"), line)) != 1:
        return False
    return all(k == want or k in allowed for k in got)


def job(row):
    ctx = ""
    if row["hist"]:
        ctx = "Earlier in the talk: " + " / ".join(row["hist"][-4:]) + "\n"
    allowed = set(nums_of(" ".join(row["hist"] + [row["user"]])))
    allowed = {k.replace(",", "") for k in allowed}
    try:
        txt = local.ask(PROMPT.format(ctx=ctx, u=row["user"], e=row["target"][5:], r=row["result"], k=2), max_tokens=200, temperature=0.9)
    except Exception:
        return []
    out = []
    for line in txt.strip().split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
        if ok_reply(line, row["result"], allowed):
            out.append({"hist": row["hist"], "user": row["user"], "expr": row["target"][5:], "result": row["result"], "reply": line, "type": row["type"]})
    return out


def main():
    random.seed(4)
    rows = [json.loads(l) for l in open("data/router/train_v2aug.jsonl")]
    rows = [r for r in rows if r["target"].startswith("CALC")]
    quota = {"expr": 1500, "followup": 1000}
    if os.environ.get("ONLY_BIG"):          # 1,000以上の結果だけ (カンマバグで落ちていた分)を追加で作る
        rows = [r for r in rows if "," in r["result"]]
        quota = {t: 400 for t in {r["type"] for r in rows}}
    by = {}
    for r in rows: by.setdefault(r["type"], []).append(r)
    pick = []
    for t, v in by.items():
        random.shuffle(v); pick += v[:quota.get(t, 700)]
    print("基本", len(pick), flush=True)
    with ThreadPoolExecutor(local.WORKERS) as ex:
        res = list(ex.map(job, pick))
    out = [x for ps in res for x in ps]
    for r in pick:                                            # 固定文 (約1割)
        if random.random() < 0.1:
            out.append({"hist": r["hist"], "user": r["user"], "expr": r["target"][5:], "result": r["result"],
                        "reply": random.choice(FIXED).format(r=r["result"]), "type": r["type"]})
    seen, final = set(), []
    for x in out:
        k = (x["user"], x["reply"])
        if k not in seen: seen.add(k); final.append(x)
    with open("data/router/answer_train.jsonl", "a" if os.environ.get("ONLY_BIG") else "w", encoding="utf-8") as f:
        for x in final: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("答え文", len(final), "採用率", sum(len(p) for p in res) / (2 * len(pick)), dict(Counter(x["type"] for x in final)))


if __name__ == "__main__":
    main()
