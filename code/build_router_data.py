# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — ルーターMia用の学習データと凍結テストを組み立てる

入力: data/router/para_calc_{train,test}.jsonl, para_chatnum_{train,test}.jsonl (gen_router_para.py)
      会話の否定例: data/dialogues_train_expanded.jsonl (学習) / data/dialogues_val.jsonl (テスト)
出力: data/router/{train,test}.jsonl  1行 = {"hist": [過去の発言...], "user": 今の発言, "target": "CALC 式" or "CHAT", "type": ...}
  - 計算 (CALC): 文章題 (言い換え込み) + 前の発言の数を受ける followup
  - CHAT: 数字を含む会話 (chat_num) + 普通の会話の発言 (履歴つき)。計算の約1.2倍
テストは学習と発言の重複を除く。使い方: python build_router_data.py
"""
import json
import random
from collections import Counter

import gen_router_data as R

import os
D = "data/router"
RV = os.environ.get("RV", "")          # v2 なら RV=v2: para_calc_v2train など
SFX = ("_" + RV) if RV else ""
if os.environ.get("AUG"): SFX += "aug"
ACK = ["That is nice.", "Okay.", "I see.", "Good.", "That sounds good.", "Oh, nice."]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def followups(n, rnd):
    random.seed(rnd)
    out = []
    for _ in range(n):
        (s1, s2, q), e = R.t_followup()
        out.append({"hist": [s1, random.choice(ACK), s2, random.choice(ACK)], "user": q,
                    "target": f"CALC {e}", "result": R.solve(e), "type": "followup"})
    return out


def chat_rows(path, n, seed):
    """普通の会話から、(履歴, 発言) の否定例を n 件。履歴の長さは 0〜全体で散らす。"""
    rnd = random.Random(seed)
    convs = [r["turns"] for r in load(path) if len(r["turns"]) >= 2]
    out = []
    for _ in range(n):
        t = rnd.choice(convs)
        k = rnd.choice(range(0, len(t) - 1, 2))      # user のターン位置 (偶数)
        out.append({"hist": t[:k][-6:], "user": t[k], "target": "CHAT", "type": "chat_plain"})
    return out


def calc_rows(rows):
    return [{"hist": [], "user": r["user"], "target": "CALC " + r["route"][len("<calc> "):], "result": r["result"], "type": r["type"]}
            for r in rows]


def add_history(rows, chat_path, seed, p=0.45, pool=None):
    """履歴のない発言の前に、前のやりとり (別の計算題と固定文の答え / 普通の会話) を1〜2組つける。
    会話の途中で前の問題の数字を混ぜない・前の計算を引きずって chat を calc と誤判断しない、ための学習。"""
    rnd = random.Random(seed)
    math_pool = pool or [r for r in rows if r["target"].startswith("CALC") and not r["hist"]]
    convs = [r["turns"] for r in load(chat_path) if len(r["turns"]) >= 2]
    out = []
    for r in rows:
        if r["hist"] or rnd.random() > p:
            out.append(r); continue
        h = []
        for _ in range(rnd.choice([1, 1, 2])):
            if rnd.random() < 0.7:
                m = rnd.choice(math_pool); h += [m["user"], f"The answer is {m['result']}."]
            else:
                t = rnd.choice(convs); k = rnd.choice(range(0, len(t) - 1, 2)); h += t[k:k + 2]
        out.append({**r, "hist": h[-6:], "aug": 1})
    return out


def build(split, n_follow, chat_path, seed):
    calc = calc_rows(load(f"{D}/para_calc_{RV}{split}.jsonl")) + followups(n_follow, seed)
    num = [{"hist": [], "user": r["user"], "target": "CHAT", "type": "chat_num"} for r in load(f"{D}/para_chatnum_{RV}{split}.jsonl")]
    n_plain = max(int(len(calc) * 1.2) - len(num), 0)
    rows = calc + num + chat_rows(chat_path, n_plain, seed)
    if os.environ.get("AUG"):
        rows = add_history(rows, chat_path, seed + 100)
    random.Random(seed).shuffle(rows)
    return rows


def main():
    tr = build("train", 1500, "data/dialogues_train_expanded.jsonl", 1)
    te = build("test", 150, "data/dialogues_val.jsonl", 2)
    key = lambda r: (r["user"].lower(), tuple(r["hist"]))   # followup は質問文が定型なので履歴も含めて判定
    seen = {key(r) for r in tr}
    te = [r for r in te if key(r) not in seen]
    for name, rows in (("train", tr), ("test", te)):
        with open(f"{D}/{name}{SFX}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), dict(Counter(r["type"] for r in rows)))


if __name__ == "__main__":
    main()
