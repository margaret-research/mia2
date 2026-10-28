# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — ルーターMia用: 型ごとの文章題をQwenに言い換えさせる (ローカル、費用なし)

流れ: gen_router_data の型から基本問題を作る → Qwen が数字をそのままに別の言い方で3通り書く
      → プログラムで検査 (数の集まりが元と同じ / 英語のASCIIのみ / 重複なし)。式と答えは元のまま (検算済み)。
また、数字を含むが計算ではない発言 (chat_num) もQwenに書かせる。
出力: data/router/para_calc.jsonl, data/router/para_chatnum.jsonl
使い方: python gen_router_para.py 基本問題数 [chatnum件数]   ※先にvLLM(Qwen)を起動
"""
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa_local as local
import gen_router_data as R
from calc_tool import normalize, is_num

OUT = "data/router"
PROMPT = """Rewrite this math word problem for a child in grade 1-2 in {k} different ways.
Problem: {u}
Rules:
- Keep every number and the question the same in meaning. The answer must not change. Do not add or drop any number.
- Change the words, the people, the things and the order of the sentences. Use only very easy words a 7-year-old knows. Short sentences, plain ASCII, no markdown.
- Style hints: {styles}
- Do not write the answer. End with a question.
Output {k} lines only, one rewritten problem per line, no numbering."""
STYLES = ["a short story with a name", "very short and plain", "first sentence gives a fact, then the question", "the question comes last, with 'altogether' or 'in all' or 'left' as needed",
          "use a different thing (not the same noun)", "ask in a friendly way, like talking to Mia"]


def nums_of(text):
    """文中の数 (数字と英単語の数詞) の多重集合。"""
    toks = normalize(text, True)
    return Counter(t for t in toks if is_num(t))


SOLVE = """Solve this problem for a child. Think step by step briefly, then write the final answer as a single number on the last line in the form: ANSWER: <number>
Problem: {u}"""


def solves_to(u, want):
    """Qwen が言い換え後の問題を独立に解き、答えが元と一致するか (意味が変わった言い換えを落とす)。"""
    try:
        txt = local.ask(SOLVE.format(u=u), max_tokens=400, temperature=0.0)
    except Exception:
        return False
    m = re.findall(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)", txt)
    return bool(m) and m[-1].replace(",", "") == want.replace(",", "")


def para_job(job):
    base = job
    if base["type"] == "expr":
        return []          # 式だけの型は言い換えない
    styles = "; ".join(random.sample(STYLES, 3))
    try:
        txt = local.ask(PROMPT.format(k=3, u=base["user"], styles=styles), max_tokens=500, temperature=0.9)
    except Exception as e:
        return []
    out = []
    want = nums_of(base["user"])
    for line in txt.strip().split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not line or not line.isascii() or not line.endswith("?") or len(line.split()) > 45:
            continue
        if nums_of(line) != want:
            continue
        if not solves_to(line, base["result"]):
            continue
        out.append({**base, "user": line, "src": "para"})
    return out


CHATNUM_PROMPT = """Write {k} different short things a child might say to a friendly chat helper. Every one contains a number (as a digit or a number word), but it is NOT a math problem and asks for no calculation.
Examples: "I am 7 years old." / "My room is number 12." / "I saw 4 birds today." / "Do you like the number 8?" / "My bus comes at 8 o'clock."
Rules: very easy words, plain ASCII, 1-2 short sentences, different topics (age, time, address, favorite number, a count of things told as a fact, a date, a score). Some are questions to the helper. Never ask to add, subtract, multiply, divide, share or count in total.
Output {k} lines only, one per line, no numbering."""


def chatnum_job(_):
    try:
        txt = local.ask(CHATNUM_PROMPT.format(k=10), max_tokens=500, temperature=1.0)
    except Exception:
        return []
    out = []
    for line in txt.strip().split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
        if line and line.isascii() and len(line.split()) <= 25 and nums_of(line):
            out.append({"user": line, "route": "<chat>", "type": "chat_num", "src": "para"})
    return out


def main(n_base, n_chatnum):
    os.makedirs(OUT, exist_ok=True)
    random.seed(int(os.environ.get('SEED', 11)))
    tag = os.environ.get('TAG', '')
    bases = []
    for _ in range(n_base):
        f = random.choice(R.CALC_TYPES)
        u, e = f()
        bases.append({"user": u, "route": f"<calc> {e}", "result": R.solve(e), "type": f.__name__[2:]})
    with ThreadPoolExecutor(local.WORKERS) as ex:
        res = list(ex.map(para_job, bases))
    seen, rows = set(), []
    for b, ps in zip(bases, res):
        for r in [{**b, "src": "tmpl"}] + ps:
            k = r["user"].lower()
            if k not in seen:
                seen.add(k)
                rows.append(r)
    with open(f"{OUT}/para_calc{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("calc", len(rows), dict(Counter(r["type"] for r in rows)), "para ok", sum(len(p) for p in res), "of", 3 * n_base)
    if n_chatnum:
        with ThreadPoolExecutor(local.WORKERS) as ex:
            res = list(ex.map(chatnum_job, range(n_chatnum // 10)))
        seen, rows = set(), []
        for ps in res:
            for r in ps:
                if r["user"].lower() not in seen:
                    seen.add(r["user"].lower())
                    rows.append(r)
        with open(f"{OUT}/para_chatnum{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("chatnum", len(rows))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
