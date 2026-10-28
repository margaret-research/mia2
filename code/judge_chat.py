# -*- coding: utf-8 -*-
"""会話の返事を DeepSeek に OK/BAD で判定させる (正規表現の合格判定の代わり)。費用上限 COST_CAP(USD)。
入力: eval_results/<tag>.json (eval_chat_only.py / eval_chat.py の出力)。--ids に新モデルの結果を指定すると、同じ問題だけ旧版も判定する。
使い方: COST_CAP=0.3 python judge_chat.py chat_sft m120k
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa as base

PROMPT = """You are checking one reply of a small chatbot named Mia. The chatbot talks with a child about 10 years old.
Conversation so far (the last USER turn is the one being answered):
{conv}

Mia's reply: "{reply}"

Judge ONLY this reply. It is OK if all of these hold:
- It answers or fits what the user just said (not off-topic).
- The English is natural and easy to understand for a child. A short, simple reply is fine.
- It does not claim things a chatbot cannot have (a body, family, eating, going places) as real facts.
- It does not contradict earlier turns. {extra}
Otherwise it is BAD.
Answer with JSON only: {{"verdict": "OK" or "BAD", "reason": "<=12 words"}}"""


def conv_text(turns):
    return "\n".join(("USER: " if i % 2 == 0 else "MIA: ") + t for i, t in enumerate(turns))


def judge(args):
    turns, reply = args
    extra = "If the user asks about something said earlier, the reply must use that earlier information." if len(turns) > 1 else ""
    txt = base.ask(PROMPT.format(conv=conv_text(turns), reply=reply.replace('"', "'"), extra=extra), max_tokens=60, temperature=0.0)
    try:
        r = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        return r.get("verdict") == "OK", r.get("reason", "")
    except ValueError:
        return False, "parse-error: " + txt[:40]


def main(tag, ref_ids_from=None):
    rows = json.load(open(f"eval_results/{tag}.json", encoding="utf-8"))
    if ref_ids_from:
        ids = {r["id"] for r in json.load(open(f"eval_results/{ref_ids_from}.json", encoding="utf-8"))}
        rows = [r for r in rows if r["id"] in ids]
    jobs = [(r["turns"], rep) for r in rows for rep in r["replies"]]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(judge, jobs))
    k = 0
    for r in rows:
        r["judge"] = []
        for _ in r["replies"]:
            r["judge"].append(dict(ok=res[k][0], reason=res[k][1]))
            k += 1
        r["ok_rate"] = sum(j["ok"] for j in r["judge"]) / len(r["judge"])
    json.dump(rows, open(f"eval_results/judge_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


if __name__ == "__main__":
    new, old = sys.argv[1], sys.argv[2:]
    out = {new: main(new)}
    for o in old:
        out[o] = main(o, new)
    cats = list(dict.fromkeys(r["cat"] for r in out[new])) + ["全体"]
    print(f"{'カテゴリ':<10}" + "".join(f"{t:>12}" for t in out))
    for c in cats:
        line = f"{c:<10}"
        for t, rows in out.items():
            rs = [r for r in rows if c == "全体" or r["cat"] == c]
            line += f"{sum(r['ok_rate'] for r in rs) / max(1, len(rs)):>12.2f}"
        print(line)
    print("\n--- 新モデルのBAD (1問につき最初のBADだけ) ---")
    for r in out[new]:
        for rep, j in zip(r["replies"], r["judge"]):
            if not j["ok"]:
                print(f"[{r['cat']}] {' / '.join(r['turns'])}\n     {rep}   <- {j['reason']}")
                break
    print(f"\n費用(見積もり) ${base.cost:.4f}")
