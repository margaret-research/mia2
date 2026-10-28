# -*- coding: utf-8 -*-
"""上限の目安: 会話モデルの36問(eval_chat_only の結果と同じ問題)を Qwen に Mia として答えさせ、eval_results/qwen.json に保存する
"""
import json
import sys

import gen_chat_qa_local as local

SYS = ("You are Mia, a friendly chat helper that talks with a child about 10 years old. Reply with 1-2 short, simple sentences using words a 10-year-old knows. "
       "Plain text only. You are a chat helper without a body: do not claim to eat, sleep, have family or go places. If you do not know something, say so kindly.")


def reply(turns_so_far):
    conv = "\n".join(("USER: " if i % 2 == 0 else "MIA: ") + t for i, t in enumerate(turns_so_far))
    p = f"{SYS}\n\nConversation so far:\n{conv}\n\nWrite only Mia's next reply (no label)."
    return local.ask(p, max_tokens=80, temperature=0.7).strip().strip('"')


ref = json.load(open("eval_results/chat_sft.json", encoding="utf-8"))
out = []
for r in ref:
    reps = []
    for _ in range(3):
        hist = []
        for u in r["turns"]:
            hist.append(u)
            rp = reply(hist)
            hist.append(rp)
        reps.append(rp)
    out.append(dict(id=r["id"], cat=r["cat"], turns=r["turns"], ok=r["ok"], replies=reps))
json.dump(out, open("eval_results/qwen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "問 保存", "(ローカル・無料)")
