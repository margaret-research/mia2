# -*- coding: utf-8 -*-
"""MIA2 項目3(読解)の試作3: 説明文と質問20問を DeepSeek が1回で書く (オーナー指示 2026-10-25。Qwen+DeepSeek事実確認方式をやめた)

流れ: A. DeepSeek(API・費用上限 COST_CAP)が、題材の説明文(低学年向け)と質問をまとめて書く
      B. プログラムが型ごとに数をそろえる: wh 9 / yes 2 / no 2 / count 3 / unk 4 = 20問(足りない項目は不採用)
      C. Qwen(ローカルvLLM)が各質問と答えを検証し、bad の質問を除く(説明文の中に答えがあるか)
出力: data/reading_entry/raw3_<tag>.jsonl
使い方: COST_CAP=0.3 python gen_entry_qa3.py pilot8 20      (タグ, 項目数) ※先にvLLMサーバーを起動、DEEPSEEK_API_KEY が必要
"""
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa as ds
import gen_chat_qa_local as local
import gen_entry_qa as G
import gen_entry_qa2 as Q2

PROMPT = """You write one short encyclopedia entry for young children, then reading questions about it.
Title of the entry: {title}
The reader is a child in grade 1-2 (6 to 8 years old). Most important rule: use only very easy, everyday words that a 7-year-old knows. No science words or textbook words (not: atmosphere, gravity, temperature, surface, scientists, material, produce, contain); say it in easy words instead. Every sentence is short: at most 12 words. Names of the thing itself (like Mars) are fine, but add no other hard names.
Entry rules:
- {n_words}. Plain paragraphs of short, simple sentences. No lists, no headings, no markdown, no emoji.
- Every fact must be TRUE. Pack in facts: about every sentence gives a new fact about the title (what it is, what it looks like, where it is or lives, what it does, how it works or changes, a size or a count, something surprising). No repeated fact, no filler, no closing summary, no feelings words. Never contradict yourself. State a number or a size only if you are sure it is true.
Questions: write exactly 20 questions about the entry, in this order, with short answers in one full sentence that repeat the words of the question. Every question is clear and complete by itself, in correct easy English, and has exactly one answer that is written in the entry. Use each fact for at most one question, and make the questions different from each other.
- 9 "wh" questions (what, who, where, which, how, why): 3 about the first third of the entry, 3 about the middle third, 3 about the last third.
- 2 "yes" questions whose answer is "Yes, ..." and 2 "no" questions whose answer is "No, ..." (a "no" question says something false according to the entry, and the answer corrects it).
- 3 "count" questions whose answer is a number written in the entry (for example "How many ...?"); the answer sentence contains that number.
- 4 "unk" questions: believable questions about real facts of the title that a curious child could ask, but that the entry does NOT answer anywhere (not even partly), about 4 different things. They assume nothing false. The answer is exactly: I do not know.
Output only one JSON object on one line:
{{"entry": "...", "qas": [{{"type": "wh" or "yes" or "no" or "count" or "unk", "q": "...", "a": "..."}}, ...]}}"""


def one(job):
    i, seed = job
    rnd = random.Random(seed)
    if i == 0:
        domain, title = "space", "Mars"
    else:
        domain = rnd.choice(sorted(G.TOPICS))
        title = rnd.choice(G.TOPICS[domain])
    return build(title, domain, rnd)


def build(title, domain, rnd):
    r = {"title": title, "domain": domain}
    d = Q2.j(ds.ask(PROMPT.format(title=title, n_words=rnd.choice(G.LENGTHS)), max_tokens=3000, temperature=0.7))
    if not d or not isinstance(d.get("entry"), str) or not isinstance(d.get("qas"), list):
        r["fail"] = "parse"; return r
    entry = re.sub(r"\s+", " ", d["entry"].translate(G.MAP)).strip()
    r["entry"] = entry
    qas = [{"type": x.get("type"), "q": str(x.get("q", "")).strip(), "a": str(x.get("a", "")).strip()} for x in d["qas"] if isinstance(x, dict)]
    r["qas_raw"] = len(qas)
    pick = []
    for t, n in Q2.NEED.items():
        got = [x for x in qas if x["type"] == t and x["q"] and x["a"]]
        if t == "yes":
            got = [x for x in got if x["a"].lower().startswith("yes")]
        if t == "no":
            got = [x for x in got if x["a"].lower().startswith("no")]
        if t == "unk":
            got = [x for x in got if x["a"] == "I do not know."]
        if len(got) < n:
            r["fail"] = f"short:{t}"; return r
        pick += got[:n]
    r["qas"] = pick
    r["own_tokens"] = len(local.tok.encode(entry).ids)
    r["split_words"] = sorted({w for w in re.findall(r"[A-Za-z']+", entry) if len(local.tok.encode(" " + w).ids) > local.MAX_PIECES})
    pairs = "\n".join(f"{k + 1}. Q: {x['q']} A: {x['a']}" for k, x in enumerate(pick))
    v = Q2.j("{\"v\": " + local.ask(G.VERIFY.format(entry=entry, pairs=pairs), max_tokens=250, temperature=0.0) + "}")
    r["verdict"] = v["v"] if v and isinstance(v.get("v"), list) and len(v["v"]) == len(pick) else None
    return r


def main(tag, count):
    os.makedirs(G.OUT, exist_ok=True)
    rnd = random.Random(2026)
    jobs = [(i, rnd.randrange(10**9)) for i in range(count)]
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, jobs))
    with open(os.path.join(G.OUT, f"raw3_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in res if "qas" in r]
    print(f"{tag}: 生成 {len(res)} / 質問まで完成 {len(ok)} / 失敗 {[r.get('fail') for r in res if 'fail' in r]}  {time.time() - t0:.0f}s  DeepSeek費用の見積もり ${ds.cost:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
