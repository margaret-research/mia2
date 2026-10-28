# -*- coding: utf-8 -*-
"""MIA2 項目3(読解)の試作: 「タイトル + 200〜500トークンの項目 + 質問と答え」を Qwen(ローカルvLLM)で作る

1回の呼び出しで1項目(題材ごと)。題材は data/reading_qa/topics_merged.json(23分野)から。
項目は10歳が知っている言葉だけで書かせる。質問は20問: wh 9 / yes-no 4(Yes 2・No 2) / 数・比較 3 / 項目に書いていないこと 4(答えは「I do not know.」)。
出力: data/reading_entry/raw_<tag>.jsonl  {"title","domain","entry","qas":[{"type","q","a"}], "own_tokens", "split_words":[...]}
使い方: python gen_entry_qa.py pilot 20      (タグ, 項目数) ※先にvLLMサーバーを起動
生成テキストは信頼できない外部由来として扱う(整形と保存だけ)。
"""
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa_local as local

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "reading_entry")
TOPICS = json.load(open(os.path.join(HERE, "data", "reading_entry", "topics_themes_v2.json"), encoding="utf-8"))
MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "})

PROMPT = """You write one short encyclopedia entry for children, then reading questions about it.
Title of the entry: {title}
Write the entry for a child in grade 1-2 (6 to 8 years old). This is the most important rule: use only very easy, everyday words that a 7-year-old knows and uses. Never use a science word, a long word, or a word from a textbook (for example: atmosphere, gravity, temperature, surface, scientists, material, produce, contain). Say it with easy words instead (for example: "the air around it", "it pulls things down", "how hot or cold it is", "the top part", "people who study it", "made of", "make", "have"). Every sentence is short: at most 12 words. Names of the thing itself (like Mars) are fine, but do not add other hard names. Facts must be true and simple.
Rules for the entry:
- {n_words}. Plain paragraphs of very short, simple sentences. No lists, no headings, no markdown, no emoji.
- Pack in facts: about every sentence must give a new fact about the title (what it is, what it looks like, where it is, how it works or changes, how big or how many, something surprising). Do not repeat a fact. No filler sentences, no summary, no closing sentence, no feelings or "fascinating"-type words. Never contradict yourself. Do not state a number unless you are sure it is true.
Then write exactly 20 questions about the entry, in this order, with short answers in one full sentence that repeat the words of the question. Every question must be clear and complete by itself, in correct easy English, and have exactly one answer that is written in the entry. Use each fact of the entry for at most one question (no two questions about the same sentence), and make the questions different from each other:
- 9 "wh" questions (what, who, where, which, how, why): 3 about the first third of the entry, 3 about the middle third, 3 about the last third.
- 4 "yes" questions (yes/no): exactly 2 have the answer "Yes, ..." and 2 have the answer "No, ..." (a "No" question says something that is false according to the entry, and the answer corrects it).
- 3 "count" questions whose answer is a number or a comparison written in the entry.
- 4 "unk" questions: believable questions about real facts of the title that a curious child could ask, but that this entry does NOT answer anywhere (not even partly), about 4 different things. They must not assume anything false about the title. Good example for an entry on Mars: "How long is one day on Mars?" when the entry never says it. The answer is exactly: I do not know.
Output only one JSON object on one line, nothing else:
{{"entry": "...", "qas": [{{"type": "wh", "q": "...", "a": "..."}}, ...]}}"""

LENGTHS = ["About 170 to 220 words", "About 220 to 270 words", "About 270 to 320 words"]


def one(job):
    i, seed = job
    rnd = random.Random(seed)
    if i == 0:
        domain, title = "space", "Mars"
    else:
        domain = rnd.choice(sorted(TOPICS))
        title = rnd.choice(TOPICS[domain])
    n_words = rnd.choice(LENGTHS)
    txt = local.ask(PROMPT.format(title=title, n_words=n_words), max_tokens=2600, temperature=0.8)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0).translate(MAP)) if m else None
    except Exception:
        d = None
    return {"title": title, "domain": domain, "n_words": n_words, "obj": d, "text": txt}


VERIFY = """Below are an entry and some question/answer pairs. For each pair, decide:
- For a pair whose answer is "I do not know.": "ok" only if the entry really does NOT answer the question (not even partly) and the question assumes nothing false; otherwise "bad".
- For every other pair: "ok" only if the question is clear, easy, correct English AND the entry clearly states the answer AND the given answer is correct according to the entry; otherwise "bad".
ENTRY: {entry}
PAIRS:
{pairs}
Output only one JSON list of "ok"/"bad" strings, one per pair, in order. Nothing else."""


def verify(c):
    pairs = "\n".join(f"{i + 1}. Q: {q['q']} A: {q['a']}" for i, q in enumerate(c["qas"]))
    txt = local.ask(VERIFY.format(entry=c["entry"], pairs=pairs), max_tokens=250, temperature=0.0)
    m = re.search(r"\[.*?\]", txt, re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except Exception:
        v = None
    c["verdict"] = v if isinstance(v, list) and len(v) == len(c["qas"]) else None
    return c


def check(r):
    d = r["obj"]
    if not d or not isinstance(d.get("entry"), str) or not isinstance(d.get("qas"), list):
        return None
    entry = re.sub(r"\s+", " ", d["entry"].translate(MAP)).strip()
    qas = [{"type": q.get("type"), "q": q.get("q", "").strip(), "a": q.get("a", "").strip()} for q in d["qas"] if isinstance(q, dict)]
    ids = local.tok.encode(entry).ids
    words = re.findall(r"[A-Za-z']+", entry)
    split = sorted({w for w in words if len(local.tok.encode(" " + w).ids) > local.MAX_PIECES})
    return {"title": r["title"], "domain": r["domain"], "entry": entry, "qas": qas, "own_tokens": len(ids), "split_words": split,
            "q_types": [q["type"] for q in qas]}


def main(tag, count):
    os.makedirs(OUT, exist_ok=True)
    rnd = random.Random(2026)
    jobs = [(i, rnd.randrange(10**9)) for i in range(count)]
    t0 = time.time()
    with ThreadPoolExecutor(local.WORKERS) as ex:
        res = list(ex.map(one, jobs))
    ok, bad = [], 0
    cs = [c for c in (check(r) for r in res) if c]
    with ThreadPoolExecutor(local.WORKERS) as ex:
        cs = list(ex.map(verify, cs))
    res = cs
    with open(os.path.join(OUT, f"raw_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            ok.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad = count - len(ok)
    print(f"{tag}: 生成 {len(res)} / 読み取れた {len(ok)} / 失敗 {bad}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
