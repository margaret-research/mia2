# -*- coding: utf-8 -*-
"""MIA2 項目3(読解)の試作2: 説明文を DeepSeek が事実確認してから、質問を作る (オーナー指示 2026-10-25)

流れ: A. Qwen(ローカルvLLM)が題材の説明文だけを書く
      B. DeepSeek(API・費用上限 COST_CAP)が説明文を事実確認し、間違い・怪しい文を直す(直せない文は削除)。言葉のやさしさは保つ
      C. Qwen が、直した説明文から質問を作る。プログラムで型ごとに数をそろえる: wh 9 / yes 2 / no 2 / count 3 / unk 4 = 20問(足りない項目は不採用)
      D. Qwen が各質問と答えを検証し、bad の質問を除く
送るのはQwenが作った説明文だけ(個人情報なし)。DeepSeekの出力は信頼できない外部由来として、整形と保存だけ。
出力: data/reading_entry/raw2_<tag>.jsonl  (entry_orig=Qwenの原文, entry=直した文, issues=DeepSeekが見つけた問題, qas, verdict)
使い方: COST_CAP=0.3 python gen_entry_qa2.py pilot5 20      (タグ, 項目数) ※先にvLLMサーバーを起動、DEEPSEEK_API_KEY が必要
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

OUT = G.OUT
NEED = {"wh": 9, "yes": 2, "no": 2, "count": 3, "unk": 4}

ENTRY_PROMPT = """You write one short encyclopedia entry for children.
Title of the entry: {title}
Write the entry for a child in grade 1-2 (6 to 8 years old). This is the most important rule: use only very easy, everyday words that a 7-year-old knows and uses. Never use a science word, a long word, or a word from a textbook (for example: atmosphere, gravity, temperature, surface, scientists, material, produce, contain). Say it with easy words instead. Every sentence is short: at most 12 words. Names of the thing itself (like Mars) are fine, but do not add other hard names. Facts must be true and simple.
- {n_words}. Plain paragraphs of very short, simple sentences. No lists, no headings, no markdown, no emoji.
- Pack in facts: about every sentence must give a new fact about the title (what it is, what it looks like, where it is, how it works or changes, how big or how many, something surprising). Do not repeat a fact. No filler sentences, no summary, no closing sentence, no feelings or "fascinating"-type words. Never contradict yourself. Do not state a number unless you are sure it is true.
Output only the entry text, nothing else."""

CHECK_PROMPT = """You are a fact-checker for a children's encyclopedia (readers are 6 to 8 years old). The text is deliberately simple, so simplifications are fine.
Title: {title}
Numbered sentences:
{numbered}

Mark a sentence ONLY if it is clearly FALSE (a wrong fact, a wrong number, a wrong name, a wrong comparison), or it clearly contradicts another sentence.
If you are not sure a sentence is false, do NOT mark it. Before you write a FIX, check that the fix itself is true.
Do NOT mark a sentence just because it is simple, vague, "not always true", "depends", or because you cannot be sure. Do NOT mark style problems. Most entries have only a few problems; many have none.
For each marked sentence give a FIX: one corrected sentence in the same very easy words (a 7-year-old can read it, at most 12 words), or null to delete it only if it cannot be fixed.
Output only one JSON object on one line (the numbers refer to the numbered sentences):
{{"issues": [{{"n": 3, "problem": "<=12 words", "fix": "..." or null}}]}}
If there is no problem, output {{"issues": []}}."""


def sentences(text):
    return [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]


QA_PROMPT = """Below is a short encyclopedia entry for children. Write exactly 20 questions about it, in this order, with short answers in one full sentence that repeat the words of the question. Every question must be clear and complete by itself, in correct easy English, and have exactly one answer that is written in the entry. Use each fact of the entry for at most one question (no two questions about the same sentence), and make the questions different from each other.
- 9 "wh" questions (what, who, where, which, how, why): 3 about the first third of the entry, 3 about the middle third, 3 about the last third.
- 2 "yes" questions whose answer is "Yes, ..." and 2 "no" questions whose answer is "No, ..." (a "no" question says something that is false according to the entry, and the answer corrects it).
- 3 "count" questions whose answer is a number written in the entry (for example "How many ...?"); the answer sentence must contain that number.
- 4 "unk" questions: believable questions about real facts of the title that a curious child could ask, but that this entry does NOT answer anywhere (not even partly), about 4 different things. They must not assume anything false about the title. The answer is exactly: I do not know.
Title: {title}
Entry:
{entry}
Output only one JSON object on one line:
{{"qas": [{{"type": "wh" or "yes" or "no" or "count" or "unk", "q": "...", "a": "..."}}, ...]}}"""


def j(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0).translate(G.MAP)) if m else None
    except Exception:
        return None


def one(job):
    i, seed = job
    rnd = random.Random(seed)
    if i == 0:
        domain, title = "space", "Mars"
    else:
        domain = rnd.choice(sorted(G.TOPICS))
        title = rnd.choice(G.TOPICS[domain])
    n_words = rnd.choice(G.LENGTHS)
    r = {"title": title, "domain": domain}
    entry0 = re.sub(r"\s+", " ", local.ask(ENTRY_PROMPT.format(title=title, n_words=n_words), max_tokens=700, temperature=0.8).translate(G.MAP)).strip()
    r["entry_orig"] = entry0
    sents = sentences(entry0)
    d = j(ds.ask(CHECK_PROMPT.format(title=title, numbered="\n".join(f"{k + 1}. {x}" for k, x in enumerate(sents))), max_tokens=1000, temperature=0.0))
    if not d or not isinstance(d.get("issues"), list):
        r["fail"] = "check-parse"; return r
    r["issues"], fix, dele = [], {}, set()
    for x in d["issues"]:
        if not isinstance(x, dict) or not isinstance(x.get("n"), int) or not 1 <= x["n"] <= len(sents):
            continue
        r["issues"].append({"sentence": sents[x["n"] - 1], "problem": x.get("problem"), "fix": x.get("fix")})
        if isinstance(x.get("fix"), str) and x["fix"].strip():
            fix[x["n"] - 1] = x["fix"].strip().translate(G.MAP)
        else:
            dele.add(x["n"] - 1)
    if len(dele) > 0.15 * len(sents):
        r["fail"] = "too-many-deleted"; return r
    entry = " ".join(fix.get(k, x) for k, x in enumerate(sents) if k not in dele)
    r["entry"] = entry
    q = j(local.ask(QA_PROMPT.format(title=title, entry=entry), max_tokens=2600, temperature=0.7))
    if not q or not isinstance(q.get("qas"), list):
        r["fail"] = "qa-parse"; return r
    qas = [{"type": x.get("type"), "q": str(x.get("q", "")).strip(), "a": str(x.get("a", "")).strip()} for x in q["qas"] if isinstance(x, dict)]
    r["qas_raw"] = len(qas)
    pick = []
    for t, n in NEED.items():
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
    v = j("{\"v\": " + local.ask(G.VERIFY.format(entry=entry, pairs=pairs), max_tokens=250, temperature=0.0) + "}")
    r["verdict"] = v["v"] if v and isinstance(v.get("v"), list) and len(v["v"]) == len(pick) else None
    return r


def main(tag, count):
    os.makedirs(OUT, exist_ok=True)
    rnd = random.Random(2026)
    jobs = [(i, rnd.randrange(10**9)) for i in range(count)]
    t0 = time.time()
    with ThreadPoolExecutor(local.WORKERS) as ex:
        res = list(ex.map(one, jobs))
    with open(os.path.join(OUT, f"raw2_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in res if "qas" in r]
    print(f"{tag}: 生成 {len(res)} / 質問まで完成 {len(ok)} / 失敗 {[r.get('fail') for r in res if 'fail' in r]}  {time.time() - t0:.0f}s  DeepSeek費用の見積もり ${ds.cost:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
