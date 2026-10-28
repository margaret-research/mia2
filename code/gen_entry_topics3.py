# -*- coding: utf-8 -*-
"""MIA2 項目3(読解): 題材リストの追加(オーナー承認 2026-10-25「Aでお願いします」)
既存の topics_themes_v2.json にない題材を、観点を16個に増やして Qwen に挙げさせ、同じ条件(ok(): 語彙・辞書・頻度)と判定(judge)をかけて足す。
出力: data/reading_entry/topics_add.json (テーマ→追加題材) / topics_add.txt。既存のファイルは変えない。
使い方: python gen_entry_topics3.py  (先にvLLMサーバーを起動 / PYTHONPATH に nltk と wordfreq の場所)"""
import json, os, re
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local
import gen_entry_topics as T

ANGLES2 = ["names of animals or things that start with the letters A to F", "names that start with the letters G to M", "names that start with the letters N to S", "names that start with the letters T to Z",
           "things found in a story book for children", "things found in a picture dictionary for children", "things that children learn about in class in grades 1 to 3", "things seen on a trip or a holiday",
           "things found in a house, a garden, a farm, a zoo, or a shop", "things that are very old, or from long ago, that children still know", "things that people make or build",
           "things that live or grow in hot places, cold places, water, or forests", "things that help or hurt people", "parts of things, or groups of things, that children can name",
           "things that make a sound, move, or change", "things children ask 'why' or 'how' about"]
old = json.load(open(os.path.join(T.OUT, "topics_themes_v2.json")))
seen = {t for v in old.values() for t in v}
ADD = os.path.join(T.OUT, "topics_add.json")
prev = json.load(open(ADD)) if os.path.exists(ADD) else {t: [] for t in T.THEMES}   # 前回までの追加分(何度回しても足していく)
seen |= {t for v in prev.values() for t in v}
rem = set()
if os.path.exists(os.path.join(T.OUT, "topics_themes_v2.txt")):
    rem = {x.strip() for x in open(os.path.join(T.OUT, "topics_themes_v2.txt"), encoding="utf-8").read().split("## 消した題材")[-1].split("\n", 1)[-1].split(",")}


def main():
    with ThreadPoolExecutor(local.WORKERS) as ex:
        res = list(ex.map(T.job, [(t, a) for t in T.THEMES for a in ANGLES2]))
    new, drop = {t: [] for t in T.THEMES}, 0
    for theme, items in res:
        for it in items:
            t = re.sub(r"\s+", " ", it.lower()).strip()
            if not t or t in seen or t in rem:
                continue
            if not T.ok(t):
                drop += 1; continue
            seen.add(t); new[theme].append(t)
    before = sum(len(v) for v in new.values())
    jobs = [(t, v[i:i + 80]) for t, v in new.items() for i in range(0, len(v), 80)]
    with ThreadPoolExecutor(local.WORKERS) as ex:
        judged = list(ex.map(T.judge, jobs))
    new = {t: list(prev[t]) for t in T.THEMES}
    for theme, items in judged:
        new[theme] += items
    json.dump(new, open(os.path.join(T.OUT, "topics_add.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(T.OUT, "topics_add.txt"), "w", encoding="utf-8") as f:
        for t, v in new.items():
            f.write(f"## {t} ({len(v)})\n" + ", ".join(v) + "\n\n")
    print(f"既存にない候補 {before} (語彙で除外 {drop}) → 判定後 {sum(len(v) for v in new.values())}")
    for t in T.THEMES:
        print(f"{t}: {len(new[t])}")


if __name__ == "__main__":
    main()
