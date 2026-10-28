# -*- coding: utf-8 -*-
"""MIA2 項目3(読解): テーマごとに、小学校低学年が知っている「題材」を Qwen(ローカルvLLM)で挙げさせ、整えてリストにする

テーマ12個 × 呼び出し4回(観点を変える)。整え方: 小文字・空白の正規化、重複除去、英字と空白だけ、1〜3語、
  自作トークナイザーで1語が3トークン以上に割れる語を含む題材は除く(語彙になじまない)。
出力: data/reading_entry/topics_themes.json (テーマ→題材のリスト)、topics_themes.txt (人が読む用)
使い方: python gen_entry_topics.py   (先にvLLMサーバーを起動)
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import nltk
# WordNetが必要: 初回に python -c "import nltk; nltk.download('wordnet')" を実行
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

import gen_chat_qa_local as local

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "reading_entry")
THEMES = {
    "animals": "animals (pets, farm animals, wild animals, sea animals, birds)",
    "insects": "insects and tiny creatures (bugs, spiders, worms, butterflies)",
    "plants": "plants (trees, flowers, fruits and vegetables as plants, grass, seeds)",
    "earth": "the Earth (land, water, rocks, volcanoes, rivers, mountains, oceans, and things that happen on the Earth)",
    "weather": "weather and seasons (rain, snow, wind, clouds, summer, winter, rainbow)",
    "space": "space (the Sun, the Moon, planets, stars, rockets, astronauts)",
    "body": "the human body (parts of the body, the senses, sleeping, growing, teeth)",
    "food": "food and drinks (things children eat and drink, and where they come from)",
    "vehicles": "vehicles and ways to travel (cars, trains, boats, planes, bikes)",
    "home": "things at home and things we use every day (rooms, furniture, tools, clothes, toys)",
    "school_work": "school and jobs (things in a classroom, people who help us, jobs)",
    "play": "games, sports and play (ball games, playground, music, drawing)",
}
ANGLES = ["the most common and famous ones", "less common ones that a 7-year-old still knows", "ones a child sees at home, at school, or in the park", "ones children find fun or surprising",
          "ones that are big, strong, or fast", "ones that are small", "ones that are useful to people", "ones from other countries or far places that children still learn about in class"]
PROMPT = """List 50 different {kind} that a child in grade 1-2 (6 to 8 years old) knows and can ask about.
Angle: {angle}.
Stay strictly inside the theme. Rules: each item is the plain NAME of a kind of thing: one common noun, or a fixed two-word name (for example: "frog", "fire truck", "ice cream"). NEVER put an adjective, a color, a size, or a feeling in front (not "red ant", "big ferry", "quiet moth"). No names of people, places or brands. No hard or rare words. All different from each other.
Output only the items, one per line, no numbers, no extra text."""


def job(a):
    theme, angle = a
    txt = local.ask(PROMPT.format(kind=THEMES[theme], angle=angle), max_tokens=500, temperature=1.0)
    return theme, [l.strip().strip("-*•").strip() for l in txt.splitlines()]


JUDGE = """Theme: {kind}
Below is a numbered list of candidate words. Keep a word ONLY IF all of these are true:
- it is a noun (a thing, animal, plant, place, person, body part, food, tool, or event) or a noun phrase, not an adjective, verb, or feeling;
- it clearly belongs to the theme above (an animal is not an "earth" or "space" word, a machine is not a "body" word, and so on);
- a child in grade 1-2 (6 to 8 years old) already knows what it is;
- it is a real thing (not magic, not a made-up thing), and it is not just a color + animal combination like "red ant" (real kinds like "ladybug" are fine).
Output only the numbers of the words to keep, as a JSON list of integers. Nothing else.
{lines}"""


def judge(a):
    theme, items = a
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))
    txt = local.ask(JUDGE.format(kind=THEMES[theme], lines=lines), max_tokens=700, temperature=0.0)
    m = re.search(r"\[.*?\]", txt, re.S)
    try:
        keep = {int(x) for x in json.loads(m.group(0))}
    except Exception:
        keep = set()
    return theme, [t for i, t in enumerate(items) if i + 1 in keep]


PIECES = int(os.environ.get("TOPIC_PIECES", 4))     # 1語がこれより多く割れる語を含む題材は除く(オーナー指示 2026-10-25: 2→4に緩めた)


def ok(t):
    if not re.fullmatch(r"[a-z]+(?: [a-z]+)?", t):
        return False
    if " " in t and not wn.synsets(t.replace(" ", "_")):      # 2語は WordNet に載っている決まった名前だけ(形容詞+名詞の作り物を除く)
        return False
    if " " not in t and zipf_frequency(t, "en") < 2.3:
        return False
    return all(len(local.tok.encode(" " + w).ids) <= PIECES for w in t.split())


def main():
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(local.WORKERS) as ex:
        res = list(ex.map(job, [(t, a) for t in THEMES for a in ANGLES]))
    seen, topics, drop = {}, {t: [] for t in THEMES}, {t: 0 for t in THEMES}
    for theme, items in res:
        for it in items:
            t = re.sub(r"\s+", " ", it.lower()).strip()
            if not t:
                continue
            if not ok(t):
                drop[theme] += 1; continue
            if t in seen:
                continue
            seen[t] = theme; topics[theme].append(t)
    before = {t: len(v) for t, v in topics.items()}
    jobs = [(t, v[i:i + 80]) for t, v in topics.items() for i in range(0, len(v), 80)]
    with ThreadPoolExecutor(local.WORKERS) as ex:
        judged = list(ex.map(judge, jobs))
    topics = {t: [] for t in THEMES}
    for theme, items in judged:
        topics[theme] += items
    json.dump(topics, open(os.path.join(OUT, "topics_themes.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "topics_themes.txt"), "w", encoding="utf-8") as f:
        for t, v in topics.items():
            f.write(f"## {t} ({len(v)})\n" + ", ".join(v) + "\n\n")
    for t in THEMES:
        print(f"{t}: 判定前 {before[t]} → {len(topics[t])} 題材 (語彙で除外 {drop[t]})")
    print("合計", sum(len(v) for v in topics.values()))


if __name__ == "__main__":
    main()
