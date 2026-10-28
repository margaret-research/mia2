# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — カード拡張の追加データ (ルーターv5用)
 card_switch : 履歴は別の話題Aのやりとり、今の質問は話題B(題名を言う) → `card B` (前の話題を引きずらない)
 card_nokb   : 項目にない話題の事実の質問 → `card 話題` (検索で見つからず "I do not know." にする。会話モデルに作り話をさせない)
出力 data/router/card2_{train,test}.jsonl。話題リストはQwenに作らせ、項目の題名(正規化)と重なるものを除く。使い方: python gen_card_data2.py (vLLM起動後)"""
import json, random, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local
from card_stem import stem_words
from gen_card_data import load_entries, mentions

rnd = random.Random(41)
TOPICS = """List {k} different simple things a child in grade 1-2 might ask facts about (for example: week, rainbow, thunder, glue, grandmother, bakery, volcano, whistle). Single common nouns or two-word nouns, lower case, plain ASCII, one per line, no numbering. Theme: {theme}."""
THEMES = ["time and calendar", "weather and sky", "family and people", "places in a town", "things at school", "toys and games", "tools and machines", "clothes", "food and drinks", "animals and pets", "plants and gardens", "body and health", "music and art", "sports", "materials", "feelings and manners", "transport", "the sea and rivers", "farm", "space"]
Q = """Write {k} different factual questions that a child in grade 1-2 might ask about "{t}" (what it is, where, how many, why, who, how it works). Very easy words, plain ASCII, each question names "{t}" (or its plural) explicitly, one per line, no numbering."""


def topic_job(theme):
    try:
        txt = local.ask(TOPICS.format(k=60, theme=theme), max_tokens=500, temperature=0.9)
    except Exception:
        return []
    return [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", l).strip().lower() for l in txt.strip().split("\n") if l.strip()]


def q_job(t):
    try:
        txt = local.ask(Q.format(k=4, t=t), max_tokens=300, temperature=0.8)
    except Exception:
        return []
    out = []
    for l in txt.strip().split("\n"):
        l = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", l).strip().strip('"')
        if l.endswith("?") and l.isascii() and mentions(t, l) and len(l.split()) <= 18:
            out.append({"hist": [], "user": l, "target": f"CARD {t}", "type": "card_nokb", "title": t})
    return out


def main():
    ents, qas = load_entries()
    kb = {" ".join(stem_words(t)) for t in ents}
    val = {e["title"] for e in json.load(open("benchmark/read_val_entries.json"))}
    with ThreadPoolExecutor(local.WORKERS) as ex:
        tops = [t for ts in ex.map(topic_job, THEMES * 2) for t in ts]
    tops = sorted({t for t in tops if t.isascii() and 1 <= len(t.split()) <= 2 and " ".join(stem_words(t)) not in kb})
    rnd.shuffle(tops); test_t, train_t = tops[:60], tops[60:]
    print("項目にない話題", len(tops), "例", tops[:12], flush=True)
    rows = {"train": [], "test": []}
    with ThreadPoolExecutor(local.WORKERS) as ex:
        for split, ts in (("train", train_t), ("test", test_t)):
            for ps in ex.map(q_job, ts): rows[split] += ps
    # card_switch: 履歴は別の項目 A の質問と答え、今は項目 B の質問
    cj = json.load(open("data/router/card_train.jsonl".replace("train", "train"), encoding="utf-8")) if False else None
    for split in rows:
        cards = [json.loads(l) for l in open(f"data/router/card_{split}.jsonl", encoding="utf-8")]
        explicit = [c for c in cards if c["type"] == "card_q"]
        for _ in range(3000 if split == "train" else 300):
            b = rnd.choice(explicit); a = rnd.choice(explicit)
            if a["title"] == b["title"]: continue
            ans = next((x["a"] for x in qas[a["title"]] if x["q"] == a["user"]), None)
            if not ans or ans == "I do not know.": continue
            rows[split].append({"hist": [a["user"], ans], "user": b["user"], "target": b["target"], "type": "card_switch", "title": b["title"]})
    for split in rows:
        with open(f"data/router/card2_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[split]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows[split]), dict(Counter(r["type"] for r in rows[split])))


if __name__ == "__main__":
    main()
