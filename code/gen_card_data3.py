# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — ルーターMia v6 用の追加データ (テンプレートのみ・費用なし)
オーナーの実機テストで見つかった弱点を直す:
 ・雑談が続いたあとだと、整った事実の質問も chat になる          → card_ctx (雑談の履歴 + 事実の質問 → card)
 ・「You know what is Sun?」「Tell me more about Moon」など くだけた言い方 → card_casual
 ・「grass hopper」のように分けて書いた題名                      → card_split (card grasshopper)
 ・好みを聞く/気持ちの発言は今までどおり chat                     → chat_like / chat_ctx
学習/テストの題名は gen_card_data.py と同じ分割 (card_{train,test}.jsonl に出る題名)。テストの言い方 (CASUAL_TEST) は学習 (CASUAL_TRAIN) と別。
出力 data/router/card3_{train,test}.jsonl。使い方: python gen_card_data3.py"""
import json, random
from collections import Counter

rnd = random.Random(61)
ART = lambda t: "an" if t[0] in "aeiou" else "a"
CASUAL_TRAIN = ["You know what is {t}?", "You know {t}?", "You know some {t}?", "Tell me more about {t}", "Tell me more about the {t}.", "I want to know {t}.",
                "Can you explain {t}?", "What is the {t}?", "Hey Mia, what is {a} {t}?", "oh, what is {t}?", "Do you know what {a} {t} is?", "what is {t}", "Tell me about the {t}",
                "{T}?", "Mia, tell me about {t}.", "Hmm, what is {a} {t}?", "You know about {t}?", "Please explain {t}.", "What does {t} mean?", "What is {t} like?"]
CASUAL_TEST = ["Do you know {t}?", "Explain {t} to me.", "You know what {a} {t} is?", "Tell me what is {t}.", "Can you tell me more about {t}?", "Say something about the {t}.", "What is {a} {t}, Mia?"]
LIKE = ["You like {t}?", "Do you like {t}?", "Do you like the {t}?", "I like {t}.", "I love {t}!", "Do you want {a} {t}?", "I saw {a} {t} today.", "Is {t} your favorite?", "I want {a} {t}.", "You like {t}, right?"]
ACK = ["Yes, it is nice.", "I think so.", "That is fun.", "I like it too.", "That sounds good.", "Oh, nice.", "I see."]


def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]


def chat_hist(convs):
    t = rnd.choice(convs); k = rnd.choice(range(0, len(t) - 1, 2)); n = rnd.choice([2, 2, 4, 4, 6])
    h = t[k:k + n]
    return h if len(h) % 2 == 0 else h[:-1]


def split_word(t, vocab):
    """1語の題名を2つの語に分けた書き方 (grasshopper → grass hopper)。両方の半分が語として知られているときだけ。"""
    if " " in t or len(t) < 7: return None
    c = [t[:i] + " " + t[i:] for i in range(3, len(t) - 2) if t[:i] in vocab and t[i:] in vocab]
    return rnd.choice(c) if c else None


def build(split, convs, vocab):
    cards = load(f"data/router/card_{split}.jsonl")
    titles = sorted({c["title"] for c in cards})
    qs = [c for c in cards if c["type"] in ("card_q", "card_ask")]
    negs = [c for c in cards if c["type"] == "card_neg"]
    forms = CASUAL_TRAIN if split == "train" else CASUAL_TEST
    n_ctx, n_neg = (7000, 3000) if split == "train" else (500, 250)
    rows = []
    fm = lambda f, t: f.format(t=t, a=ART(t), T=t.capitalize())
    # card_ctx: 雑談の履歴 + (整った質問 or くだけた言い方) → card
    for i in range(n_ctx):
        t = rnd.choice(titles)
        if rnd.random() < 0.5:
            u = rnd.choice([q for q in qs if q["title"] == t])["user"] if rnd.random() < 0.9 else fm(rnd.choice(forms), t)
        else:
            u = fm(rnd.choice(forms), t)
        rows.append({"hist": chat_hist(convs), "user": u, "target": f"CARD {t}", "type": "card_ctx", "title": t})
    # card_casual: 履歴なしのくだけた言い方
    for t in titles:
        for f in rnd.sample(forms, min(3 if split == "train" else 2, len(forms))):
            rows.append({"hist": [], "user": fm(f, t), "target": f"CARD {t}", "type": "card_casual", "title": t})
    # 同じ話題の好みの発言のあとに、もっと知りたい → card
    for _ in range(1500 if split == "train" else 150):
        t = rnd.choice(titles)
        rows.append({"hist": [fm(rnd.choice(LIKE[:3]), t), rnd.choice(ACK)], "user": fm(rnd.choice(forms), t), "target": f"CARD {t}", "type": "card_samecontext", "title": t})
    # chat: 好み・気持ちの短い発言 (履歴なし/雑談つき)
    for _ in range(3500 if split == "train" else 300):
        t = rnd.choice(titles); h = chat_hist(convs) if rnd.random() < 0.5 else []
        rows.append({"hist": h, "user": fm(rnd.choice(LIKE), t), "target": "CHAT", "type": "chat_like", "title": t})
    for _ in range(n_neg):
        c = rnd.choice(negs)
        rows.append({"hist": chat_hist(convs), "user": c["user"], "target": "CHAT", "type": "chat_ctx", "title": c["title"]})
    # card_split: 分けて書いた題名
    for t in titles:
        s = split_word(t, vocab)
        if not s: continue
        for f in rnd.sample(["{t}?", "What is {a} {t}?", "You know {t}?", "Tell me about {t}."], 2):
            rows.append({"hist": [] if rnd.random() < 0.6 else chat_hist(convs), "user": f.format(t=s, a=ART(s)), "target": f"CARD {t}", "type": "card_split", "title": t})
    return rows


def main():
    from gen_card_data import load_entries
    ents, _ = load_entries()
    vocab = {w for t in ents for w in t.lower().split()}
    for t in ents: vocab |= set(ents[t]["entry"].lower().replace(".", " ").replace(",", " ").split())
    vocab = {w for w in vocab if w.isalpha() and len(w) >= 3}
    for split, path in (("train", "data/dialogues_train_expanded.jsonl"), ("test", "data/dialogues_val.jsonl")):
        convs = [r["turns"] for r in load(path) if len(r["turns"]) >= 4]
        rows = build(split, convs, vocab)
        with open(f"data/router/card3_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows), dict(Counter(r["type"] for r in rows)))
    print([r for r in rows if r["type"] == "card_split"][:4])


if __name__ == "__main__":
    main()
