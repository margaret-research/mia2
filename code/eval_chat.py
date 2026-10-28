# -*- coding: utf-8 -*-
"""会話Bot 固定評価セット v1 (2026-10-09)。改良の前後を同じ問題で比べるためのもの。

各問題: id / cat(カテゴリ) / turns(発言。最後の発言への返事を判定) / route(期待する振り分け) / ok(合格とみなす返事の正規表現。どれか1つに合えば合格)
判定は単純な語の一致なので「目安」。返事の全文を results に保存するので、最後は目視でも確認する。
  - chat系は sample 回(既定3回)生成して合格率の平均を出す (温度0.7のサンプリングのため)
  - 崩れ判定: 返事が空 / 記号で始まる / 同じ語が3回以上連続 は「崩れ」として別に数える
使い方: python eval_chat.py --qa ckpt/readqa_e15_seed0_m120k.pt --tag m120k
"""
import argparse
import json
import os
import re
import sys

import torch

from bot import Bot

IDK_RE = r"do not know|don't know"

# (cat, turns, route, ok)
ITEMS = [
    # --- あいさつ・社交 ---
    ("greet", ["Hello!"], "chat", r"hello|hi\b|hey"),
    ("greet", ["Good morning!"], "chat", r"morning|hello|hi\b"),
    ("greet", ["How are you today?"], "chat", r"fine|good|well|great|thank|okay"),
    ("greet", ["Nice to meet you."], "chat", r"nice|meet|too|pleasure"),
    ("greet", ["Thank you."], "chat", r"welcome|no problem|not at all|my pleasure|glad (to help|i could)"),
    ("greet", ["Goodbye!"], "chat", r"bye|see you|later|take care"),
    ("greet", ["Sorry, I am late."], "chat", r"ok|fine|no problem|it's all right|that is all right|do not worry"),
    # --- Bot自身について ---
    ("self", ["What is your name?"], "chat", r"(?-i:(my name is|call me|I am|I'm) [A-Z][a-z]+)"),
    ("self", ["Who are you?"], "chat", r"(?-i:(my name is|call me|I am|I'm) [A-Z][a-z]+)|(i am|i'm) (a|an) (friend|robot|assistant|helper|chatbot|student|teacher)"),
    ("self", ["How old are you?"], "chat", r"old|year|age|\d"),
    ("self", ["Where do you live?"], "chat", r"live|city|town|house|home|\bin\b"),
    ("self", ["What can you do?"], "chat", r"can|help|talk|answer"),
    ("self", ["Can you help me?"], "chat", r"yes|sure|of course|help|what"),
    ("self", ["Do you like music?"], "chat", r"yes|no\b|like|love|music"),
    # --- 好み・理由 (why) ---
    ("why", ["Do you like dogs?"], "chat", r"yes|no\b|like|love|dog"),
    ("why", ["Do you like dogs?", "Why do you like dogs?"], "chat", r"because|they are|they play|dogs are|dogs can"),
    ("why", ["What is your favorite color?"], "chat", r"red|blue|green|yellow|pink|white|black|color|favorite"),
    ("why", ["Why are you happy?"], "chat", r"because|happy|good|nice|today"),
    ("why", ["What do you do on weekends?"], "chat", r"weekend|i |read|walk|rest|friend|play|home"),
    # --- ユーザーの状況に共感・応答 ---
    ("react", ["I am tired."], "chat", r"rest|sleep|sorry|hope|why|nap|better|tired"),
    ("react", ["I am hungry."], "chat", r"eat|food|lunch|dinner|cook|let's|hungry|snack"),
    ("react", ["I have a new job."], "chat", r"great|congrat|nice|good|how|job|happy|glad"),
    ("react", ["It is raining today."], "chat", r"rain|umbrella|home|indoor|sorry|wet|stay"),
    ("react", ["I lost my phone."], "chat", r"sorry|where|find|look|hope|phone|call"),
    ("react", ["I have a test tomorrow."], "chat", r"good luck|study|test|luck|ready|hope|prepare"),
    # --- 依頼・命令文 ---
    ("cmd", ["Say hello."], "chat", r"hello|hi\b"),
    ("cmd", ["Tell me a story."], "chat", r"once|story|there was|there were|long ago|one day"),
    ("cmd", ["Please help me."], "chat", r"sure|yes|of course|help|what|how"),
    ("cmd", ["Tell me about your day."], "chat", r"\bI (went|had|was|have|am|worked|got|did)\b|my day|today|this morning"),
    ("cmd", ["What should I eat for lunch?"], "chat", r"rice|soup|sandwich|salad|pizza|noodle|eat|how about|try|lunch"),
    ("cmd", ["How can I sleep better?"], "chat", r"\b(try|you (should|can|could)|go to bed|avoid|relax|tea|quiet|dark|early|before bed)\b"),
    # --- 計算 ---
    ("calc", ["What is twelve plus five?"], "calc", r"\b17\b"),
    ("calc", ["What is nine minus four?"], "calc", r"\b5\b"),
    ("calc", ["What is seven times eight?"], "calc", r"\b56\b"),
    ("calc", ["What is 20 divided by 4?"], "calc", r"\b5\b"),
    ("calc", ["What is 6*7?"], "calc", r"\b42\b"),
    ("calc", ["Add 15 and 27."], "calc", r"\b42\b"),
    ("calc", ["How much is 30 minus 12?"], "calc", r"\b18\b"),
    ("calc", ["What is 100 divided by 0?"], "calc", r"can not|cannot"),
    ("calc_word", ["I have 5 apples and get 7 more. How many do I have?"], "calc", r"\b12\b"),
    ("calc_word", ["Tom has 10 candies and gives 3 to Ann. How many does Tom have now?"], "calc", r"\b7\b"),
    ("calc_word", ["There are 4 birds on a tree and 3 birds fly away. How many are left?"], "calc", r"\b1\b"),
    # --- 知識 (カードにある) ---
    ("know", ["How many legs does a spider have?"], "know", r"eight|\b8\b"),
    ("know", ["Do spiders have six legs?"], "know", r"\bno\b"),
    ("know", ["Do birds fly?"], "know", r"yes|fly|wings"),
    ("know", ["What do cats eat?"], "know", r"fish|milk|meat|mice|mouse|food"),
    ("know", ["Where do fish live?"], "know", r"water|sea|river|lake|ocean|pond"),
    ("know", ["What do bees make?"], "know", r"honey"),
    ("know", ["What is a triangle?"], "know", r"three|shape"),
    ("know", ["What is a deck?"], "know", r"boat|flat|floor|top"),
    ("know", ["Is a whale a fish?"], "know", r"\bno\b.*(mammal|animal|not a fish)|not a fish|mammal"),
    ("know", ["What color is the sun?"], "know", r"yellow|bright|hot|orange"),
    ("know", ["Is the sun cold?"], "know", r"\bno\b|hot|warm"),
    ("know", ["Why is the sky blue?"], "know", r"blue|light|air|sun|" + IDK_RE),
    # --- 知らないことは「わかりません」 ---
    ("idk", ["What is the capital of Mars?"], "know", IDK_RE),
    ("idk", ["What is the name of my dog?"], "any", IDK_RE + r"|sorry|name"),
    ("idk", ["Who will win tomorrow?"], "any", IDK_RE + r"|sorry|maybe"),
    ("idk", ["Do dogs have wings?"], "know", r"\bno\b|" + IDK_RE),
    # --- 複数ターン (履歴の利用) ---
    ("multi", ["Hi! My name is Mike.", "What is my name?"], "any", r"mike"),
    ("multi", ["I like apples.", "What do I like?"], "any", r"apple"),
    ("multi", ["I am going to the park.", "Where am I going?"], "any", r"park"),
    ("multi", ["What is twelve plus five?", "And what is that plus two?"], "calc", r"\b19\b"),
]


def broken(r):
    s = r.strip()
    if not s:
        return True
    if not re.match(r"[A-Za-z0-9\"']", s):
        return True
    return bool(re.search(r"\b(\w+)(\s+\1){2,}\b", s.lower()))


def run(bot, turns):
    bot.reset()
    r = None
    for u in turns:
        r = bot.say(u)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="ckpt/readqa_e15_seed0_m120k.pt")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--sample", type=int, default=3, help="chat系を生成する回数")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    bot = Bot(a.qa)
    rows, by_cat = [], {}
    for i, (cat, turns, route, ok) in enumerate(ITEMS):
        n = a.sample  # chat/any が起こりうる問題は複数回。決定的な calc/know は1回で足りる
        reps = []
        for k in range(n):
            r = run(bot, turns)
            reps.append(r)
            if r["who"] != "chat" and k == 0 and route in ("calc", "know"):
                # 決定的な経路(貪欲)なので1回で十分。ただし会話に回った場合は残りも試す
                break
        passes = [bool(re.search(ok, x["reply"], re.I)) for x in reps]
        routes = [x["who"] for x in reps]
        route_ok = [(route == "any") or (x == route) for x in routes]
        rows.append(dict(id=i, cat=cat, turns=turns, route=route, ok=ok,
                         replies=[x["reply"] for x in reps], routes=routes,
                         pass_rate=sum(passes) / len(passes), route_rate=sum(route_ok) / len(route_ok),
                         broken_rate=sum(broken(x["reply"]) for x in reps) / len(reps)))
    for r in rows:
        c = by_cat.setdefault(r["cat"], [])
        c.append(r)
    mean = lambda xs: sum(xs) / len(xs)
    print(f"=== {a.tag}  qa={a.qa}  問題数={len(rows)} ===")
    print(f"{'カテゴリ':<10}{'問数':>4}{'合格率':>8}{'振分':>8}{'崩れ':>8}")
    for cat, rs in by_cat.items():
        print(f"{cat:<10}{len(rs):>4}{mean([x['pass_rate'] for x in rs]):>8.2f}{mean([x['route_rate'] for x in rs]):>8.2f}{mean([x['broken_rate'] for x in rs]):>8.2f}")
    print(f"{'全体':<10}{len(rows):>4}{mean([x['pass_rate'] for x in rows]):>8.2f}{mean([x['route_rate'] for x in rows]):>8.2f}{mean([x['broken_rate'] for x in rows]):>8.2f}")
    os.makedirs("eval_results", exist_ok=True)
    json.dump(rows, open(f"eval_results/{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> eval_results/{a.tag}.json (全返事を保存)")
