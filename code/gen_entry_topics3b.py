# -*- coding: utf-8 -*-
"""追加題材(topics_add.json)の掃除: 複数形を除く / 低学年向けでない語・商標を手で除く / Qwen が改めて最適なテーマへ振り分け(合わないものは none で除く)
出力: data/reading_entry/topics_add2.json / .txt (除いたものも末尾に載せる)"""
import json, os
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local
import gen_entry_topics as T
import gen_entry_topics2 as T2
from nltk.corpus import wordnet as wn

BLOCK = set("gun handcuff ashtray urine nipple fart booger snot drool buttocks mucus pepsi twinkies hearse missile ufo alien hunting shooting poker blackjack lotto bingo pachinko tapeworm sunburn frostbite heatstroke nosebleed nightmare scar wart callus scab bruise sore sprain cramp blister syringe crutch ointment pain itch scrape cut sneezes hiccups hiccup burp snore cast brace head louse mosquito bite sheep tick maggot".split())
add = json.load(open(os.path.join(T.OUT, "topics_add.json")))
cand, plural = {}, []
for th, v in add.items():
    for t in v:
        if t in BLOCK or any(w in BLOCK for w in t.split()):
            continue
        if " " not in t and wn.morphy(t, "n") not in (None, t):
            plural.append(t); continue
        cand.setdefault(t, th)
items = sorted(cand)
with ThreadPoolExecutor(local.WORKERS) as ex:
    res = [p for part in ex.map(T2.assign, [items[i:i + 80] for i in range(0, len(items), 80)]) for p in part]
new, removed = {t: [] for t in T.THEMES}, []
for t, th in res:
    (removed.append(t) if th == "none" else new[th].append(t))
json.dump(new, open(os.path.join(T.OUT, "topics_add2.json"), "w"), ensure_ascii=False, indent=1)
with open(os.path.join(T.OUT, "topics_add2.txt"), "w", encoding="utf-8") as f:
    for t, v in new.items():
        f.write(f"## {t} ({len(v)})\n" + ", ".join(v) + "\n\n")
    f.write(f"## 消した題材 ({len(removed)})\n" + ", ".join(removed) + f"\n\n## 複数形で消した ({len(plural)})\n" + ", ".join(plural) + "\n")
print("元", sum(map(len, add.values())), "→ 採用", sum(map(len, new.values())), "/ 消した", len(removed), "/ 複数形", len(plural))
print({t: len(v) for t, v in new.items()})
