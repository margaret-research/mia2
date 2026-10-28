# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-28) — 会話役を差し替えて試す。使い方: CHAT=ckpt/mia_rl2_s100.pt python bot_try.py  (省略すると現行Mia)"""
import os
import bot_mia2 as B
b = B.Bot(chat=os.environ.get("CHAT", "ckpt/mia_current_read.pt"))
while True:
    try:
        u = input("You: ")
    except EOFError:
        break
    if u.strip() == "exit":
        break
    w, r = b.say(u)
    print(f"Mia[{w}]: {r}")
