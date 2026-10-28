# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — 会話Botの振り分けを手書きの入力で確かめる (期待: chat / calc / know)"""
import sys, warnings
warnings.filterwarnings("ignore")
from bot import Bot, show
P = [("Hello! How are you today?", "chat"), ("What is your name?", "chat"), ("I like to play soccer after school.", "chat"),
     ("Do you have time for coffee this afternoon?", "chat"), ("My dog is sick and I am sad.", "chat"), ("Thank you very much.", "chat"),
     ("What did you do last weekend?", "chat"), ("Can you help me find the station?", "chat"), ("Where do you live?", "chat"),
     ("Do you like dogs?", "chat"), ("I went to the zoo yesterday.", "chat"), ("What time is it?", "chat"),
     ("What is twelve plus five?", "calc"), ("7 x 8", "calc"), ("What is 100 divided by 4?", "calc"), ("subtract 3 from 10", "calc"),
     ("I have 5 apples and get 7 more. How many apples do I have?", "calc"),
     ("What is a submarine?", "know"), ("Do alpacas have wool?", "know"), ("How many legs does a spider have?", "know"),
     ("Where do penguins live?", "know"), ("Tell me about the moon.", "know"), ("What do bees make?", "know"),
     ("Is a whale a fish?", "know"), ("what color is a ladybug", "know"), ("Why do leaves fall in autumn?", "know"),
     ("What does a firefighter do?", "know"), ("Which season is the warmest?", "know"), ("Can birds fly?", "know"),
     ("What is the capital of Mars?", "know")]
bot = Bot()
ok = 0
for u, exp in P:
    r = bot.say(u); bot.reset()
    ok += r["who"] == exp
    print(("OK " if r["who"] == exp else "NG ") + f"[{exp}] {u}\n     {show(r)[:150]}")
print(f"\n振り分け一致 {ok}/{len(P)}")
