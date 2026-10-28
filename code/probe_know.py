# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — 会話Botの知識の通し確認 (手書きの質問 + 評価題材のカード)"""
import warnings; warnings.filterwarnings("ignore")
from bot import Bot
import sys
bot = Bot(sys.argv[1]) if len(sys.argv) > 1 else Bot()
Q = ["What is a submarine?", "Do alpacas have wool?", "How many legs does a spider have?", "Where do penguins live?", "Tell me about the moon.",
     "What do bees make?", "Is a whale a fish?", "What color is a ladybug?", "What does a firefighter do?", "Which season is the warmest?",
     "Can birds fly?", "What is a deck?", "Do spiders have six legs?", "Does a cow eat meat?", "What is the capital of Mars?", "How old is the moon?",
     "What do rabbits eat?", "Where does a fish live?", "Is the sun cold?", "What is a carrot?"]
for q in Q:
    r = bot.say(q); bot.reset()
    print(f"[{r['who']}] {q}\n   -> {r['reply']}" + (f"\n   card: {r['card']}" if r.get('card') else ""))
