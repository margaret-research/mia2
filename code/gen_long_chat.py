# -*- coding: utf-8 -*-
"""MIA2 項目6の試作: 長い会話(10往復/30往復)を Qwen に1回で書かせ、どこまで話が続くかを見る

制約なし生成(「10歳の子が使う言葉で」と頼むだけ。gen_chat_qa_free.py と同じ方針)。1回の呼び出しで1会話。
出力: data/long_chat/raw_<tag>.jsonl  {"turns":[...], "exch": 目標の往復数, "scenario":..., "topic":...}
使い方: python gen_long_chat.py long10 10 100     (タグ, 往復数, 会話数) ※先にvLLMサーバーを起動
生成テキストは信頼できない外部由来として扱う(整形と保存だけ)。
"""
import json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa as base
import gen_chat_qa_gbnf as g
import gen_chat_qa_local as local

OUT = os.path.join(base.HERE, "data", "long_chat")
MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "})

SCENARIOS = [
    "The child tells Mia about their day, step by step. Mia reacts, asks small questions, and the child answers.",
    "The child asks Mia many questions about one topic, and each new question follows from Mia's last answer.",
    "The child has a small problem (a friend, homework, a lost toy) and Mia helps step by step. The child tries the ideas and tells how it went.",
    "A relaxed free chat. The talk moves from one topic to a related one, the way real chats wander. Mia also asks the child questions about themself.",
    "The child and Mia play a word or guessing game (riddles, 'I spy', twenty questions, naming animals) and keep chatting between turns.",
    "The child wants to make up a little story together with Mia. They take turns adding to it and talk about it.",
    "The child asks Mia about Mia ('How are you?', 'What do you like?', 'Do you have a friend?') and then the talk goes back to the child's own life.",
]

PROMPT = """You write one long English chat example for training a small chatbot that talks with children.
The chatbot is a friendly chat helper named Mia. It talks with a child about 10 years old. Write the way an ordinary 10-year-old uses English: everyday words, short clear sentences.
Topic: {topic}
What happens: {scenario}
Rules:
- Exactly {n} exchanges: {n} USER turns and {n} BOT turns, alternating, starting with USER and ending with BOT.
- BOTH the user and Mia use words a 10-year-old knows. If a hard idea comes up, say it with easy words.
- Each turn is 1-2 short sentences (at most 3). Plain text, no emoji, no lists, no markdown.
- The talk must keep going naturally: later turns refer back to things said earlier (names, things, feelings), and the same thing is never said twice. Do not end the chat early or say goodbye until the last exchange.
- Mia does not invent private facts about the child. When asked its name it says "My name is Mia."
Output only the turns, one per line, each line starting with "USER:" or "BOT:". Nothing else."""


def parse(txt):
    turns, want = [], "USER"
    for line in txt.splitlines():
        m = re.match(r"^\s*(USER|BOT)\s*:\s*(.+)$", line.strip())
        if not m:
            continue
        who, body = m.group(1), re.sub(r"\s+", " ", m.group(2).translate(MAP)).strip()
        if who != want or not body:
            return None
        turns.append(body)
        want = "BOT" if want == "USER" else "USER"
    return turns


def one(job):
    i, exch, seed = job
    rnd = random.Random(seed)
    topics = g.TOPICS_EXT.split("|") if isinstance(g.TOPICS_EXT, str) else list(g.TOPICS_EXT)
    topic = rnd.choice(topics + base.TOPICS)
    scenario = SCENARIOS[i % len(SCENARIOS)]
    txt = local.ask(PROMPT.format(n=exch, topic=topic, scenario=scenario), max_tokens=int(exch * 70 + 300), temperature=1.0)
    return {"exch": exch, "scenario": i % len(SCENARIOS), "topic": topic, "raw_ok": True, "turns": parse(txt), "text": txt}


def main(tag, exch, count):
    os.makedirs(OUT, exist_ok=True)
    rnd = random.Random(2026 + exch)
    jobs = [(i, exch, rnd.randrange(10**9)) for i in range(count)]
    t0, res = time.time(), []
    with ThreadPoolExecutor(local.WORKERS) as ex:
        for k, r in enumerate(ex.map(one, jobs)):
            res.append(r)
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{count} {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, f"raw_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in res if r["turns"]]
    full = [r for r in ok if len(r["turns"]) == 2 * exch]
    print(f"{tag}: 生成 {len(res)} / 読み取れた {len(ok)} / 目標の往復数ぴったり {len(full)}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
