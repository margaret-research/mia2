# -*- coding: utf-8 -*-
"""T-026 会話の追加生成: 想起型(recall) と あいさつ型(greet) を Qwen3.8-27B (vLLM) に書かせる

recall: 1つ目の発言で事実(名前・好きなもの・行き先など)を言い、あとで「それは何?」と聞き返す。4または6ターン。
        一部(約15%)は「まだ聞いていないよ」と答える型(untold)。作り話を禁じ、自動チェックで答えが前の発言の言葉を使っているものだけ残す。
greet : あいさつ・お礼・お別れ・おわび。2ターン中心(一部4ターン)。
評価セット(eval_chat.py の固定62問)と同じユーザー発言は捨てる。
出力: data/chat_qa/raw_x_<cat>.jsonl (採用、追記) / rej_x_<cat>.jsonl (除外と理由)。再開すると既存分を読んで続きを作る。
使い方: python gen_chat_qa_extra.py recall 20000   (先に vLLM サーバーを起動しておく)
        TAG=pilot python gen_chat_qa_extra.py recall 200   (TAG を付けると raw_x_<cat>_<TAG>.jsonl に出す: 試作用)
生成テキストは信頼できない外部由来として扱う。
"""
import collections
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa as base
import gen_chat_qa_gbnf as g
import gen_chat_qa_local as local
from gen_chat_qa_free import MAP, stats

K = 8        # greet: 1回の呼び出しで作る会話の数
KR = 5       # recall: 計画文が長く出力が文脈長2048に収まらないので少なめ
NAMES = ("Mike Ella Tom Anna Ben Lily Sam Emma Jack Zoe Max Ruby Leo Nina Dan Kate Joe Amy Ray Eva Sue Tim Ivy Lucas Mark Rosa Ned Pia Hugo Cora "
         "Omar Tess Finn Maya Noah Lena Owen Jade Kai Rita Seth Dora Luke Gina Ali Beth Carl Fay Ian Joy").split()
# recall: (事実の種類, 1つ目の発言の例, 聞き返しの例)
FACTS = [
    ("the user's name", "My name is Ella.", "What is my name?"),
    ("the user's favorite food", "I like pizza.", "What food do I like?"),
    ("the user's favorite color", "My favorite color is green.", "What is my favorite color?"),
    ("where the user is going", "I am going to the park.", "Where am I going?"),
    ("the user's pet and its name", "I have a dog named Max.", "What is my dog's name?"),
    ("the user's age", "I am nine years old.", "How old am I?"),
    ("something the user has", "I have a red kite.", "What do I have?"),
    ("what the user did today", "I played soccer today.", "What did I do today?"),
    ("the user's favorite animal", "I love horses.", "What animal do I love?"),
    ("the user's sibling", "My sister is called Amy.", "What is my sister's name?"),
    ("what the user wants", "I want a new bike.", "What do I want?"),
    ("where the user lives", "I live near the sea.", "Where do I live?"),
    ("the user's favorite game or sport", "I like to play chess.", "What do I like to play?"),
    ("the user's favorite school subject", "I like art class.", "What class do I like?"),
    ("what the user is eating or drinking", "I am drinking milk.", "What am I drinking?"),
    ("a friend of the user", "My friend Ben has a big dog.", "Who has a big dog?"),
    ("what the user will do tomorrow", "Tomorrow I will visit my aunt.", "Who will I visit tomorrow?"),
    ("what the user lost or found", "I lost my blue hat.", "What did I lose?"),
]
ASKS = ["asks it as a short question (like the example)", "asks it with 'Do you remember ...?'", "asks it with 'Can you tell me ...?'",
        "asks it all in lower case with no final mark", "asks it after one more small chat turn about something else"]
GREET = [
    "says hello or hi (many different ways: 'Hi there!', 'Hey Mia!', 'Hello, Mia.', 'hi', 'Hi, how is it going?')",
    "says good morning, good afternoon or good evening to the bot",
    "asks how the bot is ('How are you?', 'How is your day?', 'What is up?'); the bot answers simply and may ask back",
    "meets the bot for the first time ('Nice to meet you', 'I am new here', 'Are you Mia?')",
    "says thank you for help or for something kind (many different ways); the bot replies 'You are welcome' in its own words",
    "says goodbye or see you later or good night; the bot says goodbye kindly",
    "says sorry (late, forgot, broke something, bumped into someone); the bot answers kindly and says it is okay",
    "says hello and then, after the bot answers, says how the user feels; the bot responds kindly (4 turns)",
    "says thanks and then goodbye in one turn; the bot says you are welcome and goodbye",
    "comes back to the bot after a while ('I am back', 'Hi again'); the bot welcomes the user",
    "wishes the bot or someone a happy birthday, a nice day or good luck; the bot says thanks kindly",
    "says hello and tells its name; the bot greets the user by that name",
]
GSTYLES = ["all lowercase with no final mark", "with an exclamation mark", "very short (one or two words plus the bot's name)", "polite and complete",
           "friendly and a little long (one or two short sentences)", "calls the bot 'Mia'", "casual", "excited"]

PROMPT_R = """You write short English chat examples for training a small chatbot that talks with children.
The chatbot is a friendly chat helper named Mia. The child is about 10 years old. Write the way an ordinary 10-year-old uses English: everyday words, short clear sentences.
Write {k} different conversations about REMEMBERING what the user said earlier in the same conversation.
Plan for each conversation (one per line, in order; follow it):
{plans}
Rules:
- Each conversation has exactly {turns} turns, alternating USER, BOT, USER, BOT ... It starts with USER and ends with BOT.
- Turn 1 (USER) tells the fact (use the example only as a style guide; change the name, thing and words). Turn 2 (BOT) reacts kindly in one short sentence and does NOT repeat the fact word for word.
- The LAST user turn asks the bot to give back the fact. For kind=answer, the last BOT turn gives the correct answer from the earlier turn, in a short full sentence that uses the fact word (for example "Your name is Ella." or "You like pizza.").
- For kind=untold, the user asks about something it NEVER said (for example its pet's name when no pet was mentioned). The last BOT turn says honestly that it has not been told yet (for example "You have not told me yet. What is it?"). It must not guess.
- Never invent facts. The bot only uses what the user said: it must not use any name that the user did not say. The bot is Mia and does not claim human things.
- The user's turn is 1-2 short sentences. The bot's reply is 1-2 short sentences. Plain text, no emoji, no lists, no markdown.
Output JSON Lines only (no code block, no numbering), one object per line, in the same order: {{"kind": "answer or untold", "turns": ["user text", "bot text", ...]}}"""

PROMPT_G = """You write short English chat examples for training a small chatbot that talks with children.
The chatbot is a friendly chat helper named Mia. The child is about 10 years old. Write the way an ordinary 10-year-old uses English: everyday words, short clear sentences.
Write {k} different conversations of greetings and polite small talk.
Plan for each conversation (one per line, in order; follow it):
{plans}
Rules:
- Each conversation has the number of turns in its plan, alternating USER, BOT, USER, BOT ... It starts with USER and ends with BOT.
- The user's turn is short (1-2 short sentences). The bot's reply is short and warm (1-2 short sentences). Plain text, no emoji, no lists, no markdown.
- Do not write the user turn as only the single word "Hello" or only "Thank you" or only "Goodbye": add a name, a few words or a small detail, and vary the wording from conversation to conversation.
- The bot is Mia. It does not claim human things (no sleeping, eating, family, school). It may say it is doing well, it is happy to chat, and may ask a friendly question back.
Output JSON Lines only (no code block, no numbering), one object per line, in the same order: {{"turns": ["user text", "bot text", ...]}}"""

STOP = set("the a an is are was were i you my me it to of and in on at for with that this do does did what where who which have has your can will not "
           "no yes so but be as or if its am our us we he she they them his her there here how old much many some any very too just".split())
UNTOLD = re.compile(r"\b(not told|have not|has not|did not|do not know|don't know|haven't|didn't|not said|not say|not sure|have no|no idea|never)\b|\bnot yet\b", re.I)
UNTOLD_P = float(os.environ.get("UNTOLD_P", 0))   # 「まだ聞いていない」型の割合。試作では Qwen が実際は言われたことを『聞いていない』と答える誤りが多かったので0
NAMESET = set(NAMES)
IGN = {"like", "love", "favorite", "have", "name", "tell", "remember", "know", "today", "again", "doing", "going", "want"}
CAP = {"recall": 3, "greet": 8}   # 同じ最初の発言を許す回数


def cw(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2}


def job_plans(cat, rnd, words):
    k = KR if cat == "recall" else K
    """K個の会話ぶんの計画文と、期待するターン数のリストを返す"""
    plans, turns = [], []
    for i in range(k):
        if cat == "recall":
            f = rnd.choice(FACTS)
            kind = "untold" if rnd.random() < UNTOLD_P else "answer"
            t = 4 if rnd.random() < 0.7 else 6
            name = rnd.choice(NAMES)
            hint = rnd.choice(words)
            extra = " Add one more small chat exchange in the middle." if t == 6 else ""
            plans.append(f"{i + 1}. kind={kind}; fact: {f[0]} (example: USER \"{f[1]}\" ... \"{f[2]}\"); "
                         f"if the child says a name, use '{name}'; if it fits, use the word '{hint}'; the question {rnd.choice(ASKS)}.{extra} Exactly {t} turns.")
            turns.append(t)
        else:
            t = 4 if rnd.random() < 0.2 else 2
            style = rnd.choice(GSTYLES)
            plans.append(f"{i + 1}. the user {rnd.choice(GREET)}; the user's first turn is {style}. Exactly {t} turns.")
            turns.append(t)
    return plans, turns


def one(job):
    cat, seed, words = job
    rnd = random.Random(seed)
    plans, turns = job_plans(cat, rnd, words)
    if cat == "recall":
        prompt = PROMPT_R.format(k=KR, plans="\n".join(plans), turns="4 or 6 (as in each plan)")
    else:
        prompt = PROMPT_G.format(k=K, plans="\n".join(plans))
    room = 2040 - int(len(local.tok.encode(prompt).ids) * 1.25) - 20   # 文脈長2048に収まる出力上限 (トークナイザーが違うので余裕を見る)
    txt = local.ask(prompt, max_tokens=min(1400, room), temperature=1.0)
    return [(r.get("turns"), r.get("kind")) for r in base.json_lines(txt) if isinstance(r, dict)]


def accept(cat, turns, kind, seen, firsts):
    if not isinstance(turns, list) or not all(isinstance(t, str) and t.strip() for t in turns):
        return "bad-format", None
    ok_n = (4, 6) if cat == "recall" else (2, 4)
    if len(turns) not in ok_n:
        return f"turn-count:{len(turns)}", None
    turns = [re.sub(r"\s+", " ", t.translate(MAP)).strip() for t in turns]
    if any(base.norm(t) in base.EVAL_USER for t in turns[0::2]):
        return "eval-overlap", None
    if re.search(r"[^\x00-\x7f]|[*#_`]|^\s*[-]\s", " ".join(turns)):
        return "symbol", None
    if any(re.match(r"^\s*(user|bot)\s*:", t, re.I) for t in turns):
        return "label", None
    key = base.norm(" ".join(turns))
    if key in seen:
        return "dup-conversation", None
    f = base.norm(turns[0])
    if firsts[f] >= CAP[cat]:
        return "dup-first-turn", None
    if cat == "recall":
        if kind not in ("answer", "untold"):
            return "bad-kind", None
        last_bot = turns[-1]
        if not re.search(r"(?i)\b(what|where|who|whom|which|when|how|why|do|does|did|can|could|is|are|am|was|were|will)\b|\?\s*$", turns[-2].strip()):
            return "last-user-not-question", None
        users = " ".join(turns[0::2])
        if any(w in NAMESET and w not in users for w in re.findall(r"[A-Z][a-z]+", " ".join(turns[1::2]))):
            return "bot-invented-name", None
        if kind == "untold":
            if cw(turns[-2]) & set().union(*[cw(t) for t in turns[:-2:2]]) - IGN:
                return "untold-but-told", None
        if kind == "untold":
            if not UNTOLD.search(last_bot):
                return "untold-no-admit", None
        else:
            if UNTOLD.search(last_bot):
                return "answer-says-untold", None
            earlier = set().union(*[cw(t) for t in turns[:-2:2]]) - cw(turns[-2])
            if not earlier & cw(last_bot):
                return "answer-not-from-earlier", None
        if len(turns[-2].split()) < 2:
            return "one-word", None
    return None, turns


def main(cat, target):
    tag = os.environ.get("TAG", "")
    suffix = f"{cat}_{tag}" if tag else cat
    out_ok, out_rej = (os.path.join(base.OUT, f"{p}_x_{suffix}.jsonl") for p in ("raw", "rej"))
    seen, firsts, have = set(), collections.Counter(), 0
    if os.path.exists(out_ok):
        for line in open(out_ok, encoding="utf-8"):
            r = json.loads(line)
            seen.add(base.norm(" ".join(r["turns"])))
            firsts[base.norm(r["turns"][0])] += 1
            have += 1
    if have >= target:
        print(f"既に {have} 件あります")
        return
    words = [w for w in g.allowed_words() if w.islower() and len(w) >= 4 and w.isalpha()]
    rnd = random.Random(int(os.environ.get("SEED", 7000)) + have)
    yield_rate = float(os.environ.get("YIELD", 0.75))
    n_calls = max(1, int((target - have) / ((KR if cat == "recall" else K) * yield_rate)))
    jobs = [(cat, rnd.randrange(10**9), words) for _ in range(n_calls)]
    why, t0, lock, done = collections.Counter(), time.time(), threading.Lock(), 0
    tot_split = tot_words = 0
    with open(out_ok, "a", encoding="utf-8") as fo, open(out_rej, "a", encoding="utf-8") as fr, ThreadPoolExecutor(local.WORKERS) as ex:
        for i, f in enumerate(ex.map(lambda j: _safe(j), jobs)):
            for turns, kind in f:
                w, clean = accept(cat, turns, kind, seen, firsts)
                if w:
                    why[w] += 1
                    fr.write(json.dumps({"why": w, "cat": cat, "kind": kind, "turns": turns}, ensure_ascii=False) + "\n")
                    continue
                n, sp, wd = stats(clean)
                tot_split, tot_words = tot_split + sp, tot_words + wd
                seen.add(base.norm(" ".join(clean)))
                firsts[base.norm(clean[0])] += 1
                fo.write(json.dumps({"cat": cat, "kind": kind, "turns": clean, "own_tokens": n}, ensure_ascii=False) + "\n")
                have += 1
            fo.flush(); fr.flush()
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} 採用累計{have} 除外{sum(why.values())} {dict(why.most_common(5))} {time.time() - t0:.0f}s", flush=True)
            if have >= target:
                break
    print(f"{suffix}: 採用 {have} 分割される単語 {tot_split}/{tot_words} = {tot_split / max(1, tot_words):.1%}  除外内訳 {dict(why)}")


def _safe(j):
    try:
        return one(j)
    except Exception as e:
        print(f"[呼び出し失敗] {e}", file=sys.stderr)
        return []


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
