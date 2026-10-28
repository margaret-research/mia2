# -*- coding: utf-8 -*-
"""T-026 会話の「型」を増やす: 質問応答型の会話データをDeepSeekで生成する (試作用)

弱点カテゴリ(依頼・命令 / Bot自身への質問 / 複数ターン / 質問に会話で直接答える)ごとに、
2〜6ターンの短い対話を作る。出力は dialogues.jsonl と同じ {"turns": [...]} (ユーザー発言から始まり、交互)。
評価セット(eval_chat.py の固定62問)と同じ発言を含む対話は除外する (学習に混ぜない方針)。
出力: data/chat_qa/raw_<tag>.jsonl (採用) / rej_<tag>.jsonl (除外と理由)
使い方: COST_CAP=0.5 python gen_chat_qa.py pilot 200
APIキーは環境変数 DEEPSEEK_API_KEY から読む。費用が COST_CAP(USD) を超えたら以降の呼び出しを止める。
生成テキストは信頼できない外部由来として扱う。
"""
import json
import os
import random
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from eval_chat import ITEMS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "chat_qa")
os.makedirs(OUT, exist_ok=True)

COST_CAP = float(os.environ.get("COST_CAP", 0.5))
# 単価は安全側の見積もり (USD / 100万トークン)。実際の請求は DeepSeek の画面で確認する
PRICE_IN, PRICE_OUT = float(os.environ.get("PRICE_IN", 0.6)), float(os.environ.get("PRICE_OUT", 1.7))
MODEL = os.environ.get("DS_MODEL", "deepseek-chat")
cost, lock = 0.0, threading.Lock()

CATS = {
    "cmd": ("the user asks the bot to DO something with a short request or command "
            "(tell a story, give a tip, name three things, say something, explain simply, make a list, suggest an idea). "
            "The bot does it right away in a few sentences."),
    "self": ("the user asks the bot about itself (name, age, home, what it can do, likes, feelings, a typical day). "
             "The bot is Mia, a friendly chat helper; it answers simply and honestly and never claims things a bot cannot do."),
    "multi": ("a conversation where a later user turn depends on an earlier turn "
              "(the user says a fact about themselves or a thing, then asks about it, or asks 'why?', 'and that?', 'what else?'). "
              "The bot uses what was said before."),
    "qa": ("the user asks a simple everyday question (how, why, what, which, can I, should I) and the bot answers it "
           "directly in 1-3 simple sentences, then maybe asks a friendly question back."),
}
TOPICS = ["food", "weather", "school", "family", "pets", "sports", "music", "travel", "health", "hobbies", "shopping",
          "friends", "home", "work", "books", "games", "holidays", "nature", "time", "feelings", "city", "clothes"]

PROMPT = """You write short English chat examples for training a very small chatbot (vocabulary: 5,000 common words).
Write {k} different conversations. Type of every conversation: {cat_desc}
Topic hints (one per conversation, in order): {topics}
Rules:
- Each conversation has {turns_desc} turns, alternating USER, BOT, USER, BOT ... It starts with USER and ends with BOT.
- Use only easy, common words (grade 3-5). Each turn is 1-3 short sentences, under 30 words. Plain ASCII only, no emoji, no lists, no markdown.
- Vary how the user speaks: questions, requests with "Please", short commands, lower-case without a final mark (sometimes).
- The bot is a friendly chat helper named Mia. When asked its name it says "My name is Mia." It does not invent private facts about the user.
- The bot answers the question that was asked, directly and kindly.
{extra}
Output JSON Lines only (no code block, no numbering), one object per line: {{"turns": ["user text", "bot text", ...]}}"""


def ask(prompt, max_tokens=3000, temperature=1.0):
    global cost
    with lock:
        if cost >= COST_CAP:
            raise RuntimeError(f"COST_CAP ${COST_CAP} に到達 (累計 ${cost:.4f})")
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    u = d.get("usage", {})
    with lock:
        cost += u.get("prompt_tokens", 0) * PRICE_IN / 1e6 + u.get("completion_tokens", 0) * PRICE_OUT / 1e6
    return d["choices"][0]["message"]["content"]


def json_lines(txt):
    for line in txt.splitlines():
        line = line.strip().strip("`")
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except ValueError:
                pass


norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
EVAL_USER = {norm(t) for _, turns, _, _ in ITEMS for t in turns}


def check(turns, seen, cat=None):
    if not isinstance(turns, list) or not all(isinstance(t, str) for t in turns):
        return "not-list"
    if not 2 <= len(turns) <= 8 or len(turns) % 2:
        return f"turn-count:{len(turns)}"
    for i, t in enumerate(turns):
        t2 = t.strip()
        if not t2 or len(t2.split()) > 40:
            return "empty-or-long"
        if not t2.isascii() or re.search(r"[*#`\[\]{}<>|_]", t2):
            return "bad-chars"
        if re.match(r"(user|bot)\s*:", t2, re.I):
            return "role-prefix"
    if cat == "multi" and len(turns) < 4:
        return "multi-too-short"
    if any(norm(t) in EVAL_USER for t in turns[0::2]):
        return "eval-overlap"
    if re.search(r"\b(\w+)(\s+\1){2,}\b", " ".join(turns).lower()):
        return "repeat"
    if norm(turns[0]) in seen:
        return "dup-first-turn"
    return None


def one(job):
    jid, cat, seed = job
    rnd = random.Random(seed)
    k = 8
    turns_desc = {"cmd": "2 or 4", "self": "2 or 4", "multi": "exactly 4 or 6", "qa": "2 or 4"}[cat]
    extra = ("- IMPORTANT: the LAST user turn must only make sense with an EARLIER turn (e.g. 'What is my name?', 'Why?', 'What else?', "
             "'Which one is bigger?'), and the bot's last answer must use that earlier information.") if cat == "multi" else ""
    txt = ask(PROMPT.format(k=k, cat_desc=CATS[cat], turns_desc=turns_desc, topics=", ".join(rnd.sample(TOPICS, k)), extra=extra))
    return cat, [r.get("turns") for r in json_lines(txt) if isinstance(r, dict)]


def main(tag, target):
    rnd = random.Random(int(os.environ.get("SEED", 2026)))
    per_call = 8
    n_calls = max(1, round(target / (per_call * 0.9)))
    cats = list(CATS)
    jobs = [(i, cats[i % len(cats)], rnd.randrange(10**9)) for i in range(n_calls)]
    ok, rej, seen = [], [], set()
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(futs):
            try:
                cat, res = f.result()
            except Exception as e:      # 費用上限・通信エラー: そこまでの分は保存する
                print(f"[停止] {e}", file=sys.stderr)
                continue
            for turns in res:
                why = check(turns, seen, cat)
                if why:
                    rej.append({"why": why, "cat": cat, "turns": turns})
                else:
                    seen.add(norm(turns[0]))
                    ok.append({"cat": cat, "turns": [t.strip() for t in turns]})
            print(f"{i + 1}/{len(futs)} 採用{len(ok)} 除外{len(rej)} 累計 ${cost:.4f}", flush=True)
    for name, data in ((f"raw_{tag}.jsonl", ok), (f"rej_{tag}.jsonl", rej)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{tag}: 採用 {len(ok)} / 除外 {len(rej)}  費用(見積もり) ${cost:.4f} / 上限 ${COST_CAP}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot", int(sys.argv[2]) if len(sys.argv) > 2 else 200)
