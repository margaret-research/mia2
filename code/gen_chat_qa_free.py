# -*- coding: utf-8 -*-
"""T-026 会話データを、制約なしで Qwen3.8-27B (vLLM) に書かせる: 「10歳くらいの子どもが使う言葉で」と頼むだけ

語彙の制約(文法)も、語彙・長さの除外も、OK/BAD判定もしない。最小限の整形と、次の3つだけ行う:
  ・JSONとして読めて、ターン数が偶数(2〜8)で、空でない文字列であること
  ・評価セット(eval_chat.py の固定62問)と同じユーザー発言は捨てる (学習に混ぜない方針)
  ・最初の発言が同じものは捨てる (完全な重複)
曲がった引用符などは ASCII にそろえる。自作トークナイザーでの長さ(own_tokens)と、分割される単語の割合は統計として記録するだけ。
出力: data/chat_qa/raw_<tag>.jsonl (採用) / rej_<tag>.jsonl (除外と理由)    ※ kept_ は作らない
使い方: python gen_chat_qa_free.py free_pilot 300      (先に vLLM サーバーを起動しておく。300=会話の目標数)
生成テキストは信頼できない外部由来として扱う。
"""
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa as base
import gen_chat_qa_gbnf as g
import gen_chat_qa_local as local

# 重複対策(区切り8から): 言い方を増やし、自作語彙の単語を1会話ごとに1語ヒントとして渡す
STYLES2 = ["starts with a statement about something the child did today", "starts with 'I' and a feeling", "starts with a question word (what, where, who, how)",
           "starts with a name of an animal or thing", "starts with 'My' (my dog, my friend, my toy)", "starts with a number or a time",
           "talks about something the child sees right now", "asks the bot to pick one of two things", "starts with 'If'", "starts with 'Do you' or 'Are you'",
           "says something silly or funny first", "starts with 'There is' or 'Today'", "asks how to do a small thing", "starts with 'Mia,' (calls the bot by name)",
           "complains a little", "shares good news"]
CONTENT_WORDS = []

K = 8   # 1回の呼び出しで作る会話の数
MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "})

PROMPT = """You write short English chat examples for training a small chatbot that talks with children.
The chatbot talks with a child about 10 years old. Write the way an ordinary 10-year-old uses English: everyday words, short clear sentences.
Write {k} different conversations. Type of every conversation: {cat_desc}
Topic hints (one per conversation, in order): {topics}
How the user's first turn sounds (one per conversation, in order): {styles}
Rules:
- Each conversation has {turns_desc} turns, alternating USER, BOT, USER, BOT ... It starts with USER and ends with BOT.
- BOTH the user and the bot use words a 10-year-old child knows. The bot must not use hard words; if a hard idea comes up, say it with easy words.
- The user's turn is 1-2 short sentences. The bot's reply is 1-2 short sentences (at most 3). Plain text, no emoji, no lists, no markdown.
- The bot is a friendly chat helper named Mia. When asked its name it says "My name is Mia." It does not invent private facts about the user.
- The bot answers the question that was asked, directly and kindly.
{extra}
Output JSON Lines only (no code block, no numbering), one object per line: {{"turns": ["user text", "bot text", ...]}}"""


def one(job):
    jid, cat, seed = job
    rnd = random.Random(seed)
    turns_desc = {"cmd": "2 or 4", "self": "2 or 4", "multi": "exactly 4 or 6", "qa": "2 or 4"}[cat]
    extra = ("- IMPORTANT: the LAST user turn must only make sense with an EARLIER turn (e.g. 'What is my name?', 'Why?', 'What else?', "
             "'Which one is bigger?'), and the bot's last answer must use that earlier information.") if cat == "multi" else ""
    topics = rnd.sample(g.TOPICS_EXT + base.TOPICS, K)
    styles = [rnd.choice(g.STYLES + STYLES2 * 2) for _ in range(K)]
    hints = ""
    if CONTENT_WORDS:
        hints = ("\nWord hints (one per conversation, in order; use the word in the conversation if it fits naturally, otherwise ignore it): "
                 + ", ".join(rnd.choice(CONTENT_WORDS) for _ in range(K)))
        hints += "\nAvoid starting the user's first turn with 'Can you help' or 'Please tell me'."
    txt = local.ask(PROMPT.format(k=K, cat_desc=base.CATS[cat], topics=", ".join(topics), styles="; ".join(styles),
                                  turns_desc=turns_desc, extra=extra + hints), max_tokens=1400, temperature=1.0)
    return cat, [r.get("turns") for r in base.json_lines(txt) if isinstance(r, dict)]


def accept(turns, seen):
    if not isinstance(turns, list) or not all(isinstance(t, str) and t.strip() for t in turns):
        return "bad-format", None
    if len(turns) % 2 or not 2 <= len(turns) <= 8:
        return f"turn-count:{len(turns)}", None
    turns = [re.sub(r"\s+", " ", t.translate(MAP)).strip() for t in turns]
    if any(base.norm(t) in base.EVAL_USER for t in turns[0::2]):
        return "eval-overlap", None
    if base.norm(turns[0]) in seen:
        return "dup-first-turn", None
    return None, turns


def stats(turns):
    """自作トークナイザーでの長さ(+1/ターンは役割トークン)と、複数に分割された単語数/全単語数"""
    n, split, words = 0, 0, 0
    for t in turns:
        n += len(local.tok.encode(t).ids) + 1
        for w in t.split():
            w = w.strip(".,!?;:\"()")
            if w:
                words += 1
                split += len(local.tok.encode(w).ids) > 1
    return n, split, words


def main(tag, target):
    rnd = random.Random(int(os.environ.get("SEED", 2026)))
    n_calls = max(1, int(target / K * 1.15))
    cats = list(base.CATS)
    jobs = [(i, cats[i % len(cats)], rnd.randrange(10**9)) for i in range(n_calls)]
    CONTENT_WORDS.extend(w for w in g.allowed_words() if w.islower() and len(w) >= 4 and w.isalpha())
    print(f"ヒント用の単語 {len(CONTENT_WORDS)} 語", flush=True)
    ok, rej, seen, t0 = [], [], g.preload_seen(), time.time()
    tot_split = tot_words = 0
    with ThreadPoolExecutor(local.WORKERS) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(futs):
            try:
                cat, res = f.result()
            except Exception as e:
                print(f"[停止] {e}", file=sys.stderr)
                continue
            for turns in res:
                why, clean = accept(turns, seen)
                if why:
                    rej.append({"why": why, "cat": cat, "turns": turns})
                else:
                    n, sp, w = stats(clean)
                    tot_split, tot_words = tot_split + sp, tot_words + w
                    seen.add(base.norm(clean[0]))
                    ok.append({"cat": cat, "turns": clean, "own_tokens": n})
            if (i + 1) % 20 == 0 or i + 1 == len(futs):
                print(f"{i + 1}/{len(futs)} 採用{len(ok)} 除外{len(rej)} {time.time() - t0:.0f}s", flush=True)
    for name, data in ((f"raw_{tag}.jsonl", ok), (f"rej_{tag}.jsonl", rej)):
        with open(os.path.join(base.OUT, name), "w", encoding="utf-8") as fo:
            for r in data:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    tk = [r["own_tokens"] for r in ok]
    print(f"{tag}: 採用 {len(ok)} / 除外 {len(rej)}  1会話の自作トークン 平均{sum(tk) / max(1, len(tk)):.1f} 最大{max(tk, default=0)}  "
          f"複数に分割される単語 {tot_split}/{tot_words} = {tot_split / max(1, tot_words):.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "free_pilot", int(sys.argv[2]) if len(sys.argv) > 2 else 300)
