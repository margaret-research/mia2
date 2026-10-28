# -*- coding: utf-8 -*-
"""T-026 会話の「型」を増やす: 質問応答型の会話データを、ローカルの Qwen3.8-27B (vLLM) で生成する

gen_chat_qa.py (DeepSeek版) と同じ4種類の型・同じ出力形式 {"cat", "turns"} だが、
  ・小学校低学年(grade 1-2)向けのやさしい英語
  ・拡張トークナイザー(tokenizer_ext.json, 語彙9488)で数えて、語彙になじまない単語・長すぎる会話は除外
  ・API費用なし (サーバーは 127.0.0.1:8000 の vLLM)
評価セット(eval_chat.py の固定62問)と同じ発言を含む対話は除外する (学習に混ぜない方針)。
出力: data/chat_qa/raw_<tag>.jsonl (採用) / rej_<tag>.jsonl (除外と理由)
使い方: python gen_chat_qa_local.py qwen_pilot 200        (先に vLLM サーバーを起動しておく)
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

from tokenizers import Tokenizer

import gen_chat_qa as base   # CATS / TOPICS / json_lines / norm / check を再利用 (中身は DeepSeek 版のまま)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = base.OUT
URL = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen")
WORKERS = int(os.environ.get("WORKERS", 12))
MAX_OWN_TOKENS = 250            # llm5k の窓 256 から、<bos>/<eos>/役割トークン分を引いた本文の上限の目安
MAX_PIECES = 2                  # 1語がこれより多く分割される語は「語彙になじまない」とみなす
tok = Tokenizer.from_file(os.path.join(HERE, "data", "ts", "tokenizer_ext.json"))
usage, lock = {"prompt": 0, "completion": 0}, threading.Lock()

PROMPT = """You write short English chat examples for training a very small chatbot.
The chatbot speaks like a kind helper for children in grade 1-2 (age 6-8).
Write {k} different conversations. Type of every conversation: {cat_desc}
Topic hints (one per conversation, in order): {topics}
Rules:
- Each conversation has {turns_desc} turns, alternating USER, BOT, USER, BOT ... It starts with USER and ends with BOT.
- Use only very easy words that a 7-year-old knows. Short sentences. Each turn is 1-2 sentences, under 20 words. Plain ASCII only, no emoji, no lists, no markdown.
- Do not use rare words, names of places or brands, or made-up words and sounds.
- Vary how the user speaks: questions, requests with "Please", short commands, lower-case without a final mark (sometimes).
- The bot is a friendly chat helper named Mia. When asked its name it says "My name is Mia." It does not invent private facts about the user.
- The bot answers the question that was asked, directly and kindly.
{extra}
Output JSON Lines only (no code block, no numbering), one object per line: {{"turns": ["user text", "bot text", ...]}}"""


def ask(prompt, max_tokens=1400, temperature=0.9):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    u = d.get("usage", {})
    with lock:
        usage["prompt"] += u.get("prompt_tokens", 0)
        usage["completion"] += u.get("completion_tokens", 0)
    return d["choices"][0]["message"]["content"]


def vocab_check(turns):
    """自作トークナイザーで数える。(除外理由 or None, 本文トークン数, 最大分割数)"""
    n, worst = 0, 0
    for t in turns:
        n += len(tok.encode(t).ids) + 1          # +1: 役割トークン
        for w in t.split():
            w = w.strip(".,!?;:\"()")
            if not w:
                continue
            p = len(tok.encode(w).ids)
            worst = max(worst, p)
            if p > MAX_PIECES + (1 if "'" in w else 0):
                return f"rare-word:{w.lower()}", n, p
    if n > MAX_OWN_TOKENS:
        return f"too-long:{n}", n, worst
    return None, n, worst


def one(job):
    jid, cat, seed = job
    rnd = random.Random(seed)
    k = 8
    turns_desc = {"cmd": "2 or 4", "self": "2 or 4", "multi": "exactly 4 or 6", "qa": "2 or 4"}[cat]
    extra = ("- IMPORTANT: the LAST user turn must only make sense with an EARLIER turn (e.g. 'What is my name?', 'Why?', 'What else?', "
             "'Which one is bigger?'), and the bot's last answer must use that earlier information.") if cat == "multi" else ""
    txt = ask(PROMPT.format(k=k, cat_desc=base.CATS[cat], turns_desc=turns_desc,
                            topics=", ".join(rnd.sample(base.TOPICS, k)), extra=extra))
    return cat, [r.get("turns") for r in base.json_lines(txt) if isinstance(r, dict)]


def main(tag, target):
    rnd = random.Random(int(os.environ.get("SEED", 2026)))
    n_calls = max(1, round(target / (8 * 0.7)))      # 語彙・長さの除外が増えるので DeepSeek 版より多めに呼ぶ
    cats = list(base.CATS)
    jobs = [(i, cats[i % len(cats)], rnd.randrange(10**9)) for i in range(n_calls)]
    ok, rej, seen = [], [], set()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(futs):
            try:
                cat, res = f.result()
            except Exception as e:      # 通信エラー: そこまでの分は保存する
                print(f"[停止] {e}", file=sys.stderr)
                continue
            for turns in res:
                why = base.check(turns, seen, cat)
                n_tok = worst = None
                if not why:
                    why, n_tok, worst = vocab_check(turns)
                if why:
                    rej.append({"why": why, "cat": cat, "turns": turns})
                else:
                    seen.add(base.norm(turns[0]))
                    ok.append({"cat": cat, "turns": [t.strip() for t in turns], "own_tokens": n_tok})
            print(f"{i + 1}/{len(futs)} 採用{len(ok)} 除外{len(rej)}", flush=True)
    for name, data in ((f"raw_{tag}.jsonl", ok), (f"rej_{tag}.jsonl", rej)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{tag}: 採用 {len(ok)} / 除外 {len(rej)}  qwenトークン prompt {usage['prompt']} / completion {usage['completion']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen_pilot", int(sys.argv[2]) if len(sys.argv) > 2 else 200)
