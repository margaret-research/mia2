# -*- coding: utf-8 -*-
"""T-026 会話データを「語彙の中の単語だけ」で生成する (vLLM の文法制約つきデコード)

自作トークナイザー(tokenizer_ext.json)で「空白付き1トークン」になる英単語(7千語台)だけを許す文法を作り、
vLLM の structured_outputs(grammar) に渡す。文頭の大文字は「その形のまま語彙にあるもの」だけ許す(厳密)。
実在しない断片(caut, microsc ...)は /usr/share/dict/american-english で除く。1〜2文字の語は固定の許可リストだけ。
1回の呼び出しで1会話を作る(型 cat と話題 topic をプロンプトで指定)。
出力: data/chat_qa/raw_<tag>.jsonl / rej_<tag>.jsonl   (gen_chat_qa_local.py と同じ形式)
使い方: python gen_chat_qa_gbnf.py gbnf_pilot 200      (先に vLLM サーバーを起動しておく)
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
import gen_chat_qa_local as local

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT_OK = set("a I am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we ha".split())
SHORT_OK |= {"OK", "Ok", "Mm", "Mr", "Ms"}
NAMES = {"Mia", "Tom", "Sam", "Ben", "Max", "Tim", "Anna", "Amy", "Lily", "Jack", "Mom", "Dad"}   # 名前はこれだけ許す
SUFFIXES = ["s", "m", "t", "d", "re", "ll", "ve"]          # 短縮形: word + ' + suffix (don't = don + 't)


def allowed_words():
    toks = local.tok.get_vocab()
    dic = {w.strip() for w in open("/usr/share/dict/american-english", encoding="utf-8", errors="replace")}
    dic_low = {w.lower() for w in dic if w.isalpha()}
    dic_lc = {w for w in dic if w.isalpha() and w.islower()}      # 小文字の形でも辞書にある語 (人名・地名だけの語を除く)
    words = set()
    for k in toks:
        if not (k.startswith("Ġ") and re.fullmatch(r"[A-Za-z]+", k[1:])):
            continue
        w = k[1:]
        if len(w) <= 2:
            if w in SHORT_OK:
                words.add(w)
        elif w in NAMES or w.lower() in dic_lc:
            words.add(w)
    # 短縮形の前半(don, isn, can ...)は辞書にない形も多いので、'tが続く語は語彙にあれば許す
    for k in toks:
        if k.startswith("Ġ") and k[1:].endswith("n") and re.fullmatch(r"[A-Za-z]+", k[1:]) and k[1:-1].lower() in dic_low:
            words.add(k[1:])
    return sorted(words)


def lit(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_grammar(n_pairs):
    """n_pairs=(lo, hi): ユーザー+Bot の組を lo〜hi 組(= 2lo〜2hi ターン)"""
    words = allowed_words()
    alt = " | ".join(lit(w) for w in words)
    suf = " | ".join(lit(s) for s in SUFFIXES)
    lo, hi = n_pairs
    body = " ".join(['pair'] + ['(", " pair)'] * (lo - 1) + ['(", " pair)?'] * (hi - lo))
    g = f"""root ::= "{{\\"turns\\": [" {body} "]}}"
pair ::= utter ", " bot
utter ::= "\\"" turn ender? "\\""
bot ::= "\\"" turn ender "\\""
turn ::= clause (ender " " clause){{0,2}}
clause ::= seg (", " seg){{0,2}}
seg ::= word (" " word){{0,14}}
ender ::= "." | "!" | "?"
word ::= base ("'" ({suf}))?
base ::= {alt}
"""
    return g, len(words)


PROMPT = """You write one short English chat example for training a very small chatbot.
The chatbot speaks like a kind helper for children in grade 1-2 (age 6-8).
Type of the conversation: {cat_desc}
Topic hint: {topic}
Rules:
- The conversation has {turns_desc} turns, alternating USER, BOT, USER, BOT ... It starts with USER and ends with BOT.
- Use only very easy words that a 7-year-old knows. Short sentences. Each turn is 1-2 sentences, under 20 words.
- The bot is a friendly chat helper named Mia. When asked its name it says "My name is Mia." It does not invent private facts about the user.
- The bot answers the question that was asked, directly and kindly.
{extra}
Output one JSON object only: {{"turns": ["user text", "bot text", ...]}}"""

GRAMMARS = {}
CONTENT_WORDS = []
# EXT=1 のとき: 話題を増やし、ユーザーの話し方のヒントも毎回変える (重複する最初の発言を減らすため)
EXT = os.environ.get("EXT") == "1"
TOPICS_EXT = ("birthday|bath time|the zoo|rain|snow|a rainbow|the moon|stars|the sun|a bus ride|a picnic|a toy box|a kite|a puzzle|"
              "bedtime|breakfast|lunch|a lost shoe|a new friend|the park|the playground|painting|a bird|a fish|a dog|a cat|a farm|"
              "a cow|a tree|flowers|a garden|the beach|the sea|a boat|a train|a car|a plane|the library|a story book|counting|"
              "colors|shapes|letters|a game of tag|hide and seek|a ball|a bike|a swing|ice cream|cake|apples|bread|milk|soup|"
              "a cold|a doctor|brushing teeth|washing hands|a hat|a coat|shoes|a school bag|a teacher|homework|a lunch box|"
              "a clock|a robot|a king|a dragon|a rabbit|a bear|a frog|a duck|a horse|a pig|a bee|a butterfly|the wind|a cloud|"
              "a river|a hill|a house|a door|a bed|a chair|a cup|a boat trip|a haircut|a gift|a song|a dance|a drum|a joke").split("|")
STYLES = ["very short and polite", "uses the word please", "all lowercase with no final mark", "excited, with an exclamation mark",
          "curious, asks why", "a little shy", "asks for help", "gives a short command", "says a fact first, then asks"]


def preload_seen():
    """SEEN_FROM=tag1,tag2 の raw_/rej_ にある最初の発言を、重複扱いにする"""
    seen = set()
    for tag in filter(None, os.environ.get("SEEN_FROM", "").split(",")):
        for pre in ("raw_", "rej_"):
            p = os.path.join(base.OUT, f"{pre}{tag}.jsonl")
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    t = json.loads(line).get("turns")
                    if isinstance(t, list) and t and isinstance(t[0], str):
                        seen.add(base.norm(t[0]))
    return seen


def one(job):
    jid, cat, seed = job
    rnd = random.Random(seed)
    pairs = (2, 3) if cat == "multi" else (1, 2)
    turns_desc = "exactly 4 or 6" if cat == "multi" else "2 or 4"
    extra = ("- IMPORTANT: the LAST user turn must only make sense with an EARLIER turn (e.g. 'What is my name?', 'Why?', 'What else?', "
             "'Which one is bigger?'), and the bot's last answer must use that earlier information.") if cat == "multi" else ""
    topic = rnd.choice(TOPICS_EXT + base.TOPICS) if EXT else rnd.choice(base.TOPICS)
    if EXT:
        extra += ("\n" if extra else "") + f"- The user's first turn sounds like this: {rnd.choice(STYLES)}."
    if CONTENT_WORDS:       # WORDSEED=1: 許可語彙からランダムに1語を「使ってみて」とヒントに出す (話題の組み合わせを実質無限にする)
        extra += ("\n" if extra else "") + f"- If it fits naturally, try to use this word: {rnd.choice(CONTENT_WORDS)}."
    prompt = PROMPT.format(cat_desc=base.CATS[cat], topic=topic, turns_desc=turns_desc, extra=extra)
    body = json.dumps({"model": local.MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 400, "temperature": 1.0, "top_p": 0.95, "seed": seed % (2**31),
                       "chat_template_kwargs": {"enable_thinking": False},
                       "structured_outputs": {"grammar": GRAMMARS[pairs]}}).encode()
    import urllib.request
    req = urllib.request.Request(local.URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    try:
        return cat, json.loads(d["choices"][0]["message"]["content"]).get("turns")
    except ValueError:
        return cat, None


def main(tag, target):
    for pairs in ((1, 2), (2, 3)):
        GRAMMARS[pairs], n = build_grammar(pairs)
    print(f"許可した単語 {n} 語 / 文法 {len(GRAMMARS[(1, 2)]) // 1000} KB", flush=True)
    if os.environ.get("WORDSEED") == "1":
        CONTENT_WORDS.extend(w for w in allowed_words() if w.islower() and len(w) >= 4 and w.isalpha())
        print(f"ヒント用の単語 {len(CONTENT_WORDS)} 語", flush=True)
    rnd = random.Random(int(os.environ.get("SEED", 2026)))
    n_calls = int(target * 1.2)
    cats = list(base.CATS)
    jobs = [(i, cats[i % len(cats)], rnd.randrange(10**9)) for i in range(n_calls)]
    ok, rej, seen, t0 = [], [], preload_seen(), time.time()
    with ThreadPoolExecutor(local.WORKERS) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(futs):
            try:
                cat, turns = f.result()
            except Exception as e:
                print(f"[停止] {e}", file=sys.stderr)
                continue
            why = base.check(turns, seen, cat) if turns is not None else "bad-json"
            n_tok = None
            if not why:
                why, n_tok, _ = local.vocab_check(turns)
            if why:
                rej.append({"why": why, "cat": cat, "turns": turns})
            else:
                seen.add(base.norm(turns[0]))
                ok.append({"cat": cat, "turns": [t.strip() for t in turns], "own_tokens": n_tok})
            if (i + 1) % 20 == 0 or i + 1 == len(futs):
                print(f"{i + 1}/{len(futs)} 採用{len(ok)} 除外{len(rej)} {time.time() - t0:.0f}s", flush=True)
    for name, data in ((f"raw_{tag}.jsonl", ok), (f"rej_{tag}.jsonl", rej)):
        with open(os.path.join(base.OUT, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{tag}: 採用 {len(ok)} / 除外 {len(rej)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gbnf_pilot", int(sys.argv[2]) if len(sys.argv) > 2 else 200)
