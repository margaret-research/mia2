# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — ルーターMiaの「カード」(項目)拡張: 質問 → `card 題名` のデータを作る
出力: data/router/card_{train,test}.jsonl  {"hist","user","target":"CARD 題名","type":card_q/card_ask/card_follow/card_neg, "title"}
  card_q 項目の質問(題名を含むもの) / card_ask 「Tell me about X」型 / card_follow 代名詞の質問(履歴に題名) / card_neg 題名を含むが事実を聞かない発言(→CHAT)
  評価用に 読解テストの50項目+ランダム100題名 = 150題名 を学習から外す (新しい題材へ一般化できるかを見る)。
使い方: python gen_card_data.py  ※先にvLLM(Qwen)を起動"""
import glob, json, os, random, re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local

D = "data/router"
rnd = random.Random(31)


def load_entries():
    ents = {}
    for f in sorted(glob.glob("data/reading_entry/full*/raw3_*.jsonl")):
        for l in open(f):
            r = json.loads(l); ents.setdefault(r["title"], r)
    qas = defaultdict(list)
    for t, r in ents.items():
        qas[t] += [x for x in r["qas"]]
    for f in glob.glob("data/reading_entry/extra/qa_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["title"] in ents: qas[r["title"]] += r["qas"]
    return ents, qas


def mentions(t, text):
    low = text.lower(); t = t.lower()
    stems = {t, t + "s", t + "es"}
    if t.endswith("y"): stems.add(t[:-1] + "ies")
    if t.endswith("f"): stems.add(t[:-1] + "ves")
    if t.endswith("fe"): stems.add(t[:-2] + "ves")
    if t.endswith("s"): stems.add(t[:-1])
    return any(re.search(rf"\b{re.escape(s)}\b", low) for s in stems)


ASK = ["Tell me about {a} {t}.", "What is {a} {t}?", "Can you tell me about {a} {t}?", "Please tell me about {t}.", "I want to learn about {a} {t}.",
       "Do you know about {a} {t}?", "Tell me something about {a} {t}.", "what is {a} {t}"]
ART = lambda t: "an" if t[0] in "aeiou" else "a"
FOLLOW = """Rewrite this question so that it does NOT say the words "{t}". Use "it", "they", "this" or "that" instead, as in a talk where {t} was just discussed. Keep the meaning. Easy English, plain ASCII, one line only.
Question: {q}"""
NEG = """A child chats with a friendly helper named Mia. Write 5 different things the child says that mention "{t}" but do NOT ask for facts about it. Use: opinions, feelings, wishes, personal stories, or questions to Mia about herself (like "Do you like {t}s?", "I want to see a {t}.", "Have you ever seen a {t}?"). Very easy words, plain ASCII, one per line, no numbering."""


def follow_job(job):
    t, qa, qb = job["title"], job["a"], job["b"]
    try:
        txt = local.ask(FOLLOW.format(t=t, q=qb["q"]), max_tokens=80, temperature=0.7).strip().split("\n")[0].strip().strip('"')
    except Exception:
        return None
    if not txt or not txt.isascii() or mentions(t, txt) or not txt.endswith("?") or len(txt.split()) > 18:
        return None
    return {"hist": [qa["q"], qa["a"]], "user": txt, "target": f"CARD {t}", "type": "card_follow", "title": t}


def neg_job(t):
    try:
        txt = local.ask(NEG.format(t=t), max_tokens=250, temperature=0.9)
    except Exception:
        return []
    out = []
    for line in txt.strip().split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
        if line and line.isascii() and mentions(t, line) and len(line.split()) <= 20:
            out.append({"hist": [], "user": line, "target": "CHAT", "type": "card_neg", "title": t})
    return out


def main():
    ents, qas = load_entries()
    val = {e["title"] for e in json.load(open("benchmark/read_val_entries.json"))}
    rest = sorted(set(ents) - val); rnd.shuffle(rest)
    test_t = sorted(val | set(rest[:100])); train_t = [t for t in ents if t not in set(test_t)]
    print("題名", len(ents), "学習", len(train_t), "テスト", len(test_t), flush=True)
    rows = {"train": [], "test": []}
    for split, titles, per in (("train", train_t, 6), ("test", test_t, 7)):
        for t in titles:
            ok = [x for x in qas[t] if mentions(t, x["q"])]; rnd.shuffle(ok)
            for x in ok[:per]:
                rows[split].append({"hist": [], "user": x["q"], "target": f"CARD {t}", "type": "card_q", "title": t, "qtype": x["type"]})
            for f in rnd.sample(ASK, 2 if split == "train" else 3):
                rows[split].append({"hist": [], "user": f.format(a=ART(t), t=t), "target": f"CARD {t}", "type": "card_ask", "title": t})
    # 代名詞の質問 (Qwen)
    fj = {"train": [], "test": []}
    for split, titles, n in (("train", train_t, 2200), ("test", test_t, 330)):
        for _ in range(n):
            t = rnd.choice(titles); ok = qas[t]
            a = [x for x in ok if mentions(t, x["q"]) and x["a"] != "I do not know."]
            b = [x for x in ok if x["type"] != "unk"]
            if a and len(b) > 1:
                fj[split].append({"title": t, "a": rnd.choice(a), "b": rnd.choice(b)})
    with ThreadPoolExecutor(local.WORKERS) as ex:
        for split in fj:
            rows[split] += [r for r in ex.map(follow_job, fj[split]) if r]
        for split, titles, n in (("train", train_t, 800), ("test", test_t, 100)):
            for ps in ex.map(neg_job, rnd.sample(titles, min(n, len(titles)))):
                rows[split] += ps
    for split in rows:
        with open(f"{D}/card_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[split]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows[split]), dict(Counter(r["type"] for r in rows[split])))


if __name__ == "__main__":
    main()
