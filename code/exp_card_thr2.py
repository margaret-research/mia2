import json, sys, itertools, numpy as np
sys.path.insert(0, ".")
from card_retrieve import EmbIndex
from card_stem import stem_words
idx = EmbIndex()
def best(q):
    v = idx.st.encode([f"Instruct: {idx.TASK}\nQuery:{q}"], normalize_embeddings=True, convert_to_numpy=True)[0]
    s = idx.E @ v; j = int(s.argmax()); return float(s[j]), idx.titles[j]
def lex(topic, title):
    body = set(stem_words(title + " " + idx.ents[title]["entry"])); w = [x for x in stem_words(topic)]
    return bool(w) and all(x in body for x in w)
R = json.load(open("logs/card_routed.json"))
A = []
for r in R:
    if r["routed"].split()[:1] == ["card"]:
        s, t = best(r["q"]); A.append((s, lex(r["routed"][4:].strip(), t)))
N = []
for l in open("data/router/card2_test.jsonl"):
    r = json.loads(l)
    if r["type"] == "card_nokb":
        s, t = best(r["user"]); N.append((s, lex(r["title"], t)))
print("項目の質問: 話題が本文に出る率", np.mean([x[1] for x in A]).round(3), "/ 項目にない: 本文に出る率", np.mean([x[1] for x in N]).round(3))
print(f"{'lo':>5}{'hi':>5}  項目の質問を捨てる  項目にない質問を捨てる")
for lo, hi in itertools.product((0.40, 0.45, 0.50), (0.55, 0.60, 0.65)):
    acc = lambda L: np.mean([not ((l and s >= lo) or s >= hi) for s, l in L])
    print(f"{lo:5.2f}{hi:5.2f}  {acc(A):.3f}  {acc(N):.3f}")
