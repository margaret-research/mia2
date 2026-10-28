# RAFT用 手順3: 学習データを組み立てる
#  正解あり: Facts: (正解カード+まぎれ2〜4枚を混ぜる) Question: q → 正解カードに基づく答え(m_para_trainの答え)
#  正解なし: Qwenが「どのカードでも答えられない」と判定した候補(上位4枚) → I do not know.
# 評価用の正解カード(281問の根拠)は、まぎれカードにも使わない。
import json, re, random, glob, sys
import numpy as np
from knowledge_kb import KB
random.seed(11)
kb = KB(); n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
idx = {n(c): i for i, c in enumerate(kb.cards)}
pos = json.load(open("data/raft_pos.json")); top = np.load("data/raft_top12.npy")
# 評価用の正解カード番号(eval_reader_k.py と同じ求め方)
ev = set()
for r in map(json.loads, open("data/kb_cov/measure_pilot_f.jsonl")):
    if r["v"] == "A" and r["flag"] != "conflict":
        e = n(r["evidence"]); i = idx.get(e)
        if i is None:
            c = [j for c_, j in idx.items() if e in c_]; i = c[0] if c else None
        if i is not None: ev.add(i)
print("評価用の正解カード", len(ev))
ng = [json.loads(l) for f in sorted(glob.glob("data/raft_nogold_*.jsonl")) for l in open(f)]
ng = [x for x in ng if x["v"] == "NO"]
seen = set(); ng2 = []
for x in ng:
    if x["i"] in seen: continue
    seen.add(x["i"]); ng2.append(x)
ng = ng2
print("正解なし候補(NO判定)", len(ng))

def mk(prompt_cards, q): return "Facts: " + " ".join(prompt_cards) + " Question: " + q
def row(p, cards, ans, idk, typ):
    return dict(prompt=mk(cards, p["q"]), answer=ans, idk=idk, type=typ, name="", q=p["q"], card=" | ".join(cards))

N_GOLD, N_NO, T_GOLD, T_NO = int(sys.argv[1]), int(sys.argv[2]), 1500, 500
order = list(range(len(pos))); random.shuffle(order)
gold_rows = []
for i in order:
    p = pos[i]
    if p["g"] in ev: continue
    pool = [int(c) for c in top[i][:7] if int(c) != p["g"] and int(c) not in ev]
    k = random.choice([2, 3, 3, 4])
    if len(pool) < k: continue
    cs = [p["card"]] + [kb.cards[c] for c in random.sample(pool, k)]
    random.shuffle(cs)
    gold_rows.append(row(p, cs, p["a"], False, "gold"))
    if len(gold_rows) >= N_GOLD + T_GOLD: break
random.shuffle(ng)
no_rows = []
for x in ng:
    if any(n(c) in {n(kb.cards[e]) for e in ev} for c in x["cards"]): continue
    cs = list(x["cards"]); random.shuffle(cs)
    no_rows.append(row(pos[x["i"]], cs, "I do not know.", True, "nogold"))
    if len(no_rows) >= N_NO + T_NO: break
print("gold", len(gold_rows), "nogold", len(no_rows))
tr = gold_rows[T_GOLD:] + no_rows[T_NO:]; random.shuffle(tr)
te = gold_rows[:T_GOLD] + no_rows[:T_NO]
with open("data/raft_train.jsonl", "w") as f:
    for r in tr: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("data/raft_test.jsonl", "w") as f:
    for r in te: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train", len(tr), "test", len(te))
