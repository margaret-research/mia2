# -*- coding: utf-8 -*-
"""ルーター(振り分け)の精度チェック: 入力を 1 会話(そのまま返す) / 2 知識(カードを見る) / 3 計算 のどれに回すかを測る。
Bot(bot_v3.py)と同じ判定: 計算ルーター(router_heads_ext_nohist.pt)が式を取れたら計算 → 知識ルーター確率>=0.5 かつ「Bot自身への質問」でなければ知識 → それ以外は会話。
評価の問題: 計算=calc_test(1,041)+手書きプローブ / 知識=RAFT評価の281問+学習に使っていない新しい質問 / 会話=会話検証データの発言+手書き。出力 eval_results/router_eval.json と router_eval_errors.jsonl
使い方: python router_eval.py"""
import json, re, random, collections, ast, torch
from router_ext import Router
from know_router_ext import KnowRouter

random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
R = lambda f: [json.loads(l) for l in open(f, encoding="utf-8")]
items = []   # (期待, 出どころ, 発言)
# 計算
for r in R("data/calc_test.jsonl"):   # callsあり=計算(式を呼ぶ発言)。neg/negh=数字入りの雑談で、計算に回してはいけない(期待=会話)
    if r["calls"]: items.append(("calc", "calc_test:" + r["cat"], r["turns"][int(next(iter(r["calls"]))) - 1]))
    else: items.append(("chat", "calc_test:" + r["cat"], r["turns"][-2]))
src = open("eval_chat.py", encoding="utf-8").read()
i0 = src.index("ITEMS = ["); i1 = src.index("\n]\n", i0) + 3
ns = {"IDK_RE": r"do not know|don't know"}; exec(src[i0:i1], ns)
for cat, turns, route, _ in ns["ITEMS"]:
    if len(turns) == 1 and route in ("chat", "know", "calc"): items.append((route, "eval_chat:" + cat, turns[0]))
# 知識: RAFT評価の281問 + 新しい質問(難しい版・yes/no・数字から)
for r in R("data/kb_cov/raft_eval_raft.jsonl"): items.append(("know", "know:C281", r["q"]))
for f, n in (("data/new_q_hard/final.jsonl", 400), ("data/new_q/final.jsonl", 300), ("data/new_q_num/final.jsonl", 200)):
    rows = R(f); random.shuffle(rows)
    for r in rows[:n]: items.append(("know", "know:" + f.split("/")[1], r["q"]))
# 会話: 会話の検証データ(学習時と同じ条件=あいさつ・自分のこと・話し手自身に関わる発言・質問でない発言)
PRON = re.compile(r"\b(i|my|me|mine|we|our|you|your|yours|mia)\b", re.I)
rows = R("data/chat_qa/chat_clean_val.jsonl")
us = [(r["cat"], t) for r in rows for t in r["turns"][0::2] if r["cat"] in ("greet", "self") or PRON.search(t) or not t.rstrip().endswith("?")]
random.shuffle(us)
for cat, t in us[:1500]: items.append(("chat", "chat_val:" + cat, t))
# 会話の紛らわしい手書き(数字入りの発言、事実っぽいが会話のもの)
HAND = ["I have 3 cats and 2 dogs.", "I am 12 years old.", "Meet me at 5 o'clock.", "My phone number is 555-1234.", "I will stay for 3-4 days.",
        "Let us meet on the 5th.", "The movie starts at 7:30.", "I like number 7.", "Thank you very much!", "What is your favorite food?",
        "Do you like cats?", "How was your day?", "Can you help me with my homework?", "Tell me a joke.", "What is your name?", "Where do you live?"]
for t in HAND: items.append(("chat", "chat:手書き", t))
print("問題数", collections.Counter(e for e, _, _ in items), flush=True)

calc = Router(dev, "ckpt/router_heads_ext_nohist.pt", nohist=True)
know = KnowRouter(calc.tok, calc.model, dev)
def route(u):
    out = calc.route([u])
    if out is not None: return "calc", 1.0
    p = know.prob(u)
    about_bot = bool(re.search(r"\b(you|your|yours)\b", u.lower())) and not re.search(r"\b(know|tell)\b", u.lower())
    return ("know" if p >= 0.5 and not about_bot else "chat"), p
res = [(e, s, u) + route(u) for e, s, u in items]
L = ["chat", "know", "calc"]
conf = {e: {g: sum(1 for x in res if x[0] == e and x[3] == g) for g in L} for e in L}
out = {"confusion(行=期待,列=判定)": conf, "per_class_acc": {e: round(conf[e][e] / max(sum(conf[e].values()), 1), 3) for e in L},
       "overall": round(sum(1 for x in res if x[0] == x[3]) / len(res), 3)}
bysrc = collections.defaultdict(lambda: [0, 0])
for e, s, u, g, p in res: bysrc[s][0] += e == g; bysrc[s][1] += 1
out["by_source"] = {k: f"{v[0]}/{v[1]} = {v[0] / v[1]:.3f}" for k, v in sorted(bysrc.items())}
json.dump(out, open("eval_results/router_eval.json", "w"), indent=1, ensure_ascii=False)
with open("eval_results/router_eval_errors.jsonl", "w", encoding="utf-8") as f:
    for e, s, u, g, p in res:
        if e != g: f.write(json.dumps(dict(expected=e, got=g, src=s, text=u, p_know=round(p, 3)), ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False))
