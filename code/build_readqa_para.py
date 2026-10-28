# -*- coding: utf-8 -*-
"""言い換え質問(data/kb_para/qa_raw.jsonl) → 読む役の追加学習ファイル data/m_para_train.jsonl
 ・正例: Facts: カード Question: 質問 → 答えの文 (今回の281問の正解カードは学習から除く = 評価の漏れを防ぐ)
 ・IDK例: 同じ題材の別カード(質問の語と答えの語が重ならないもの)と、別の題材のカードを半々。答えは "I do not know."
 ・古い学習データ m120k_train.jsonl (IDK 31%) を丸ごと足す
使い方: python build_readqa_para.py"""
import json, random, re, collections
R = [json.loads(l) for l in open("data/kb_para/qa_raw.jsonl", encoding="utf-8")]
n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
gold = {n(json.loads(l)["gold_card"]) for l in open("data/kb_cov/reader_k.jsonl", encoding="utf-8")}
R = [r for r in R if n(r["card"]) not in gold]
STOP = set("a an the is are was were of to in on at and or it its they them he she you your we our do does did can what which who where when why how many much this that these those with for from by as be has have had not no yes".split())
cw = lambda s: {w[:5] for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}
rnd = random.Random(5)
bysub = collections.defaultdict(list)
for r in R:
    bysub[r["subject"]].append(r["card"])
cards = sorted({r["card"] for r in R})
subs = [s for s in bysub]
out = []
for r in R:
    t = "yes" if r["a"].startswith("Yes") else "no" if r["a"].startswith("No,") else "wh"
    out.append(dict(prompt=f"Facts: {r['card']} Question: {r['q']}", answer=r["a"], idk=False, type=t, domain=r["domain"], subject=r["subject"], name="", q=r["q"], card=r["card"]))
n_pos = len(out)
neg = 0
target = int(n_pos * 0.18)
qs = [r for r in R if r["kind"] in (1, 2)]
rnd.shuffle(qs)
for r in qs:
    if neg >= target:
        break
    qw, aw = cw(r["q"]), cw(r["a"])
    if rnd.random() < 0.5:
        pool = [c for c in bysub[r["subject"]] if c != r["card"]]
    else:
        pool = [rnd.choice(cards) for _ in range(6)]
    rnd.shuffle(pool)
    for c in pool[:8]:
        if c == r["card"]:
            continue
        cc = cw(c)
        if len(aw & cc) / max(1, len(aw)) < 0.3 and len(qw & cc) <= 1:   # 質問にも答えにも関係がうすいカード
            out.append(dict(prompt=f"Facts: {c} Question: {r['q']}", answer="I do not know.", idk=True, type="mis", domain=r["domain"], subject=r["subject"], name="", q=r["q"], card=c))
            neg += 1
            break
old = [json.loads(l) for l in open("data/m120k_train.jsonl", encoding="utf-8")]
allr = out + old
rnd.shuffle(allr)
with open("data/m_para_train.jsonl", "w", encoding="utf-8") as f:
    for x in allr:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
idk = sum(x["idk"] for x in allr)
print(f"新: 正例 {n_pos} + IDK {neg} / 古: {len(old)} → 合計 {len(allr)} (IDK {idk} = {idk/len(allr):.0%}) 除外した評価用カード {len(gold)}")
