# -*- coding: utf-8 -*-
"""MIA2 読解SFT用データ: 読解項目(data/reading_entry/full/raw3_*.jsonl)+会話の復習分を混ぜて整形する

読解の1サンプル = 項目1つ + 質問5問(unk 1問 + 他4問)の多ターン会話。
  1ターン目 user: "Title: <題名>. <本文> Question: <質問1>" / bot: 答え / 2ターン目以降 user: 質問 / bot: 答え  (損失は答えと<eos>だけ)
除く: Qwen検証が bad の質問 / 検証結果なしの項目 / count の答えに数(数字か数の語)がない質問
検証用: 50項目を学習から外し、1項目につき固定の5問サンプルを2つ。
会話の復習: data/chat_mix の train から短い会話 N_SHORT + 長い会話 N_LONG をランダムに選ぶ。検証は chat_mix の val から1,000会話。
出力: data/read_mix/sft_{train,val}.npz  (val は 先頭 N_RV 件が読解、残りが会話)
"""
import glob, json, os, random, re
import numpy as np
from build_chat_sft import encode

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "read_mix_v4"); os.makedirs(OUT, exist_ok=True)
random.seed(1)
PER_ENTRY, N_VAL_ENTRY, N_SHORT, N_LONG = 6, 50, 90000, 18000   # 項目が約2倍になったので読解は12,000サンプル前後
NUMW = r"\d|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|half|both|once|twice|first|second|third"

R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "data/reading_entry/full*/raw3_*.jsonl"))) for l in open(f, encoding="utf-8")]
ent, dropped = [], 0
for r in R:
    if not r.get("verdict") or len(r["verdict"]) != len(r["qas"]):
        dropped += 1; continue
    qs = []
    for q, v in zip(r["qas"], r["verdict"]):
        if v != "ok": continue
        if q["type"] == "count" and not re.search(r"\b(" + NUMW + r")\b|\d", q["a"].lower()): continue
        qs.append(q)
    if sum(q["type"] == "unk" for q in qs) >= 1 and len(qs) >= 8:
        ent.append({"title": r["title"], "entry": r["entry"], "qs": qs})
    else:
        dropped += 1
print(f"項目 {len(R)} → 採用 {len(ent)} (除外 {dropped}) / 質問 {sum(len(e['qs']) for e in ent)}")
random.shuffle(ent)
VT = {e["title"] for e in json.load(open(os.path.join(HERE, "data/read_mix/val_entries.json")))}
ev, et = [e for e in ent if e["title"] in VT], [e for e in ent if e["title"] not in VT]
print("検証項目", len(ev), "学習項目", len(et))


def sample(e, k=5):
    unk = [q for q in e["qs"] if q["type"] == "unk"]
    oth = [q for q in e["qs"] if q["type"] != "unk"]
    pick = [random.choice(unk)] + random.sample(oth, min(k - 1, len(oth)))
    random.shuffle(pick)
    T = []
    for i, q in enumerate(pick):
        T.append((f"Title: {e['title']}. {e['entry']} Question: {q['q']}" if i == 0 else q["q"]))
        T.append(q["a"])
    return T


def write(name, convs_ids):
    ids, w, off = [], [], [0]
    for i, ww in convs_ids:
        ids += list(i); w += list(ww); off.append(len(ids))
    np.savez(os.path.join(OUT, f"sft_{name}.npz"), ids=np.array(ids, np.uint16), w=np.array(w, np.uint8), off=np.array(off, np.int64))
    print(f"{name}: {len(convs_ids)} 会話 {len(ids) / 1e6:.2f}Mトークン (損失対象 {int(np.sum(np.array(w, np.int64))) / 1e6:.2f}M)")


def chat(name, n_short, n_long, n_short_total):
    z = np.load(os.path.join(HERE, f"data/chat_mix_v4/sft_{name}.npz")); ids, w, off = z["ids"], z["w"], z["off"]
    n = len(off) - 1
    idx = list(range(n_short_total)); lg = list(range(n_short_total, n))
    sel = random.sample(idx, n_short) + random.sample(lg, n_long)
    return [(ids[off[i]:off[i + 1]], w[off[i]:off[i + 1]]) for i in sel]


rtr = [encode(sample(e)) for e in et for _ in range(PER_ENTRY)]
rva = [encode(sample(e)) for e in ev for _ in range(2)]
write("train", rtr + chat("train", N_SHORT, N_LONG, 326115))
write("val", rva + chat("val", 500, 500, 3000))
json.dump(ev, open(os.path.join(OUT, "val_entries.json"), "w"), ensure_ascii=False)
print("読解 val 件数 N_RV =", len(rva), " 最長トークン", max(len(i) for i, _ in rtr + rva))
