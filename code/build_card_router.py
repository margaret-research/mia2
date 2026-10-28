# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — ルーターMia v4 (calc / card / chat) のデータ: 計算+会話 (train_v2aug/test_v2aug) に card_*.jsonl を合流
card 行のうち履歴のない行の約3割には、前の計算・会話のやりとりを履歴としてつける (build_router_data.add_history)。
出力: data/router/train_v4.jsonl, test_v4.jsonl (+ build_router_sft.py に RV=v4 を渡すと data/router_sft_v4)"""
import json, random
import build_router_data as B

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

for split, seed, chat in (("train", 7, "data/dialogues_train_expanded.jsonl"), ("test", 8, "data/dialogues_val.jsonl")):
    base = load(f"data/router/{split}_v2aug.jsonl")
    card = load(f"data/router/card_{split}.jsonl")
    pool = [r for r in base if r['target'].startswith('CALC') and not r['hist']]
    card = B.add_history(card, chat, seed, p=0.3, pool=pool)
    rows = base + card
    random.Random(seed).shuffle(rows)
    with open(f"data/router/{split}_v4.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(split, len(rows), "(card系", len(card), ")")
