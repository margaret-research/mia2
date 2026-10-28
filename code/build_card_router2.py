# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — ルーターMia v5: train_v4/test_v4 に card2_*.jsonl (card_switch, card_nokb) を合流 → train_v5/test_v5.jsonl"""
import json, random
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
for split, seed in (("train", 9), ("test", 10)):
    rows = load(f"data/router/{split}_v4.jsonl") + load(f"data/router/card2_{split}.jsonl")
    random.Random(seed).shuffle(rows)
    with open(f"data/router/{split}_v5.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(split, len(rows))
