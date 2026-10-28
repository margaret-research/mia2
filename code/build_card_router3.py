# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — ルーターMia v6: train_v5 に card3_train を合流 → train_v6.jsonl / test_v6.jsonl (= test_v5 + card3_test)。
凍結テスト test_v5.jsonl は変えない。v6 の新しい弱点用の行は test_v6new.jsonl (card3_test のみ) にも出す。"""
import json, random
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
tr = load("data/router/train_v5.jsonl") + load("data/router/card3_train.jsonl"); random.Random(11).shuffle(tr)
new = load("data/router/card3_test.jsonl"); te = load("data/router/test_v5.jsonl") + new; random.Random(12).shuffle(te)
for name, rows in (("train_v6", tr), ("test_v6", te), ("test_v6new", new)):
    with open(f"data/router/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))
