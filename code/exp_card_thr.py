import json, sys, numpy as np
sys.path.insert(0, ".")
from card_retrieve import EmbIndex
idx = EmbIndex()
def top1(q):
    v = idx.st.encode([f"Instruct: {idx.TASK}\nQuery:{q}"], normalize_embeddings=True, convert_to_numpy=True)[0]
    return float((idx.E @ v).max())
inkb = [q["q"] for e in json.load(open("benchmark/read_val_entries.json")) for q in e["qs"]]
nokb = [json.loads(l)["user"] for l in open("data/router/card2_test.jsonl") if '"card_nokb"' in l]
a = np.array([top1(q) for q in inkb]); b = np.array([top1(q) for q in nokb])
print("項目の質問", len(a), "分位 5/25/50/75:", np.percentile(a, [5, 25, 50, 75]).round(3))
print("項目にない", len(b), "分位 25/50/75/95:", np.percentile(b, [25, 50, 75, 95]).round(3))
for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
    print(f"しきい値 {thr:.2f}: 項目の質問を誤って捨てる {np.mean(a < thr):.3f} / 項目にない質問を捨てられる {np.mean(b < thr):.3f}")
