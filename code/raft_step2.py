# RAFT用 手順2: 「正解なし」候補(正解カードを除いた上位4枚)を、Qwenが「どのカードでも答えられない」と判定したものだけ残す
# 使い方: python raft_step2.py START END  (pos の番号範囲)  → data/raft_nogold_START.jsonl に追記
import json, sys, random, re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa_local as local
from knowledge_kb import KB
kb = KB()
pos = json.load(open("data/raft_pos.json")); top = np.load("data/raft_top12.npy")
s, e = int(sys.argv[1]), int(sys.argv[2])
P = """A children's question-answering bot reads fact cards.
Question: {q}
Cards:
{cards}
Does ANY card contain the information needed to answer the question (even in different words)? Reply with one word: YES or NO."""
random.seed(s)
def job(i):
    p = pos[i]
    cs = [int(c) for c in top[i] if int(c) != p["g"]][:4]
    cards = [kb.cards[c] for c in cs]
    t = local.ask(P.format(q=p["q"], cards="\n".join(f"- {c}" for c in cards)), max_tokens=5, temperature=0)
    return dict(i=i, q=p["q"], cards=cards, v=("NO" if t.strip().upper().startswith("NO") else "YES"))
out = open(f"data/raft_nogold_{s}.jsonl", "a")
with ThreadPoolExecutor(16) as ex:
    for k, r in enumerate(ex.map(job, range(s, e))):
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
        if k % 500 == 0: print(k, flush=True)
