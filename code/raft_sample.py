# RAFT風サンプル(見せるだけ): 正解カード+検索で上位に出たまぎれカード、または正解なし
import json, re, random, numpy as np, torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from knowledge_kb import KB
random.seed(3)
kb = KB(); n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
idx = {n(c): i for i, c in enumerate(kb.cards)}
D = np.load("data/kb_emb_qwen3_0.6b.npy")
rows = [json.loads(l) for l in open("data/m_para_train.jsonl")]
pos = [r for r in rows if not r["idk"] and n(r["card"]) in idx]
S = random.sample(pos, 6)
st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={"torch_dtype": "float16"})
st.max_seq_length = 128; st.tokenizer.padding_side = "left"
TASK = "Given a question from a child, retrieve the fact card that answers it"
E = st.encode([f"Instruct: {TASK}\nQuery:{r['q']}" for r in S], normalize_embeddings=True, convert_to_numpy=True)
tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
out = []
for k, (r, e) in enumerate(zip(S, E)):
    g = idx[n(r["card"])]
    order = [int(i) for i in np.argsort(-(D @ e))[:12] if int(i) != g]
    dis = order[:4]
    withg = [kb.cards[g]] + [kb.cards[i] for i in dis[:3]]; random.shuffle(withg)
    nog = [kb.cards[i] for i in order[:4]]
    for tag, cards, ans in (("正解あり", withg, r["answer"]), ("正解なし", nog, "I do not know.")):
        p = "Facts: " + " ".join(cards) + " Question: " + r["q"]
        print(f"--- 例{k+1} [{tag}] ({len(tok.encode(p).ids)+len(tok.encode(ans).ids)}トークン)\n入力: {p}\n答え: {ans}\n")
