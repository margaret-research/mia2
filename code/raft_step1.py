# RAFT用 手順1: 学習データの質問ごとに、Qwen3-Embeddingで上位12枚のカード番号を求めて保存
import json, re, numpy as np, torch
from sentence_transformers import SentenceTransformer
from knowledge_kb import KB
kb = KB(); n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
idx = {n(c): i for i, c in enumerate(kb.cards)}
rows = [json.loads(l) for l in open("data/m_para_train.jsonl")]
pos = [dict(q=r["q"], a=r["answer"], card=r["card"], g=idx[n(r["card"])]) for r in rows if not r["idk"] and n(r["card"]) in idx]
print("positives", len(pos), flush=True)
D = torch.tensor(np.load("data/kb_emb_qwen3_0.6b.npy")).cuda().half()
st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={"torch_dtype": "float16"})
st.max_seq_length = 64; st.tokenizer.padding_side = "left"
T = "Given a question from a child, retrieve the fact card that answers it"
top = np.zeros((len(pos), 12), dtype=np.int32)
B = 512
for s in range(0, len(pos), B):
    e = st.encode([f"Instruct: {T}\nQuery:{p['q']}" for p in pos[s:s+B]], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).half()
    top[s:s+B] = (e @ D.T).topk(12, dim=1).indices.cpu().numpy()
    if (s // B) % 40 == 0: print(s, flush=True)
np.save("data/raft_top12.npy", top)
json.dump(pos, open("data/raft_pos.json", "w"))
print("done")
