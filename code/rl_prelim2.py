# -*- coding: utf-8 -*-
"""GRPO予備実験(その3): 「カード1枚+質問」で、報酬 1/0 が学習信号として成り立つかを測る。学習はしない。
  A 合わないカード(Qwenが答えられないと判定した上位4枚): 「わからない」→1、答える→0
  B 正しいカード(学習で見た質問): 答えが見本と合う→1
  C 学習に使っていない281問: 正しいカード1枚→答え合致で1、まぎれ3枚(答えられるカードが混じる)の「わからない」率
各カードにつき貪欲1本+サンプル8本(温度1.0)。グループ内に正解が1つでもある割合、などを出す。
使い方: MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python rl_prelim2.py
出力 eval_results/rl_prelim2.json / rl_prelim2_wrong_answered.jsonl"""
import json, re, os, random, glob, collections
import numpy as np, torch
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from knowledge_kb import KB
from model import GPT, Config

random.seed(5); torch.manual_seed(5)
dev = "cuda"; G = 8; BLOCK = 256
QA = os.environ.get("QA", "ckpt/readqa_qB_raft.pt")
NA, NB = int(os.environ.get("NA", 300)), int(os.environ.get("NB", 300))
IDK = "i do not know"
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
STOP = set("a an the is are was were of to in on it its and or for with that this as at by be from has have can do does not".split())
stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
cw = lambda s: [stem(w) for w in norm(s).split() if w not in STOP]
is_idk = lambda a: norm(a).startswith(IDK)
def yn(s):
    w = norm(s).split(); return w[0] if w and w[0] in ("yes", "no") else None

# ---- データ
kb = KB(); idx = {norm(c): i for i, c in enumerate(kb.cards)}
ev = set()
for r in map(json.loads, open("data/kb_cov/measure_pilot_f.jsonl")):
    if r["v"] == "A" and r["flag"] != "conflict":
        e = norm(r["evidence"]); i = idx.get(e)
        if i is None:
            c = [j for c_, j in idx.items() if e in c_]; i = c[0] if c else None
        if i is not None: ev.add(i)
evc = {norm(kb.cards[i]) for i in ev}
pos = json.load(open("data/raft_pos.json"))
ng = [x for f in sorted(glob.glob("data/raft_nogold_*.jsonl")) for x in map(json.loads, open(f)) if x["v"] == "NO"]
ng = [x for x in ng if not any(norm(c) in evc for c in x["cards"])]
random.shuffle(ng)
A = [dict(q=x["q"], card=c) for x in ng[:NA] for c in x["cards"]]
Bp = [p for p in pos if p["g"] not in ev]; random.shuffle(Bp)
B = [dict(q=p["q"], card=p["card"], ref=p["a"]) for p in Bp[:NB]]
R = [json.loads(l) for l in open("data/kb_cov/raft_eval_raft.jsonl")]
C_gold = [dict(q=r["q"], card=r["gold_card"], exp=r["expected"], ans=r["answer"]) for r in R]
C_dis = [dict(q=r["q"], card=c) for r in R for c in r["gold4_cards"] if c != r["gold_card"]]
print("A", len(A), "B", len(B), "C_gold", len(C_gold), "C_distractor", len(C_dis), flush=True)

# ---- 読む役
tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
qa = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=BLOCK)).to(dev)
qa.load_state_dict(torch.load(QA, map_location=dev)["model"]); qa.eval()


@torch.no_grad()
def sample(prompt, g=G, max_new=60):
    head = [BOS, USER] + tok.encode(prompt).ids + [BOT]
    ids = torch.tensor([head] * (g + 1), device=dev)
    done = torch.zeros(g + 1, dtype=torch.bool, device=dev); out = [[] for _ in range(g + 1)]
    for _ in range(max_new):
        logits = qa(ids[:, -BLOCK:])[0][:, -1].float()
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        nxt[0] = logits[0].argmax()
        for i in range(g + 1):
            if not done[i]:
                if int(nxt[i]) == EOS: done[i] = True
                else: out[i].append(int(nxt[i]))
        ids = torch.cat([ids, nxt[:, None]], 1)
        if done.all(): break
    return [tok.decode(o).strip() for o in out]


def run(items, name):
    for k, it in enumerate(items):
        it["gen"] = sample(f"Facts: {it['card']} Question: {it['q']}")
        if (k + 1) % 400 == 0: print(name, k + 1, flush=True)
for items, n in ((A, "A"), (B, "B"), (C_gold, "C_gold"), (C_dis, "C_dis")): run(items, n)
del qa; torch.cuda.empty_cache()

st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=dev, model_kwargs={"torch_dtype": "float16"})
st.max_seq_length = 128; st.tokenizer.padding_side = "left"
cache = {}
def emb(texts):
    need = sorted({t for t in texts if t not in cache})
    if need:
        for t, v in zip(need, st.encode(need, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)): cache[t] = v
    return np.stack([cache[t] for t in texts])

def ok_B(it, a, thr):
    if is_idk(a): return 0
    ry = yn(it["ref"])
    if ry is not None and yn(a) != ry: return 0
    return int(float(emb([a])[0] @ emb([it["ref"]])[0]) >= thr)
def ok_C(it, a, mode="half"):
    if is_idk(a): return 0
    ry = yn(it["exp"])
    if ry is not None and yn(a) != ry: return 0
    ref = set(cw(it["ans"])); got = set(cw(a))
    if not ref: return 1 if ry is not None else 0
    f = len(ref & got) / len(ref)
    return int(f == 1 if mode == "all" else f >= 0.5)

def stats(items, fn):
    """fn(it, answer)->0/1。グループ=1カードにつきサンプル8本。"""
    ns = np.array([sum(fn(it, a) for a in it["gen"][1:]) for it in items])
    gd = np.array([fn(it, it["gen"][0]) for it in items])
    return dict(n_groups=len(items), greedy_rate=round(float(gd.mean()), 3), sample_rate=round(float(ns.mean() / G), 3),
                groups_with_1plus=round(float((ns >= 1).mean()), 3), groups_with_all8=round(float((ns == G).mean()), 3),
                groups_with_signal_1to7=round(float(((ns >= 1) & (ns < G)).mean()), 3),
                count_dist={int(k): round(float((ns == k).mean()), 3) for k in range(G + 1)})
idk_fn = lambda it, a: int(is_idk(a))
res = dict(model=QA,
           A_wrong_card_idk=stats(A, idk_fn),
           B_gold_card_seen_thr0_8=stats(B, lambda it, a: ok_B(it, a, 0.8)),
           B_gold_card_seen_thr0_9=stats(B, lambda it, a: ok_B(it, a, 0.9)),
           C_gold_card_unseen_half=stats(C_gold, lambda it, a: ok_C(it, a, "half")),
           C_gold_card_unseen_all=stats(C_gold, lambda it, a: ok_C(it, a, "all")),
           C_distractor_idk=stats(C_dis, idk_fn))
# 質問単位: Aは4枚とも「わからない」が1回でも出るか(質問ごとに見える信号の量)
qs = collections.defaultdict(list)
for it in A: qs[it["q"]].append(sum(is_idk(a) for a in it["gen"][1:]))
res["A_per_question_cards_with_idk_mean"] = round(float(np.mean([sum(v >= 1 for v in x) for x in qs.values()])), 2)
# 混合(1:1)で学習するときの信号つきグループの割合の目安
sigA = res["A_wrong_card_idk"]["groups_with_signal_1to7"]; sigB = res["B_gold_card_seen_thr0_8"]["groups_with_signal_1to7"]
res["mix_1to1_signal_groups"] = round((sigA + sigB) / 2, 3)
json.dump(res, open("eval_results/rl_prelim2.json", "w"), indent=1)
with open("eval_results/rl_prelim2_wrong_answered.jsonl", "w", encoding="utf-8") as f:
    for it in A[:600]:
        if not is_idk(it["gen"][0]): f.write(json.dumps(dict(q=it["q"], card=it["card"], greedy=it["gen"][0]), ensure_ascii=False) + "\n")
print(json.dumps(res, indent=1, ensure_ascii=False))
