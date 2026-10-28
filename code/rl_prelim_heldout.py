# -*- coding: utf-8 -*-
"""GRPO予備実験(その2): 学習に使っていない281問(評価用の正解カードは学習から除外済み)で測る。
正解カード+まぎれ3枚(raft_eval_raft.jsonl の gold4_cards)を1つのプロンプトにして、
貪欲1本+サンプル8本を生成。①引用の推定(埋め込み/語の重なり)の当たり率、
②答えのルール判定とQwen判定(raft_eval_judge_raft.json)の一致、③報酬の分布 を調べる。
使い方: MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python rl_prelim_heldout.py
出力 eval_results/rl_prelim_heldout.json / _samples.jsonl"""
import json, re, os, random
import numpy as np, torch
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from model import GPT, Config

random.seed(5); torch.manual_seed(5)
dev = "cuda"; G = 8; BLOCK = 256
QA = os.environ.get("QA", "ckpt/readqa_qB_raft.pt")
IDK = "i do not know"
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
STOP = set("a an the is are was were of to in on it its and or for with that this as at by be from has have can do does not".split())
stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
cw = lambda s: [stem(w) for w in norm(s).split() if w not in STOP]

R = [json.loads(l) for l in open("data/kb_cov/raft_eval_raft.jsonl")]
Jq = {r["q"]: x["j"] for r, x in zip(R, json.load(open("data/kb_cov/raft_eval_judge_raft.json")))}
items = []
for r in R:
    cards = r["gold4_cards"]
    items.append(dict(q=r["q"], cards=cards, gold=cards.index(r["gold_card"]), exp=r["expected"], ans=r["answer"],
                      prompt="Facts: " + " ".join(cards) + " Question: " + r["q"], old_greedy=r["gold4"], qj=Jq[r["q"]]))

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


gens = [sample(it["prompt"]) for it in items]
del qa; torch.cuda.empty_cache()
res = dict(n=len(items), greedy_same_as_old=round(float(np.mean([gens[k][0] == it["old_greedy"] for k, it in enumerate(items)])), 3))

st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=dev, model_kwargs={"torch_dtype": "float16"})
st.max_seq_length = 128; st.tokenizer.padding_side = "left"
cache = {}
def emb(texts):
    need = sorted({t for t in texts if t not in cache})
    if need:
        for t, v in zip(need, st.encode(need, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)): cache[t] = v
    return np.stack([cache[t] for t in texts])
def cite_emb(a, cards): return int(np.argmax(emb(cards) @ emb([a])[0]))
def cite_word(a, cards):
    s = set(cw(a)); return int(np.argmax([len(s & set(cw(c))) / max(len(s), 1) for c in cards]))
def is_idk(a): return norm(a).startswith(IDK)
def yn(s):
    w = norm(s).split(); return w[0] if w and w[0] in ("yes", "no") else None

# 答えのルール: 見本の短い答え(ans)の内容語が、生成した答えに「全部/半分以上/1つでも」入っているか。yes/noは先頭語の一致
def rule(it, a, mode):
    if is_idk(a): return 0
    ry = yn(it["exp"])
    if ry is not None and yn(a) != ry: return 0
    ref = set(cw(it["ans"])); got = set(cw(a))
    if not ref: return 1 if ry is not None else 0
    f = len(ref & got) / len(ref)
    return int(f == 1 if mode == "all" else f >= 0.5 if mode == "half" else f > 0)

# ---- ① 引用の推定(貪欲の答え, IDKでない行)
ks = [k for k, it in enumerate(items) if not is_idk(gens[k][0])]
res["n_not_idk_greedy"] = len(ks); res["greedy_idk_rate"] = round(1 - len(ks) / len(items), 3)
res["cite_acc_emb"] = round(float(np.mean([cite_emb(gens[k][0], items[k]["cards"]) == items[k]["gold"] for k in ks])), 3)
res["cite_acc_word"] = round(float(np.mean([cite_word(gens[k][0], items[k]["cards"]) == items[k]["gold"] for k in ks])), 3)
res["cite_chance"] = 0.25

# ---- ② 答えのルール判定とQwen判定の一致(貪欲の答えが以前の答えと同じものだけ。Qwen判定は以前の答えに付いている)
same = [k for k in ks if gens[k][0] == items[k]["old_greedy"] and items[k]["old_greedy"] in items[k]["qj"]]
res["n_compare_with_qwen"] = len(same)
qC = np.array([items[k]["qj"][items[k]["old_greedy"]] == "C" for k in same])
res["qwen_C_rate"] = round(float(qC.mean()), 3)
for mode in ("all", "half", "any"):
    r_ = np.array([rule(items[k], gens[k][0], mode) for k in same]).astype(bool)
    res[f"rule_{mode}"] = dict(pass_rate=round(float(r_.mean()), 3), agree=round(float((r_ == qC).mean()), 3),
                               pass_when_qwenC=round(float(r_[qC].mean()), 3), pass_when_qwenW=round(float(r_[~qC].mean()), 3))

# ---- ③ 報酬の分布(引用=埋め込み、答え=ルール)
def rw(it, a, mode):
    if is_idk(a): return 0
    return int(cite_emb(a, it["cards"]) == it["gold"]) + rule(it, a, mode)
for mode in ("all", "half"):
    gr = [[rw(items[k], a, mode) for a in gens[k][1:]] for k in range(len(items))]
    res[f"reward_{mode}"] = dict(greedy_mean=round(float(np.mean([rw(items[k], gens[k][0], mode) for k in range(len(items))])), 3),
                                 sample_mean=round(float(np.mean(gr)), 3),
                                 idk_rate_sample=round(float(np.mean([is_idk(a) for g in gens for a in g[1:]])), 3),
                                 group_all_same=round(float(np.mean([len(set(x)) == 1 for x in gr])), 3),
                                 group_std_mean=round(float(np.mean([np.std(x) for x in gr])), 3),
                                 dist={s: round(float(np.mean([v == s for x in gr for v in x])), 3) for s in (0, 1, 2)})
res["baseline_always_idk"] = 0.0   # 正解ありだけの評価集合なので、常にIDKは0点
res["baseline_copy_gold_card_half"] = round(float(np.mean([rw(it, it["cards"][it["gold"]], "half") for it in items])), 3)
res["baseline_copy_distractor_half"] = round(float(np.mean([rw(it, random.choice([c for j, c in enumerate(it["cards"]) if j != it["gold"]]), "half") for it in items])), 3)

json.dump(res, open("eval_results/rl_prelim_heldout.json", "w"), indent=1)
with open("eval_results/rl_prelim_heldout_samples.jsonl", "w", encoding="utf-8") as f:
    for k, it in enumerate(items):
        f.write(json.dumps(dict(q=it["q"], exp=it["exp"], ans=it["ans"], gold_card=it["cards"][it["gold"]], greedy=gens[k][0], samples=gens[k][1:],
                                greedy_cite=None if is_idk(gens[k][0]) else cite_emb(gens[k][0], it["cards"]), gold_idx=it["gold"],
                                qwen=it["qj"].get(gens[k][0])), ensure_ascii=False) + "\n")
print(json.dumps(res, indent=1, ensure_ascii=False))
