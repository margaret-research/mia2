# -*- coding: utf-8 -*-
"""GRPO予備実験: 学習はしない。RAFT後の読む役から、1問につき貪欲1本+サンプル8本を生成し、
報酬(引用+1 / 答え+1)の分布と、報酬判定(Qwen3-Embedding+ルール)の外れ方を測る。
引用B案: 答えの文と最も似ているカードを「根拠にしたカード」とみなす。
使い方: MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python rl_prelim.py
出力 eval_results/rl_prelim.json, eval_results/rl_prelim_samples.jsonl"""
import json, re, os, random, collections
import numpy as np, torch
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from model import GPT, Config

random.seed(5); torch.manual_seed(5)
dev = "cuda"; G = 8; BLOCK = 256
N_GOLD, N_NO = int(os.environ.get("N_GOLD", 400)), int(os.environ.get("N_NO", 200))
QA = os.environ.get("QA", "ckpt/readqa_qB_raft.pt")
IDK = "i do not know"
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
STOP = set("a an the is are was were of to in on it its and or for with that this as at by be from has have can do does not".split())
cw = lambda s: [w for w in norm(s).split() if w not in STOP]

# ---- データ: 学習に使っていないRAFTテスト行。正解カードは raft_pos.json から復元
rows = [json.loads(l) for l in open("data/raft_test.jsonl")]
pos = collections.defaultdict(list)
for p in json.load(open("data/raft_pos.json")): pos[p["q"]].append(p["card"])
items = []
for r in rows:
    cards = r["card"].split(" | ")
    if r["idk"]:
        items.append(dict(r, cards=cards, gold=None))
    else:
        g = [c for c in pos.get(r["q"], []) if c in cards]
        if g: items.append(dict(r, cards=cards, gold=cards.index(g[0])))
gold_i = [x for x in items if x["gold"] is not None]; no_i = [x for x in items if x["gold"] is None]
random.shuffle(gold_i); random.shuffle(no_i)
items = gold_i[:N_GOLD] + no_i[:N_NO]
print("gold行", min(N_GOLD, len(gold_i)), "正解なし行", min(N_NO, len(no_i)), flush=True)

# ---- 読む役
tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
qa = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=BLOCK)).to(dev)
qa.load_state_dict(torch.load(QA, map_location=dev)["model"]); qa.eval()


@torch.no_grad()
def sample(prompt, g=G, max_new=60):
    """行0は貪欲、行1..gは温度1.0のサンプル。"""
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


gens = []
for k, it in enumerate(items):
    gens.append(sample(it["prompt"]))
    if (k + 1) % 100 == 0: print("生成", k + 1, flush=True)
del qa; torch.cuda.empty_cache()

# ---- 埋め込み
st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=dev, model_kwargs={"torch_dtype": "float16"})
st.max_seq_length = 128; st.tokenizer.padding_side = "left"
cache = {}
def emb(texts):
    need = sorted({t for t in texts if t not in cache})
    if need:
        for t, v in zip(need, st.encode(need, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)): cache[t] = v
    return np.stack([cache[t] for t in texts])


def cite_emb(ans, cards): return int(np.argmax(emb(cards) @ emb([ans])[0]))
def cite_word(ans, cards):
    a = set(cw(ans)); return int(np.argmax([len(a & set(cw(c))) / max(len(a), 1) for c in cards]))
def is_idk(a): return norm(a).startswith(IDK)
def yn(s):
    w = norm(s).split(); return w[0] if w and w[0] in ("yes", "no") else None


def reward(it, ans, thr, cite_fn=cite_emb):
    """戻り値 (引用点, 答え点)。"""
    idk = is_idk(ans)
    if it["gold"] is None: return (1, 1) if idk else (0, 0)
    if idk: return (0, 0)
    c = 1 if cite_fn(ans, it["cards"]) == it["gold"] else 0
    ry = yn(it["answer"]); ok_pol = (ry is None) or (yn(ans) == ry)
    sim = float(emb([ans])[0] @ emb([it["answer"]])[0])
    return (c, 1 if (ok_pol and sim >= thr) else 0)


res = {}
# ---- (1) 答えの類似度: 正しい見本との類似 vs 別の質問の見本との類似(しきい値の目安)
gi = [k for k, it in enumerate(items) if it["gold"] is not None and not is_idk(gens[k][0])]
same = np.array([float(emb([gens[k][0]])[0] @ emb([items[k]["answer"]])[0]) for k in gi])
oth = np.array([float(emb([gens[k][0]])[0] @ emb([items[random.choice(gi)]["answer"]])[0]) for k in gi])
res["cos_same_pct"] = {p: round(float(np.percentile(same, p)), 3) for p in (5, 25, 50, 75, 95)}
res["cos_other_pct"] = {p: round(float(np.percentile(oth, p)), 3) for p in (5, 25, 50, 75, 95)}
res["thr_sweep"] = {t: dict(same_pass=round(float((same >= t).mean()), 3), other_pass=round(float((oth >= t).mean()), 3)) for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9)}

# ---- (2) 引用の推定: 貪欲の答え(IDKでない正解あり行)で、正解カードに当たる率
res["cite_acc_emb"] = round(float(np.mean([cite_emb(gens[k][0], items[k]["cards"]) == items[k]["gold"] for k in gi])), 3)
res["cite_acc_word"] = round(float(np.mean([cite_word(gens[k][0], items[k]["cards"]) == items[k]["gold"] for k in gi])), 3)
res["n_cite_eval"] = len(gi)
res["cite_chance"] = round(float(np.mean([1 / len(items[k]["cards"]) for k in gi])), 3)

# ---- (3) 報酬の分布(しきい値を変えて)
def summarize(thr):
    o = {}
    for typ, sel in (("gold", lambda it: it["gold"] is not None), ("nogold", lambda it: it["gold"] is None)):
        ks = [k for k, it in enumerate(items) if sel(it)]
        gr = [[sum(reward(items[k], a, thr)) for a in gens[k][1:]] for k in ks]
        gd = [sum(reward(items[k], gens[k][0], thr)) for k in ks]
        o[typ] = dict(greedy_mean=round(float(np.mean(gd)), 3), sample_mean=round(float(np.mean(gr)), 3),
                      idk_rate_sample=round(float(np.mean([is_idk(a) for k in ks for a in gens[k][1:]])), 3),
                      group_all_same=round(float(np.mean([len(set(x)) == 1 for x in gr])), 3),
                      group_std_mean=round(float(np.mean([np.std(x) for x in gr])), 3),
                      dist={s: round(float(np.mean([v == s for x in gr for v in x])), 3) for s in (0, 1, 2)})
    return o
res["reward"] = {t: summarize(t) for t in (0.7, 0.8, 0.85)}

# ---- (4) 抜け穴の確認: 常にIDK／正解カードの丸写し／まぎれカードの丸写し
thr = 0.8; base = {}
base["always_idk"] = round(float(np.mean([sum(reward(it, "I do not know.", thr)) for it in items])), 3)
gk = [it for it in items if it["gold"] is not None]
base["copy_gold_card"] = round(float(np.mean([sum(reward(it, it["cards"][it["gold"]], thr)) for it in gk])), 3)
base["copy_distractor_card"] = round(float(np.mean([sum(reward(it, random.choice([c for j, c in enumerate(it["cards"]) if j != it["gold"]]), thr)) for it in gk])), 3)
base["reference_answer"] = round(float(np.mean([sum(reward(it, it["answer"], thr)) for it in gk])), 3)
res["baselines_mean_reward_thr0.8"] = base
res["mean_reward_over_all_items_greedy_thr0.8"] = round(float(np.mean([sum(reward(it, gens[k][0], thr)) for k, it in enumerate(items)])), 3)

json.dump(res, open("eval_results/rl_prelim.json", "w"), indent=1)
with open("eval_results/rl_prelim_samples.jsonl", "w", encoding="utf-8") as f:
    for k, it in enumerate(items):
        f.write(json.dumps(dict(q=it["q"], type="nogold" if it["gold"] is None else "gold", ref=it["answer"],
                                gold_card=None if it["gold"] is None else it["cards"][it["gold"]], cards=it["cards"],
                                greedy=gens[k][0], samples=gens[k][1:],
                                greedy_cite=None if is_idk(gens[k][0]) else cite_emb(gens[k][0], it["cards"]),
                                gold_idx=it["gold"], greedy_reward=reward(it, gens[k][0], 0.8)), ensure_ascii=False) + "\n")
print(json.dumps(res, indent=1, ensure_ascii=False))
