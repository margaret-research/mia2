# -*- coding: utf-8 -*-
"""GRPO(カード1枚+質問)。報酬は 1/0。
  A 合わないカード(Qwenが答えられないと判定): 「わからない」→1、答える→0
  B 正しいカード: 答えが見本と合う(yes/noの先頭語が一致 かつ Qwen3-Embedding類似度>=0.8)→1
各ステップ: A16問+B16問、1問につき8本サンプル(温度1.0)。グループ内の報酬の標準化を利得にして方策勾配を1回更新(オンポリシー、KLなし)。
信号のないグループ(8本とも同じ報酬)は更新に使わない。埋め込みは固定(これまでと同じ)。
使い方: MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python grpo_single.py --steps 300 --lr 1e-5 --out ckpt/grpo_s
評価(5ステップごとではなく--eval-everyごと): 貪欲生成で A(合わないカードの「わからない」率) B(見た質問の正解率) C(初見281問・正しいカードの正解率) Cdis(まぎれの「わからない」率)"""
import json, re, os, random, glob, argparse, time
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from knowledge_kb import KB
from model import GPT, Config

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-in", default="ckpt/readqa_qB_raft.pt"); ap.add_argument("--out", default="ckpt/grpo_s")
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--na", type=int, default=16); ap.add_argument("--nb", type=int, default=16); ap.add_argument("--g", type=int, default=8)
ap.add_argument("--thr", type=float, default=0.8); ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--data", default="old", choices=["old", "new", "hard", "mix"], help="new: 初見の質問(data/new_q/final.jsonl)で学習する(作成スクリプトは一式に含まない)")
ap.add_argument("--idk-pen", type=float, default=0.0, help="正しいカードで「わからない」と答えたときの罰(報酬 -idk_pen)")
ap.add_argument("--soft", action="store_true", help="Bの報酬を min(類似度/thr, 1) にする(評価のB_okは従来どおり0/1)")
ap.add_argument("--raw", action="store_true", help="--soft と併用: Bの報酬を類似度そのもの(上限なし)にする")
ap.add_argument("--seed", type=int, default=5); ap.add_argument("--log", default="logs/grpo_single.json")
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)
dev = "cuda"; BLOCK = 256; G = args.g
IDK = "i do not know"
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
nz_ = norm
STOP = set("a an the is are was were of to in on it its and or for with that this as at by be from has have can do does not".split())
stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
cw = lambda s: [stem(w) for w in norm(s).split() if w not in STOP]
is_idk = lambda a: norm(a).startswith(IDK)
def yn(s):
    w = norm(s).split(); return w[0] if w and w[0] in ("yes", "no") else None

# ---- データ(評価用の正解カードは使わない。検証用の質問は学習から外す)
kb = KB(); idx = {norm(c): i for i, c in enumerate(kb.cards)}
ev = set()
for r in map(json.loads, open("data/kb_cov/measure_pilot_f.jsonl")):
    if r["v"] == "A" and r["flag"] != "conflict":
        e = norm(r["evidence"]); i = idx.get(e)
        if i is None:
            c = [j for c_, j in idx.items() if e in c_]; i = c[0] if c else None
        if i is not None: ev.add(i)
evc = {norm(kb.cards[i]) for i in ev}
ng = [x for f in sorted(glob.glob("data/raft_nogold_*.jsonl")) for x in map(json.loads, open(f)) if x["v"] == "NO"]
ng = [x for x in ng if not any(norm(c) in evc for c in x["cards"])]
pos = [p for p in json.load(open("data/raft_pos.json")) if p["g"] not in ev]
random.shuffle(ng); random.shuffle(pos)
VA, VB = 200, 200
if args.data in ("new", "hard", "mix"):
    NQ = [json.loads(l) for l in open({"hard": "data/new_q_hard/final.jsonl", "mix": "data/new_q_mix/final.jsonl"}.get(args.data, "data/new_q/final.jsonl"))]; random.shuffle(NQ)
    NQ = [x for x in NQ if nz_(x["card"]) not in evc]
    ngq = [x for x in NQ if x["wrong"]]
    A_val = [dict(q=x["q"], card=x["wrong"][0]) for x in ngq[:VA]]; A_pool = [dict(q=x["q"], cards=x["wrong"]) for x in ngq[VA:]]
    B_val = [dict(q=x["q"], card=x["card"], ref=x["a"]) for x in NQ[:VB]]; B_pool = [dict(q=x["q"], card=x["card"], a=x["a"]) for x in NQ[VB:]]
    vq = {x["q"] for x in NQ[:VB]} | {x["q"] for x in ngq[:VA]}
    A_pool = [x for x in A_pool if x["q"] not in vq]; B_pool = [x for x in B_pool if x["q"] not in vq]
else:
    A_val = [dict(q=x["q"], card=x["cards"][0]) for x in ng[:VA]]; A_pool = ng[VA:]
    B_val = [dict(q=p["q"], card=p["card"], ref=p["a"]) for p in pos[:VB]]; B_pool = pos[VB:]
R = [json.loads(l) for l in open("data/kb_cov/raft_eval_raft.jsonl")]
C_gold = [dict(q=r["q"], card=r["gold_card"], exp=r["expected"], ans=r["answer"]) for r in R]
C_dis = [dict(q=r["q"], card=c) for r in R for c in r["gold4_cards"] if c != r["gold_card"]]
if os.environ.get("QUICK"): A_val, B_val, C_gold, C_dis = A_val[:10], B_val[:10], C_gold[:10], C_dis[:10]
print("A_pool", len(A_pool), "B_pool", len(B_pool), "val", VA, VB, "C", len(C_gold), len(C_dis), flush=True)

# ---- モデル
tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
model = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=BLOCK)).to(dev)
model.load_state_dict(torch.load(args.ckpt_in, map_location=dev)["model"]); model.eval()   # ドロップアウトなし(生成と勾配計算で同じ分布)
model.wte.weight.requires_grad_(False)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=dev, model_kwargs={"torch_dtype": "float16"})
st.max_seq_length = 128; st.tokenizer.padding_side = "left"
cache = {}
def emb(texts):
    need = sorted({t for t in texts if t not in cache})
    if need:
        for t, v in zip(need, st.encode(need, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)): cache[t] = v
    return np.stack([cache[t] for t in texts])


def head_ids(card, q): return [BOS, USER] + tok.encode(f"Facts: {card} Question: {q}").ids + [BOT]

@torch.no_grad()
def sample(head, g, max_new=60, greedy_only=False):
    """行0=貪欲、行1..g=温度1.0。戻り値: 生成トークン列のリスト(EOSで止まったものは末尾にEOSを含む)。"""
    ids = torch.tensor([head] * (g + 1), device=dev)
    done = torch.zeros(g + 1, dtype=torch.bool, device=dev); out = [[] for _ in range(g + 1)]
    for _ in range(max_new):
        logits = model(ids[:, -BLOCK:])[0][:, -1].float()
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1) if g > 0 else logits.argmax(-1)
        nxt[0] = logits[0].argmax()
        for i in range(g + 1):
            if not done[i]:
                out[i].append(int(nxt[i]))
                if int(nxt[i]) == EOS: done[i] = True
        ids = torch.cat([ids, nxt[:, None]], 1)
        if done.all(): break
    return out
dec = lambda o: tok.decode([t for t in o if t != EOS]).strip()


def reward(kind, it, a, binary=False):
    if kind == "A": return int(is_idk(a))
    if is_idk(a): return 0 if binary else -args.idk_pen
    ry = yn(it["ref"])
    if ry is not None and yn(a) != ry: return 0
    c = float(emb([a])[0] @ emb([it["ref"]])[0])
    if binary or not args.soft: return int(c >= args.thr)
    return c if args.raw else min(c / args.thr, 1.0)     # 連続報酬: raw=類似度そのもの / それ以外は 類似度/閾値 (上限1)
def ok_C(it, a):    # 初見281問の短い見本答え: 内容語が半分以上入る
    if is_idk(a): return 0
    ry = yn(it["exp"])
    if ry is not None and yn(a) != ry: return 0
    ref = set(cw(it["ans"])); got = set(cw(a))
    if not ref: return 1 if ry is not None else 0
    return int(len(ref & got) / len(ref) >= 0.5)


STOPQ = set("a an the is are was were of to in on it its and or for with that this as at by be from has have can do does not what which where why how who many".split())
def evaluate():
    r = {}
    gen = lambda it: dec(sample(head_ids(it["card"], it["q"]), 0)[0])
    ga = [gen(it) for it in A_val]; r["A_idk"] = round(float(np.mean([is_idk(a) for a in ga])), 3)
    gb = [gen(it) for it in B_val]; r["B_ok"] = round(float(np.mean([reward("B", it, a, binary=True) for it, a in zip(B_val, gb)])), 3)
    r["B_idk"] = round(float(np.mean([is_idk(a) for a in gb])), 3)
    gc = [gen(it) for it in C_gold]; r["C_ok"] = round(float(np.mean([ok_C(it, a) for it, a in zip(C_gold, gc)])), 3)
    r["C_idk"] = round(float(np.mean([is_idk(a) for a in gc])), 3)
    sub = lambda f: [k for k, it in enumerate(C_gold) if f(it)]
    for nm_, f in (("C_yn", lambda it: yn(it["exp"]) is not None), ("C_num", lambda it: yn(it["exp"]) is None and bool(re.search(r"\d", it["q"] + it["exp"]))),
                   ("C_para", lambda it: yn(it["exp"]) is None and len((set(norm(it["q"]).split()) - STOPQ) & set(norm(it["card"]).split())) / max(len(set(norm(it["q"]).split()) - STOPQ), 1) <= 1 / 3)):
        ks = sub(f); r[nm_ + "_ok"] = round(float(np.mean([ok_C(C_gold[k], gc[k]) for k in ks])), 3)   # C_yn:yes/no質問 / C_num:数字を含む質問 / C_para:言い換え(質問の語がカードに1/3以下)
    gd = [gen(it) for it in C_dis]; r["Cdis_idk"] = round(float(np.mean([is_idk(a) for a in gd])), 3)
    return r


log = []
def save(step): torch.save({"model": model.state_dict(), "step": step}, f"{args.out}{step}.pt")
ev0 = evaluate(); print("step 0", ev0, flush=True); log.append(dict(step=0, eval=ev0))
t0 = time.time()
for step in range(1, args.steps + 1):
    prompts = []
    for x in random.sample(A_pool, args.na): prompts.append(("A", dict(q=x["q"], card=random.choice(x["cards"]))))
    for p in random.sample(B_pool, args.nb): prompts.append(("B", dict(q=p["q"], card=p["card"], ref=p["a"])))
    seqs = []; stat = dict(rA=[], rB=[], sigA=[], sigB=[], len=[])
    for kind, it in prompts:
        head = head_ids(it["card"], it["q"]); outs = sample(head, G)[1:]
        rs = np.array([reward(kind, it, dec(o)) for o in outs], dtype=np.float32)
        stat["r" + kind].append(rs.mean()); stat["sig" + kind].append(float(rs.std() > 0)); stat["len"].append(np.mean([len(o) for o in outs]))
        if rs.std() == 0: continue
        adv = (rs - rs.mean()) / (rs.std() + 1e-4)
        for o, a_ in zip(outs, adv): seqs.append((head, o, float(a_)))
    rec = dict(step=step, rA=round(float(np.mean(stat["rA"])), 3), rB=round(float(np.mean(stat["rB"])), 3),
               sigA=round(float(np.mean(stat["sigA"])), 3), sigB=round(float(np.mean(stat["sigB"])), 3), len=round(float(np.mean(stat["len"])), 1), n_seq=len(seqs))
    if seqs:
        warm = min(1.0, step / 10)
        for gp in opt.param_groups: gp["lr"] = args.lr * warm
        n_tok = sum(len(o) for _, o, _ in seqs); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(seqs), 64):
            chunk = seqs[i:i + 64]; L = max(len(h) + len(o) for h, o, _ in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long, device=dev); m = torch.zeros(len(chunk), L, device=dev); adv = torch.zeros(len(chunk), 1, device=dev)
            for j, (h, o, a_) in enumerate(chunk):
                s = h + o; x[j, :len(s)] = torch.tensor(s); m[j, len(h) - 1:len(s) - 1] = 1.0; adv[j] = a_   # 位置t-1の予測がトークンtの確率
            lp = torch.log_softmax(model(x)[0][:, :-1].float(), -1)
            tgt = x[:, 1:]; tl = lp.gather(-1, tgt[..., None]).squeeze(-1)
            loss = -(adv * tl * m[:, :-1]).sum() / n_tok
            loss.backward(); tot += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
        if np.isfinite(gn): opt.step()
        else: print("勾配がNaN/Inf: このステップは更新しない", flush=True)
        rec.update(loss=round(tot, 4), grad_norm=round(gn, 3))
    log.append(rec)
    if step % 10 == 0: print(rec, f"{time.time()-t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        e = evaluate(); print("eval", step, e, flush=True); log.append(dict(step=step, eval=e)); save(step)
        json.dump(log, open(args.log, "w"), ensure_ascii=False)
json.dump(log, open(args.log, "w"), ensure_ascii=False)
print("done", flush=True)
