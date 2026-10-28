# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-28) — 実機会話の強化学習 (GRPO型)。正解はなく、同じ状況の返事8本をDeepSeekが採点し、平均より良い返事を強める。
対象: 現行Miaの別コピー (会話の返事 + カードを読む返事)。ルーター/電卓/答え役は固定。基準テスト1・2と実機会話テストは別に測る。
場面: 新しく作る (雑談の発言は dialogues_train_expanded、カードは学習側の題名のみ、計算は gen_router_data)。実機会話テスト (benchmark/live_set.json) と同じ題材、読解テストの50項目は除く。
1ステップ: G_N 場面の途中まで進め (モデル自身の返事で)、次の返事を同じ状況から G 本サンプル → DeepSeek 採点 (実機会話テストと同じ20点式、カードは本文を見せ、計算は結果が正しいかを見る) → 群内標準化の利得で方策勾配 + 元モデルとのKL。
使い方 (bash -ic): COST_CAP=5 MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 MIA_TOK=data/ts_v4/tokenizer_ext.json PYTHONPATH=. python rl_live.py --steps 100 --out ckpt/mia_rl1"""
import argparse, copy, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor
import torch, torch.nn.functional as F
import gen_chat_qa as ds
import gen_router_data as R
from bot_mia2 import Bot
from card_retrieve import EmbIndex
from gen_card_data import mentions, load_entries

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=100); ap.add_argument("--groups", type=int, default=8); ap.add_argument("--g", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--beta", type=float, default=0.02); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--replay", type=int, default=0, help="1ステップごとに読解/会話の学習データから各N本の復習 (教師あり損失)"); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--save-every", type=int, default=25); ap.add_argument("--out", default="ckpt/mia_rl1"); ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--ckpt-in", default="ckpt/mia_current_read.pt"); ap.add_argument("--log", default="logs/rl_live.jsonl")
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)
ds.COST_CAP = float(os.environ.get("COST_CAP", 5.0))
dev = "cuda"; G = args.g

# ---------------- 場面 ----------------
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
live_text = " ".join(u for v in json.load(open("benchmark/live_set.json"))["scenarios"].values() for u in v)
cards = load("data/router/card_train.jsonl")          # 学習側の題名のみ (テスト用150題名・読解テストの50項目は含まれない)
by_t = {}
for c in cards:
    if c["type"] in ("card_q", "card_follow"): by_t.setdefault(c["title"], {"q": [], "f": []})["q" if c["type"] == "card_q" else "f"].append(c["user"])
TITLES = [t for t, v in by_t.items() if v["q"] and not mentions(t, live_text)]
convs = [r["turns"][0::2] for r in load("data/dialogues_train_expanded.jsonl") if len(r["turns"]) >= 4]
ASKS = ["Tell me about {t}.", "What is {a} {t}?", "Do you know what {a} {t} is?", "Tell me more about the {t}.", "You know {t}?", "Can you explain {t}?", "what is {t}", "Hey Mia, tell me about {t}."]
art = lambda t: "an" if t[0] in "aeiou" else "a"

def messy(u):
    return re.sub(r"[.?!]+$", "", u.lower()) if random.random() < 0.2 else u

def make_scenario():
    users, kinds = [], []
    for u in random.choice(convs)[:random.choice([1, 2, 3])]: users.append(u); kinds.append("chat")
    for _ in range(random.choice([1, 1, 2])):
        t = random.choice(TITLES); v = by_t[t]
        users.append(random.choice(ASKS).format(t=t, a=art(t))); kinds.append("card")
        users.append(random.choice(v["q"])); kinds.append("card")
        if v["f"] and random.random() < 0.6: users.append(random.choice(v["f"])); kinds.append("card")
        if random.random() < 0.5: users.append(random.choice(random.choice(convs))); kinds.append("chat")
    if random.random() < 0.4:
        u, e = random.choice(R.CALC_TYPES)(); users.append(u if isinstance(u, str) else u[-1]); kinds.append("calc")
    for u in random.choice(convs)[:random.choice([1, 2])]: users.append(u); kinds.append("chat")
    return [messy(u) for u in users], kinds

# ---------------- モデル ----------------
bot = Bot()                                           # ルーター・答え役は固定。チャットは方策に差し替える
bot.kb = EmbIndex(bot.dev)
policy = copy.deepcopy(bot.chat).train(False)
ref = copy.deepcopy(bot.chat).eval()
for p in ref.parameters(): p.requires_grad = False
for n, p in policy.named_parameters():
    if "wte" in n or "head" in n: p.requires_grad = False
bot.chat = policy
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
BOS, EOS, USER, BOT = bot.BOS, bot.EOS, bot.USER, bot.BOT

@torch.no_grad()
def sample(prompt, n=G, maxn=60):
    ids = torch.tensor([prompt[-960:]] * n, device=dev); done = torch.zeros(n, dtype=torch.bool, device=dev); outs = [[] for _ in range(n)]; fin = [False] * n
    for _ in range(maxn):
        lg, _ = policy(ids); nx = torch.multinomial(torch.softmax(lg[:, -1] / args.temp, -1), 1)[:, 0]
        for i in range(n):
            if done[i]: continue
            x = int(nx[i])
            if x in (EOS, USER): done[i] = True; fin[i] = x == EOS
            else: outs[i].append(x)
        if bool(done.all()): break
        ids = torch.cat([ids, nx[:, None]], 1)
    return outs, fin

# ---------------- 採点 (DeepSeek) ----------------
RUBRIC = open("benchmark/eval_live.py", encoding="utf-8").read().split('PROMPT = """')[1].split('"""')[0]
head = RUBRIC.split("Score EACH")[0].replace("Below are {n} exchanges; the USER lines are fixed, the MIA lines are Mia's replies.", "Below is a conversation so far, then the user's latest message, then {n} different candidate replies from Mia. Score each candidate reply on its own.")
JP = head + """Score EACH of the {n} candidates:
- "answer" 0-8: responds to the latest message and what it says is correct (check the PAGE or CALCULATOR line if given). Typical child = 6-7. 8 only if it also adds a reason or care. 0-2 if it ignores the question, is off topic, or repeats the question back.
- "memory" 0-6: consistent with earlier turns (names, facts, numbers; knows what "it" or "they" means). Typical child = 5 (also 5 when nothing earlier is needed).
- "language" 0-4: easy, short, understandable English. Typical child = 4 (3 if a little awkward). 0-1 if it does not make sense.
- "clean" 0-2: no repeated words or sentences, nothing unsuitable for a child.
Also flags per candidate: "ignored", "contradiction", "unsuitable", "wrong" (a wrong number, a wrong fact, a fact that is not in the PAGE although a PAGE was given, or an invented fact). Saying "I do not know." is honest but is "ignored" when the PAGE clearly contains the answer.
{conv}
Output only one JSON object on one line: {{"scores": [{{"answer": 0, "memory": 0, "language": 0, "clean": 0, "ignored": false, "contradiction": false, "unsuitable": false, "wrong": false}}, ... {n} items in the order of the candidates ...]}}"""

def one(x):
    v = min(8, max(0, int(x["answer"]))) + min(6, max(0, int(x["memory"]))) + min(4, max(0, int(x["language"]))) + min(2, max(0, int(x["clean"])))
    if x.get("wrong"): v = min(v, 8)
    if x.get("ignored"): v = min(v, 10)
    if x.get("contradiction"): v = min(v, 12)
    if x.get("unsuitable"): v = min(v, 8)
    return v

def judge(hist, user, cands, page=None):
    order = list(range(len(cands))); random.shuffle(order)
    conv = "".join(f"{'USER' if i % 2 == 0 else 'MIA'}: {t}\n" for i, t in enumerate(hist[-6:])) + f"USER (latest message): {user}\n"
    if page: conv += f"PAGE (the book page Mia was given; her answer should agree with it): {page}\n"
    conv += "Candidates:\n" + "".join(f"[{k + 1}] {cands[j] or '(empty)'}\n" for k, j in enumerate(order))
    for _ in range(2):
        try:
            t = ds.ask(JP.format(conv=conv, n=len(cands)), max_tokens=900, temperature=0.0)
            sc = json.loads(re.search(r"\{.*\}", t, re.S).group(0))["scores"]; assert len(sc) == len(cands)
            r = [0] * len(cands)
            for k, j in enumerate(order): r[j] = one(sc[k])
            return r
        except RuntimeError: raise
        except Exception: continue
    return None

# ---------------- 状態を作る ----------------
def make_group():
    for _ in range(50):
        users, kinds = make_scenario(); cand = [i for i, k in enumerate(kinds) if k in ("chat", "card")]
        t = random.choice(cand[1:] if len(cand) > 1 and random.random() < 0.8 else cand)
        bot.turns = []
        for u in users[:t]: bot.say(u)
        u = users[t]; h = bot.turns[-6:]; h = h[1:] if len(h) % 2 else h
        saved, bot.turns = bot.turns, h
        out = bot.gen(bot.router, bot.ctx(u), 40); bot.turns = saved
        if out.split()[:1] == ["card"]:
            title = (bot.kb.top(out[4:].strip(), u, 1) or [None])[0]
            if title is None: continue
            page = bot.kb.ents[title]["entry"]
            prompt = [BOS, USER] + bot.tok.encode(f"Title: {title}. {page} Question: {u}").ids + [BOT]
            return dict(prompt=prompt, hist=list(saved), user=u, page=f"Title: {title}. {page}", kind="card")
        if out.startswith("calc"): continue
        bot.turns = h; prompt = bot.ctx(u); bot.turns = saved
        return dict(prompt=prompt, hist=list(saved), user=u, page=None, kind="chat")
    raise RuntimeError("group")

def logp_of(model, prompt, reply, fin, grad):
    tgt = reply + ([EOS] if fin else [])
    if not tgt: return None
    ids = torch.tensor([(prompt[-960:] + tgt)], device=dev)
    with torch.set_grad_enabled(grad):
        lg, _ = model(ids[:, :-1]); lp = F.log_softmax(lg[0] / args.temp, -1)
    pos = torch.arange(len(tgt), device=dev) + (ids.shape[1] - 1 - len(tgt))
    return lp[pos, ids[0, -len(tgt):]]

# ---------------- 復習データ (読解/会話の教師あり) ----------------
import numpy as np
def load_npz(d):
    z = np.load(f"{d}/sft_train.npz"); return z["ids"], z["w"], z["off"]
REPLAY = [load_npz("data/read_mix_v5"), load_npz("data/chat_mix_v4")] if args.replay else []

def replay_loss():
    tot = 0.0; n = args.replay * len(REPLAY)
    for ids, w, off in REPLAY:
        for _ in range(args.replay):
            i = random.randrange(len(off) - 1); a, b = int(off[i]), int(off[i + 1])
            x = torch.tensor(ids[a:b][:1024], dtype=torch.long, device=dev)[None]; wt = torch.tensor(w[a:b][:1024], dtype=torch.float32, device=dev)[1:]
            lg, _ = policy(x[:, :-1]); ce = F.cross_entropy(lg[0], x[0, 1:], reduction="none")
            l = args.lam * (ce * wt).sum() / wt.sum().clamp(min=1) / n
            l.backward(); tot += float(l)
    return tot

# ---------------- 学習 ----------------
os.makedirs("logs", exist_ok=True); logf = open(args.log, "a", encoding="utf-8"); t0 = time.time()
try:
    for step in range(1, args.steps + 1):
        gs = [make_group() for _ in range(args.groups)]
        smp = [sample(g["prompt"]) for g in gs]
        texts = [[bot.tok.decode(o).strip() for o in outs] for outs, _ in smp]
        with ThreadPoolExecutor(args.groups) as ex:
            rs = list(ex.map(lambda a: judge(a[0]["hist"], a[0]["user"], a[1], a[0]["page"]), zip(gs, texts)))
        loss_n, tot_loss, sig, rmean = 0, 0.0, 0, []
        opt.zero_grad()
        for g, (outs, fin), tx, r in zip(gs, smp, texts, rs):
            if r is None: continue
            rm = sum(r) / len(r); sd = (sum((x - rm) ** 2 for x in r) / len(r)) ** 0.5; rmean.append(rm)
            if sd < 0.75: continue
            sig += 1
            for o, f, x in zip(outs, fin, r):
                lp = logp_of(policy, g["prompt"], o, f, True)
                if lp is None: continue
                with torch.no_grad(): rl = logp_of(ref, g["prompt"], o, f, False)
                adv = (x - rm) / (sd + 1e-6)
                l = (-adv * lp + args.beta * (lp - rl)).mean() / (G * args.groups)
                l.backward(); tot_loss += float(l); loss_n += 1
        rep = replay_loss() if args.replay else 0.0
        if loss_n or args.replay:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        rec = {"step": step, "reward": round(sum(rmean) / max(len(rmean), 1), 2), "signal_groups": sig, "n_groups": len(gs), "cost": round(ds.cost, 3), "replay_loss": round(rep, 3), "sec": int(time.time() - t0)}
        if step % 5 == 0 or step == 1:
            g0, tx0, r0 = gs[0], texts[0], rs[0]
            rec["example"] = {"user": g0["user"], "kind": g0["kind"], "best": tx0[max(range(G), key=lambda i: r0[i])] if r0 else None, "worst": tx0[min(range(G), key=lambda i: r0[i])] if r0 else None, "r": r0}
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush(); print(rec["step"], rec["reward"], rec["signal_groups"], rec["cost"], rec["sec"], flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": policy.state_dict()}, f"{args.out}_s{step}.pt")
except RuntimeError as e:
    print("停止:", e)
    torch.save({"model": policy.state_dict()}, f"{args.out}_stop.pt")
print("費用", round(ds.cost, 3))
