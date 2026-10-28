# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-07) — T-026 読解SFT(動物に限らない): 「Facts(カード) + 質問 → 答えの文」を学習し、複数の評価セットで測る
train_cardqa.py (2026-10-06) を元に、評価を型別・名前の書き写し別に変更。--max-eval で評価件数を絞る。

LLM全体を sft_5k_v2.pt から追加学習する (T-025方式Bと同じ)。損失は答えのトークン(と <eos>)だけにかける。
通常会話の能力が落ちていないかを見るため、普通の会話 (dialogues_train_expanded) を混ぜ、通常会話のval_lossも出す。

使い方 (ローカルでもColabでも同じ):
  python train_cardqa.py --data DIR --ckpt-in sft_5k_v2.pt --ckpt-out cardqa_5k.pt
      --train-file cardqa2_train.jsonl --tests cardqa2_test_real.jsonl cardqa2_test_fic.jsonl ...
v2 の追加: --train-file / --tests (複数の評価セット)、動物名の取り違え率(subject)を数える。
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
ap.add_argument("--ckpt-in", default="ckpt/sft_5k_v2.pt")
ap.add_argument("--ckpt-out", default="ckpt/cardqa_5k.pt")
ap.add_argument("--train-file", default="cardqa_sft_train.jsonl")
ap.add_argument("--tests", nargs="+", default=["cardqa_sft_test.jsonl"])
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=1.5e-4)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--plain", type=int, default=2500, help="混ぜる普通の会話の数(1エポックあたり)")
ap.add_argument("--smoke", action="store_true", help="配管確認だけ(数step)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-eval", type=int, default=600, help="評価セットごとの最大件数(貪欲生成は遅いので)")
ap.add_argument("--log", default=None, help="途中経過(学習の10/25/50/100%%時点のval_loss)をJSONで保存")
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def find(name):
    cands = [args.data] if args.data else []
    cands += ["data", "data/animal_kb", "."]
    for d in cands:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


sys.path.insert(0, args.data or ".")
from model import GPT, Config  # noqa: E402

tok = Tokenizer.from_file(find("tokenizer_5k.json"))
tok.add_special_tokens(["<user>", "<bot>", "<memory>"])
BOS, EOS = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
USER, BOT = tok.token_to_id("<user>"), tok.token_to_id("<bot>")
BLOCK = 256


def read(name):
    return [json.loads(l) for l in open(find(name), encoding="utf-8")]


def encode_qa(prompt, answer):
    """[BOS] <user> prompt <bot> answer <eos>  損失は answer と <eos> だけ。"""
    p = [BOS, USER] + tok.encode(prompt).ids + [BOT]
    a = tok.encode(answer).ids + [EOS]
    return p + a, [0] * len(p) + [1] * len(a)


def encode_plain(turns):
    ids, w = [BOS], [0]
    for i, t in enumerate(turns):
        e = tok.encode(t).ids
        if i % 2 == 0:
            ids += [USER] + e
            w += [0] * (len(e) + 1)
        else:
            ids += [BOT] + e + [EOS]
            w += [0] + [1] * (len(e) + 1)
    return ids[:BLOCK], w[:BLOCK]


train = read(args.train_file)
tests = {t: read(t)[: args.max_eval] for t in args.tests}
plain_tr = [r["turns"] for r in read("dialogues_train_expanded.jsonl") if len(r["turns"]) <= 10]
plain_va = [r["turns"] for r in read("dialogues_val.jsonl") if len(r["turns"]) <= 10][:300]
if args.smoke:
    train, plain_tr, plain_va = train[:64], plain_tr[:64], plain_va[:32]
    tests = {k: v[:24] for k, v in tests.items()}
    args.plain, args.epochs = 32, 1

model = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=BLOCK)).to(dev)
model.load_state_dict(torch.load(args.ckpt_in, map_location=dev)["model"])
print("device", dev, "| train", len(train), "| tests", {k: len(v) for k, v in tests.items()}, "| plain", len(plain_tr),
      "| params(M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), flush=True)


def batchify(items):
    L = max(len(i) for i, _ in items)
    x = torch.zeros(len(items), L, dtype=torch.long)
    y = torch.full((len(items), L), -1, dtype=torch.long)
    for r, (ids, w) in enumerate(items):
        x[r, :len(ids)] = torch.tensor(ids)
        t = torch.tensor(ids[1:] + [0])
        m = torch.tensor(w[1:] + [0], dtype=torch.bool)
        y[r, :len(ids)] = torch.where(m, t, torch.tensor(-1))
    return x.to(dev), y.to(dev)


@torch.no_grad()
def val_loss(items, bs=32):
    model.eval()
    tot, n = 0.0, 0
    for b in range(0, len(items), bs):
        x, y = batchify(items[b:b + bs])
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="sum")
        tot += loss.item()
        n += (y != -1).sum().item()
    model.train()
    return tot / max(n, 1)


first_test = list(tests)[0]
test_items = [encode_qa(r["prompt"], r["answer"]) for r in tests[first_test]]
plain_val_items = [encode_plain(t) for t in plain_va]
print(f"開始前 val_loss  カードQA({first_test}): {val_loss(test_items):.4f} | 通常会話: {val_loss(plain_val_items):.4f}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
steps_per_epoch = math.ceil((len(train) + min(args.plain, len(plain_tr))) / args.bs)
total = steps_per_epoch * args.epochs
sched = lambda s: args.lr * min(1.0, (s + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / total)))
step, t0 = 0, time.time()
MARKS = {max(1, round(total * f)) for f in (0.1, 0.25, 0.5, 1.0)}
MARK_LOG = {}
model.train()
for ep in range(args.epochs):
    items = [encode_qa(r["prompt"], r["answer"]) for r in train]
    items += [encode_plain(t) for t in random.sample(plain_tr, min(args.plain, len(plain_tr)))]
    random.shuffle(items)
    for b in range(0, len(items), args.bs):
        x, y = batchify(items[b:b + args.bs])
        for g in opt.param_groups:
            g["lr"] = sched(step)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step in MARKS:
            v = val_loss(test_items); MARK_LOG[f'{round(100 * step / total)}%'] = round(v, 4); model.train()
            print(f'  [途中経過 {round(100 * step / total)}%] step {step} カードQA val_loss {v:.4f}', flush=True)
        if step % 100 == 0:
            print(f"  step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    print(f"epoch {ep + 1}/{args.epochs}  val_loss カードQA: {val_loss(test_items):.4f} | 通常会話: {val_loss(plain_val_items):.4f}", flush=True)
os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)
torch.save({"model": model.state_dict(), "step": step}, args.ckpt_out)
print("saved", args.ckpt_out, flush=True)

# ================= 生成して評価 =================
IDK = "I do not know."
FUNC = set("a an the is are it its they their he she of in on and or to by with can cannot not do does yes no i you we this that "
           "there so as at for from also very about than then".split())
NUMW = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def cw(text):
    out = set()
    for w in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower()):
        if w in FUNC:
            continue
        if w.endswith("ies"):
            w = w[:-3] + "y"
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


@torch.no_grad()
def generate(prompt, max_new=60):
    model.eval()
    head = [BOS, USER] + tok.encode(prompt).ids + [BOT]
    ids = list(head)
    for _ in range(max_new):
        x = torch.tensor([ids[-BLOCK:]], device=dev)
        logits, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        ids.append(nxt)
    return tok.decode(ids[len(head):]).strip()


def evaluate(name, test):
    gens = [generate(r["prompt"]) for r in test]
    res = {}
    by = {}
    for r, g in zip(test, gens):
        ref, is_idk = r["answer"], r["idk"]
        pidk = g.strip() == IDK
        allowed = cw(r["prompt"]) | {str(i) for i in range(11)} | set(NUMW)
        extra = cw(g) - allowed
        refw = cw(ref)
        d = by.setdefault(r["type"], {"n": 0, "core": 0.0, "exact": 0, "idk_hit": 0, "ungr": 0, "pol": 0, "pol_n": 0, "name": 0, "name_n": 0})
        d["n"] += 1
        d["exact"] += (g == ref)
        if is_idk:
            d["idk_hit"] += pidk
        else:
            d["core"] += len(refw & cw(g)) / max(len(refw), 1)
            d["ungr"] += bool(extra) and not pidk
            if ref.lower().startswith(("yes,", "no,")):
                d["pol_n"] += 1
                d["pol"] += g.lower().startswith(ref.lower()[:3])
            if r["name"] and r["name"] in ref:
                d["name_n"] += 1
                d["name"] += (r["name"] in g)
    n_all = len(test)
    n_idk = sum(r["idk"] for r in test)
    pred_idk = sum(g.strip() == IDK for g in gens)
    tp = sum(g.strip() == IDK for r, g in zip(test, gens) if r["idk"])
    ans = [(r, g) for r, g in zip(test, gens) if not r["idk"]]
    core = sum(len(cw(r["answer"]) & cw(g)) / max(len(cw(r["answer"])), 1) for r, g in ans) / max(len(ans), 1)
    name_n = sum(1 for r, g in ans if r["name"] and r["name"] in r["answer"])
    name_ok = sum(1 for r, g in ans if r["name"] and r["name"] in r["answer"] and r["name"] in g)
    ungr = sum(1 for r, g in ans if (cw(g) - (cw(r["prompt"]) | {str(i) for i in range(11)} | set(NUMW))) and g.strip() != IDK)
    print(f"\n===== 評価: {name} ({n_all}件, 貪欲生成) =====")
    print(f"答えられる問い {len(ans)}件: 答えの核の再現率 {core:.1%} | 完全一致 {sum(g == r['answer'] for r, g in ans) / max(len(ans), 1):.1%} | カードにない語を含む率 {ungr / max(len(ans), 1):.1%}"
          f" | 名前を正しく書いた率 {name_ok}/{name_n}")
    print(f"IDK(答えられない問い {n_idk}件): 再現率 {tp / max(n_idk, 1):.1%} ({tp}/{n_idk}) | 適合率 {tp / max(pred_idk, 1):.1%} ({tp}/{pred_idk}) | 作り話で答えた率 {(n_idk - tp) / max(n_idk, 1):.1%}")
    for ty, d in sorted(by.items()):
        if ty == "unk":
            continue
        pol = f" | Yes/No一致 {d['pol']}/{d['pol_n']}" if d["pol_n"] else ""
        print(f"  型 {ty:<4} {d['n']:>4}件: 核の再現率 {d['core'] / max(d['n'], 1):.1%} | 完全一致 {d['exact'] / max(d['n'], 1):.1%}{pol}")
    print("--- 生成例(外れ8 + 一致3) ---")
    bad = [(r, g) for r, g in zip(test, gens) if g != r["answer"]]
    ok = [(r, g) for r, g in zip(test, gens) if g == r["answer"]]
    for r, g in bad[:8] + ok[:3]:
        print(f"[{r['type']}] {r['q']}\n    Card: {r['card'][:160]}\n    ref: {r['answer']}\n    gen: {g}")
    return {"core": round(core, 4), "idk_recall": round(tp / max(n_idk, 1), 4), "fabricated": round((n_idk - tp) / max(n_idk, 1), 4),
            "name": f"{name_ok}/{name_n}", "ungrounded": round(ungr / max(len(ans), 1), 4)}


RESULT = {name: evaluate(name, test) for name, test in tests.items()}
if args.log:
    json.dump({"args": vars(args), "marks": MARK_LOG, "results": RESULT, "train_n": len(train), "steps": step}, open(args.log, "w"), indent=1)
print("\nRESULT", json.dumps(RESULT, ensure_ascii=False))
