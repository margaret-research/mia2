# -*- coding: utf-8 -*-
"""T-026 会話SFT: 事前学習(ts_pre)から、整形済みの会話(build_chat_sft.py の sft_*.npz)で学習する

形式は [BOS] <user> 発言 <bot> 返事 <eos> ... で、損失は Bot の返事と <eos> だけ。長さの近い会話をまとめてパディングを減らす。
使い方: python sft_chat.py --ckpt-in ckpt/ts_pre_0.66ep.pt --out ckpt/chat_sft.pt --epochs 3
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, Config

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-in", default="ckpt/ts_pre_0.66ep.pt")
ap.add_argument("--out", default="ckpt/chat_sft.pt")
ap.add_argument("--data", default="data/chat_qa")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--freeze-emb", action="store_true", help="単語埋め込みを固定")
ap.add_argument("--block", type=int, default=256, help="文脈長(基礎モデルの位置埋め込みの数と同じにする)")
ap.add_argument("--max-tok", type=int, default=0, help="1バッチの最大トークン数(行数×最長)。0なら制限なし。長い会話でメモリを守る")
ap.add_argument("--val-split", type=int, default=0, help="検証の先頭N会話(短い会話)と残り(長い会話)を別々にも報告する")
ap.add_argument("--eval-only", action="store_true", help="--ckpt-in を読んで検証だけして終わる")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)
dev = "cuda"
tok = Tokenizer.from_file(os.environ.get("MIA_TOK", "data/ts/tokenizer_ext.json"))
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))


def load(name):
    z = np.load(os.path.join(a.data, f"sft_{name}.npz"))
    ids, w, off = z["ids"], z["w"], z["off"]
    return [(ids[off[i]:off[i + 1]], w[off[i]:off[i + 1]]) for i in range(len(off) - 1)]


train, val = load("train"), load("val")
if a.smoke:
    train, val, a.epochs = train[:640], val[:128], 1


def batchify(items):
    L = max(len(i) for i, _ in items)
    x = torch.zeros(len(items), L, dtype=torch.long)
    y = torch.full((len(items), L), -1, dtype=torch.long)
    for r, (ids, w) in enumerate(items):
        n = len(ids)
        x[r, :n] = torch.from_numpy(ids.astype(np.int64))
        t = torch.from_numpy(ids[1:].astype(np.int64))
        m = torch.from_numpy(w[1:].astype(bool))
        y[r, :n - 1] = torch.where(m, t, torch.tensor(-1))
    return x.to(dev), y.to(dev)


def batches(items):
    """シャッフル後、64バッチ分ずつ長さ順に並べてからバッチにし、バッチ順をもう一度混ぜる"""
    items = items[:]
    random.shuffle(items)
    chunk = a.bs * 64
    out = []
    for c in range(0, len(items), chunk):
        part = sorted(items[c:c + chunk], key=lambda t: len(t[0]))
        cur = []
        for it in part:                     # 長さ順なので、いま入れる会話が一番長い
            if cur and (len(cur) >= a.bs or (a.max_tok and (len(cur) + 1) * len(it[0]) > a.max_tok)):
                out.append(cur); cur = []
            cur.append(it)
        if cur:
            out.append(cur)
    random.shuffle(out)
    return out


model = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=a.block)).to(dev)
model.load_state_dict(torch.load(a.ckpt_in, map_location=dev)["model"])


@torch.no_grad()
def val_loss(items=None):
    model.eval()
    items = val if items is None else items
    tot, n = 0.0, 0
    for b in range(0, len(items), 64):
        x, y = batchify(items[b:b + 64])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="sum").item()
        n += (y != -1).sum().item()
    model.train()
    return tot / n


def val_report():
    r = f"val_loss {val_loss():.4f}"
    if a.val_split:
        r += f" (短い {val_loss(val[:a.val_split]):.4f} / 長い {val_loss(val[a.val_split:]):.4f})"
    return r


@torch.no_grad()
def reply(turns, n=60, temp=0.7):
    model.eval()
    ids = [BOS]
    for i, t in enumerate(turns):
        ids += ([USER] if i % 2 == 0 else [BOT]) + tok.encode(t).ids
    ids += [BOT]
    n0 = len(ids)
    for _ in range(n):
        lg, _ = model(torch.tensor([ids[-a.block:]], device=dev))
        nxt = int(torch.multinomial(torch.softmax(lg[0, -1] / temp, -1), 1))
        if nxt in (EOS, USER):
            break
        ids.append(nxt)
    model.train()
    return tok.decode(ids[n0:]).strip()


SAMPLES = [["Hello!"], ["What is your name?"], ["Why is the sky blue?"], ["I am tired."], ["Tell me about dogs."],
           ["My cat is named Coco.", "That is a nice name. Is Coco big or small?", "She is small. What is her name?"]]
if a.freeze_emb:
    model.wte.weight.requires_grad_(False)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
spe = len(batches(train))
total = spe * a.epochs
sched = lambda s: a.lr * min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / total)))
if a.eval_only:
    print(val_report(), flush=True)
    raise SystemExit
print(f"train {len(train)} 会話 / val {len(val)} / steps {total} ({spe}/epoch) / 開始前 {val_report()}", flush=True)
step, t0 = 0, time.time()
model.train()
for ep in range(a.epochs):
    for items in batches(train):
        x, y = batchify(items)
        for g in opt.param_groups:
            g["lr"] = sched(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            print(f"  step {step}/{total} loss {loss.item():.4f} lr {sched(step):.2e} {time.time() - t0:.0f}s", flush=True)
    print(f"== epoch {ep + 1}/{a.epochs} {val_report()} ({time.time() - t0:.0f}s)", flush=True)
    for s in SAMPLES:
        print("   ", s[-1], "->", reply(s), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step, "epoch": ep + 1}, a.out.replace(".pt", f"_e{ep + 1}.pt"))
torch.save({"model": model.state_dict(), "step": step}, a.out)
print("saved", a.out, flush=True)
