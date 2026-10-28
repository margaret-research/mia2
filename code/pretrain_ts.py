# -*- coding: utf-8 -*-
"""T-026 基礎力向上: 拡張トークナイザー + TinyStories でスクラッチ事前学習 (model.py の GPT をそのまま使う)

train.bin / val.bin (uint16 連続列) からランダムな窓を切って次トークン予測。bf16、cosine、AdamW (lr 1e-3, wd 0.1, β=(0.9,0.95), warmup 500)。
チェックポイントは {"model", "cfg", "step", "opt"} で、後続の学習は torch.load(...)["model"] で読める。--resume で続きから。
使い方: python pretrain_ts.py --data data/ts --out ckpt/ts_pre.pt --tokens 360e6
        python pretrain_ts.py --smoke      (配管確認だけ)
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, Config

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/ts")
ap.add_argument("--out", default="ckpt/ts_pre.pt")
ap.add_argument("--tokens", type=float, default=360e6, help="学習で読むトークン総数 (ちんちら目安は 20×パラメータ)")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--block", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--eval-batches", type=int, default=50)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--emb-init", default=None, help="build_emb_init.py の出力。語トークンの埋め込み初期値をQwen3由来にする")
ap.add_argument("--freeze-emb", action="store_true", help="単語埋め込み(出力の層と共有)を固定して学習する")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = "cuda"
tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer_ext.json"))
cfg = Config(vocab_size=tok.get_vocab_size(), block_size=a.block)
model = GPT(cfg).to(dev)
if a.emb_init:
    _e = torch.load(a.emb_init)
    with torch.no_grad():
        model.wte.weight[_e["ids"]] = _e["vec"].to(model.wte.weight.dtype).to(dev)
    print("埋め込み初期値を借りた語トークン", len(_e["ids"]), flush=True)
if a.freeze_emb:
    model.wte.weight.requires_grad_(False)
    print("単語埋め込みを固定", flush=True)
n_params = sum(p.numel() for p in model.parameters())
tokens_per_step = a.bs * a.block
steps = 30 if a.smoke else int(a.tokens / tokens_per_step)
print(f"params {n_params / 1e6:.2f}M  vocab {cfg.vocab_size}  steps {steps}  tokens/step {tokens_per_step}  "
      f"総トークン {steps * tokens_per_step / 1e6:.0f}M (= {steps * tokens_per_step / n_params:.1f} トークン/パラメータ)", flush=True)

train = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
print(f"train {len(train) / 1e6:.1f}Mトークン / val {len(val) / 1e6:.2f}Mトークン", flush=True)


def batch(data, g=rng):
    ix = g.integers(0, len(data) - a.block - 1, a.bs)
    x = np.stack([data[i:i + a.block] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + a.block] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
step0 = 0
if a.resume and os.path.exists(a.out):
    ck = torch.load(a.out, map_location=dev)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    step0 = ck["step"]
    print(f"resume from step {step0}", flush=True)
cmodel = torch.compile(model)


def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    p = (s - a.warmup) / max(1, steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate():
    model.eval()
    g = np.random.default_rng(123)           # 毎回同じ窓で比べる
    tot = 0.0
    for _ in range(a.eval_batches):
        x, y = batch(val, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = cmodel(x, y)
        tot += loss.item()
    model.train()
    return tot / a.eval_batches


def save(step):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": vars(cfg) if hasattr(cfg, "__dict__") else cfg.__dict__, "step": step,
                "opt": opt.state_dict()}, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)


model.train()
t0 = time.time()
for step in range(step0, steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = batch(train)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = cmodel(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 100 == 0:
        el = time.time() - t0
        print(f"step {step}/{steps} loss {loss.item():.3f} lr {lr_at(step):.2e} {((step - step0 + 1) * tokens_per_step) / max(el, 1e-9) / 1e3:.0f}k tok/s", flush=True)
    if (step + 1) % a.eval_every == 0 or step + 1 == steps:
        print(f"== step {step + 1} val_loss {evaluate():.4f}", flush=True)
        save(step + 1)
print("done", flush=True)
