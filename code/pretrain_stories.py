# -*- coding: utf-8 -*-
"""MIA2: 基礎学習(事前学習) SimpleStories + TinyStories をランダムに混ぜ、「お話ごと」に学習する (文脈長1,024)

 ・2つのデータの全部の話を1つにして、毎エポックごとにシャッフル(割合は話の数のまま)。
 ・1つの話が1行(<bos> ... <eos>)。ほかの話をまたいで見ない。位置は0から。1,024を超える話は先頭1,024で切る。
 ・同じ長さの話をまとめて(約16,000話のプールを長さ順に並べて)バッチにし、足りない分は <pad>(損失なし)で埋める。
   バッチは「1ステップあたり埋め込み込みで約32,768トークン」(pretrain_ts.py の bs128×256 と同じ大きさ)。バッチの順番はシャッフル。
 ・そのほか(モデル、bf16、AdamW lr1e-3 wd0.1、cosine、warmup 500、Qwen埋め込み固定)は pretrain_ts.py と同じ。
使い方: MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python pretrain_stories.py --out ckpt/pre_stories_B1024.pt --epochs 3 --emb-init ckpt/emb_init_v3_512.pt --freeze-emb
        python pretrain_stories.py --smoke      (配管確認だけ)
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
ap.add_argument("--train", nargs="+", default=["data/ts_v3/train.bin", "data/simplestories/train.bin"])
ap.add_argument("--val", nargs="+", default=["data/ts_v3/val.bin", "data/simplestories/val.bin"])
ap.add_argument("--val-names", nargs="+", default=["tinystories", "simplestories"])
ap.add_argument("--tok", default="data/ts_v3/tokenizer_ext.json")
ap.add_argument("--out", default="ckpt/pre_stories_B1024.pt")
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--block", type=int, default=1024)
ap.add_argument("--step-tokens", type=int, default=32768, help="1ステップの埋め込み込みトークン数")
ap.add_argument("--pool", type=int, default=16384)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--eval-stories", type=int, default=3000)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--emb-init", default=None)
ap.add_argument("--freeze-emb", action="store_true")
ap.add_argument("--shift-prob", type=float, default=0.5, help="話の開始位置をずらす確率(ずらす場合は 0〜(block−話の長さ) の一様乱数)")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--bench", type=int, default=0, help="速さの測定: このステップ数だけ走らせる")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"
tok = Tokenizer.from_file(a.tok)
BOS, PAD = tok.token_to_id("<bos>"), tok.token_to_id("<pad>")
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


def index(paths):
    """各bin(<bos>で始まり<eos>で終わる話の連結)から、(ファイル番号, 開始, 長さ)の配列を作る"""
    mms = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
    f, s, l = [], [], []
    for k, m in enumerate(mms):
        st = np.flatnonzero(m == BOS)
        ln = np.diff(np.append(st, len(m)))
        f.append(np.full(len(st), k, np.int8)); s.append(st.astype(np.int64)); l.append(np.minimum(ln, a.block + 1).astype(np.int32))
    return mms, np.concatenate(f), np.concatenate(s), np.concatenate(l)


def plan(lens, order, step_tokens, keep_partial=False):
    """order(話の並び)から、バッチ(話番号の配列)のリストを作る。プールごとに長さ順にまとめる
    学習用は形(行数×長さ)の種類を増やさないため、プール末尾の端数バッチは捨てる(1エポックの1%未満、毎エポック別の話が捨てられる)"""
    batches = []
    for i in range(0, len(order), a.pool):
        pool = order[i:i + a.pool]
        pool = pool[np.argsort(lens[pool], kind="stable")]
        j = 0
        while j < len(pool):
            # 長さ順なので、候補の最後の話が一番長い。埋め込み込みで step_tokens を超えない行数にする
            n = 1
            while j + n < len(pool) and (n + 1) * (-(-(lens[pool[j + n]] - 1) // 64) * 64) <= step_tokens:
                n += 1
            T = -(-(lens[pool[j + n - 1]] - 1) // 64) * 64
            if keep_partial or n == step_tokens // T:
                batches.append(pool[j:j + n])
            j += n
    return batches


def make(mms, F, S, L, ids, mode="train", rng=None):
    """mode: train=確率 shift_prob で開始位置をずらす / start=位置0 / end=1,024の末尾にそろえる(後ろの位置の確認用)"""
    T = int(-(-(L[ids].max() - 1) // 64) * 64)
    x = np.full((len(ids), T), PAD, np.int64)
    y = np.full((len(ids), T), -1, np.int64)
    for r, i in enumerate(ids):
        seq = mms[F[i]][S[i]:S[i] + L[i]].astype(np.int64)
        x[r, :len(seq) - 1] = seq[:-1]
        y[r, :len(seq) - 1] = seq[1:]
    n_in = (L[ids] - 1).astype(np.int64)
    room = a.block - n_in                      # 話を置いたあと、後ろに残る位置の数(0以上)
    if mode == "end":
        off = room
    elif mode == "train" and a.shift_prob > 0:
        off = np.where(rng.random(len(ids)) < a.shift_prob, (rng.random(len(ids)) * (room + 1)).astype(np.int64), 0)
    else:
        off = np.zeros(len(ids), np.int64)
    pos = np.minimum(off[:, None] + np.arange(T)[None, :], a.block - 1)   # <pad> の位置は範囲内に収めるだけ(損失なし)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), torch.from_numpy(pos).to(dev)


mms, F, S, L = index(a.train)
N = len(L)
print(f"params {n_params / 1e6:.2f}M  vocab {cfg.vocab_size}  block {a.block}  話数 {N:,} (総 {int(L.sum()) / 1e6:.0f}Mトークン)", flush=True)
vals = {}
for name, p in zip(a.val_names, a.val):
    vm, vF, vS, vL = index([p])
    pick = np.random.default_rng(123).permutation(len(vL))[:a.eval_stories]
    vals[name] = (vm, vF, vS, vL, plan(vL, np.sort(pick), a.step_tokens, keep_partial=True))
    print(f"val {name}: {len(vL):,} 話のうち {len(pick)} 話で測る ({int(vL[pick].sum()) / 1e6:.2f}Mトークン)", flush=True)

epoch_plans = {}


def epoch_plan(e):
    if e not in epoch_plans:
        epoch_plans.clear()
        order = np.random.default_rng(a.seed * 1000 + e).permutation(N)
        b = plan(L, order, a.step_tokens)
        epoch_plans[e] = [b[k] for k in np.random.default_rng(a.seed * 1000 + 500 + e).permutation(len(b))]
    return epoch_plans[e]


per_epoch = len(epoch_plan(0))
steps = 30 if a.smoke else (a.bench or int(per_epoch * a.epochs))
print(f"1エポック {per_epoch:,} ステップ → 合計 {steps:,} ステップ ({a.epochs}エポック)", flush=True)

decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
step0 = 0
if a.resume and os.path.exists(a.out):
    ck = torch.load(a.out, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    print(f"resume from step {step0}", flush=True)
torch._dynamo.config.cache_size_limit = 64
cmodel = torch.compile(model, dynamic=False)


def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    p = (s - a.warmup) / max(1, steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for name, mode, (vm, vF, vS, vL, bl) in [(k, m, v) for k, v in vals.items() for m in ("start", "end")]:
        tot = n = 0.0
        for ids in bl:
            x, y, pos = make(vm, vF, vS, vL, ids, mode)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, None, pos)
            for r in range(0, x.shape[0], 16):          # 行を小分けにして float 化(メモリ節約)
                lg, yy = logits[r:r + 16], y[r:r + 16]
                tot += torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)), yy.reshape(-1), ignore_index=-1, reduction="sum").item()
                n += (yy != -1).sum().item()
            del logits
        out[f"{name}[{mode}]"] = tot / n
    model.train()
    return out


def save(step):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "opt": opt.state_dict()}, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)


shift_rng = np.random.default_rng(a.seed * 7 + 99 + step0)   # 再開時は別の乱数列になる(ずらし方だけ。厳密な再現はしない)
model.train()
t0, seen_tok = time.time(), 0
for step in range(step0, steps):
    e, k = divmod(step, per_epoch)
    ids = epoch_plan(e)[k]
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y, pos = make(mms, F, S, L, ids, "train", shift_rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = cmodel(x, y, pos)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    seen_tok += int((y != -1).sum())
    if step % 100 == 0:
        el = time.time() - t0
        print(f"step {step}/{steps} ep {e} loss {loss.item():.3f} lr {lr_at(step):.2e} {seen_tok / max(el, 1e-9) / 1e3:.0f}k 実トークン/秒 (行{x.shape[0]}×{x.shape[1]})", flush=True)
    if (step + 1) % a.eval_every == 0 or step + 1 == steps:
        v = evaluate()
        print(f"== step {step + 1} val_loss " + " / ".join(f"{k} {x:.4f}" for k, x in v.items()), flush=True)
        save(step + 1)
print("done", flush=True)
