# -*- coding: utf-8 -*-
"""会話モデル単体の評価: eval_chat.py の62問のうち、会話モデルが答える問題(route が chat/any)だけを、新しい会話SFTモデルに直接聞く
(計算・知識カードに回る問題は含めない)。合格判定・崩れ判定は eval_chat.py と同じ。旧結果(eval_results/*.json)があれば同じ問題だけ集計して並べる。
使い方: python eval_chat_only.py --ckpt ckpt/chat_sft.pt --tag chat_sft --base m120k,s100k
"""
import argparse
import json
import os
import re

import torch
from tokenizers import Tokenizer

from eval_chat import ITEMS, broken
from eval_extra_items import EXTRA
from model import GPT, Config

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/chat_sft.pt")
ap.add_argument("--tag", default="chat_sft")
ap.add_argument("--sample", type=int, default=3)
ap.add_argument("--base", default="m120k,s100k")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--extra", action="store_true", help="追加評価(想起型・あいさつ型, eval_extra_items.py)も一緒に聞く。id は 62 以降")
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"
tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
model = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=256)).to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
model.eval()


@torch.no_grad()
def say(hist, u):
    ctx = ([BOS] + hist + [USER] + tok.encode(u).ids + [BOT])[-200:]
    if ctx[0] != BOS:
        ctx = [BOS] + ctx
    gen = model.generate(torch.tensor([ctx], device=dev), 40, temperature=0.7, top_k=40, eos=EOS)[0].tolist()[len(ctx):]
    reply = tok.decode([i for i in gen if i != EOS]).strip()
    return reply, hist + [USER] + tok.encode(u).ids + [BOT] + tok.encode(reply).ids + [EOS]


ALL = list(ITEMS) + (EXTRA if a.extra else [])
ids = [i for i, it in enumerate(ALL) if it[2] in ("chat", "any")]
rows = []
for i in ids:
    cat, turns, route, ok = ALL[i]
    reps = []
    for _ in range(a.sample):
        hist, r = [], ""
        for u in turns:
            r, hist = say(hist, u)
        reps.append(r)
    rows.append(dict(id=i, cat=cat, turns=turns, ok=ok, replies=reps, pass_rate=sum(bool(re.search(ok, x, re.I)) for x in reps) / len(reps),
                     broken_rate=sum(broken(x) for x in reps) / len(reps)))
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
bases = {}
for b in filter(None, a.base.split(",")):
    p = f"eval_results/{b}.json"
    if os.path.exists(p):
        bases[b] = {r["id"]: r for r in json.load(open(p, encoding="utf-8"))}
print(f"=== {a.tag}  ckpt={a.ckpt}  会話モデルの問題 {len(rows)}問 (各{a.sample}回) ===")
print(f"{'カテゴリ':<10}{'問数':>4}{'合格率':>8}{'崩れ':>8}" + "".join(f"{'旧' + b:>10}" for b in bases))
cats = list(dict.fromkeys(r["cat"] for r in rows)) + ["全体"]
for c in cats:
    rs = [r for r in rows if c == "全体" or r["cat"] == c]
    line = f"{c:<10}{len(rs):>4}{mean([r['pass_rate'] for r in rs]):>8.2f}{mean([r['broken_rate'] for r in rs]):>8.2f}"
    for b, d in bases.items():
        line += f"{mean([d[r['id']]['pass_rate'] for r in rs if r['id'] in d]):>10.2f}"
    print(line)
os.makedirs("eval_results", exist_ok=True)
json.dump(rows, open(f"eval_results/{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"-> eval_results/{a.tag}.json")
for r in rows:
    print(f"[{r['cat']}] {' / '.join(r['turns'])}  合格{r['pass_rate']:.2f}\n     {r['replies'][0]}")
