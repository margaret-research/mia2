# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — 答える側の確認: 電卓の結果を読解の形で渡したとき、今のMiaが正しい数で答えるか (学習なし)
使い方: MIA_TOK=data/ts_v4/tokenizer_ext.json MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python eval_router_answer.py ckpt/mia_current_read.pt [件数]"""
import json, os, re, sys, random, collections
import torch
from tokenizers import Tokenizer
from model import GPT, Config
tok = Tokenizer.from_file(os.environ["MIA_TOK"])
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
m = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=1024)).cuda()
m.load_state_dict(torch.load(sys.argv[1])["model"]); m.eval()
@torch.no_grad()
def gen(ids):
    n0 = len(ids)
    for _ in range(40):
        lg, _ = m(torch.tensor([ids], device="cuda")); nx = int(lg[0, -1].argmax())
        if nx in (EOS, USER): break
        ids.append(nx)
    return tok.decode(ids[n0:]).strip()
rows = [json.loads(l) for l in open("data/router/test.jsonl") if '"CALC' in l and '"hist": []' in l]
random.seed(0); random.shuffle(rows); rows = rows[:int(sys.argv[2]) if len(sys.argv) > 2 else 200]
FORMS = {"A": "Title: Calculation. {e} = {r}. Question: {q}", "B": "Facts: {e} = {r}. Question: {q}"}
for name, f in FORMS.items():
    hit, bad = collections.Counter(), []
    for r in rows:
        e = r["target"][5:]
        a = gen([BOS, USER] + tok.encode(f.format(e=e, r=r["result"], q=r["user"])).ids + [BOT])
        nums = re.findall(r"\d[\d,]*", a)
        ok = r["result"].replace(",", "") in [x.replace(",", "") for x in nums]
        hit[r["type"]] += ok; hit["_n_" + r["type"]] += 1
        if not ok and len(bad) < 6: bad.append((r["user"], e, r["result"], a))
    n = sum(v for k, v in hit.items() if k.startswith("_n_")); h = sum(v for k, v in hit.items() if not k.startswith("_n_"))
    print(f"形式{name}: {h}/{n} = {h / n:.2f}")
    for b in bad: print("  ", b)
