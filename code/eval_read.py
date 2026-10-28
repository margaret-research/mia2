# -*- coding: utf-8 -*-
"""読解SFTの評価: 学習に使わなかった50項目(data/read_mix/val_entries.json)の全質問を、1問ずつ(項目+質問1つ)貪欲生成で答えさせる
unk は答えが "I do not know." と一致するか、他の型は模範解答の内容語(3文字以上)の6割以上が出ているか。
使い方: MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python eval_read.py ckpt/read_sft_B1024.pt"""
import json, re, sys, collections
import torch
from tokenizers import Tokenizer
from model import GPT, Config
import os
tok = Tokenizer.from_file(os.environ.get("MIA_TOK", "data/ts_v3/tokenizer_ext.json"))
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
m = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=1024)).cuda()
m.load_state_dict(torch.load(sys.argv[1])["model"]); m.eval()
STOP = set("the and are was were that this with for from have has does did not what how who which where why".split())
def words(s): return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) >= 3 and w not in STOP}
@torch.no_grad()
def ask(e, q):
    ids = [BOS, USER] + tok.encode(f"Title: {e['title']}. {e['entry']} Question: {q}").ids + [BOT]
    n0 = len(ids)
    for _ in range(60):
        lg, _ = m(torch.tensor([ids], device="cuda")); nx = int(lg[0, -1].argmax())
        if nx in (EOS, USER): break
        ids.append(nx)
    return tok.decode(ids[n0:]).strip()
E_ = json.load(open("benchmark/read_val_entries.json"))
hit, tot, bad = collections.Counter(), collections.Counter(), []
for e in E_:
    for q in e["qs"]:
        a = ask(e, q["q"]); t = q["type"]; tot[t] += 1
        if t == "unk": ok = a == "I do not know."
        else:
            g = words(q["a"]); ok = a != "I do not know." and len(g & words(a)) >= 0.6 * len(g)
            if t in ("yes", "no"): ok = ok and a.lower().startswith(q["a"].lower().split(",")[0])
        hit[t] += ok
        if not ok and len(bad) < 14: bad.append((t, q["q"], q["a"], a))
for t in tot: print(f"{t}: {hit[t]}/{tot[t]} = {hit[t] / tot[t]:.2f}")
print(f"全体: {sum(hit.values())}/{sum(tot.values())} = {sum(hit.values()) / sum(tot.values()):.2f}")
for b in bad: print(b)
