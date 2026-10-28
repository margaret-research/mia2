# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — 読解モデルのckptを、不一致IDK評価と「答えられる問いへの誤IDK率」で測る(貪欲生成)
使い方: python eval_idk.py ckpt/xxx.pt"""
import json, sys, warnings
import torch
warnings.filterwarnings("ignore")
from router import load_tok
from model import GPT, Config
dev = "cuda"
tok = load_tok()
m = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=256)).to(dev)
m.load_state_dict(torch.load(sys.argv[1], map_location=dev)["model"]); m.eval()
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
@torch.no_grad()
def gen(p):
    ids = [BOS, USER] + tok.encode(p).ids + [BOT]; n = len(ids)
    for _ in range(60):
        nx = int(m(torch.tensor([ids[-256:]], device=dev))[0][0, -1].argmax())
        if nx == EOS: break
        ids.append(nx)
    return tok.decode(ids[n:]).strip()
rd = lambda f: [json.loads(l) for l in open("data/reading_qa/" + f, encoding="utf-8")]
mis = rd("m_test_mismatch.jsonl")
g = [gen(r["prompt"]) for r in mis]
print(f"不一致IDK(300件): IDK率 {sum(x == 'I do not know.' for x in g) / len(g):.1%}")
for r, x in list(zip(mis, g))[:300]:
    if x != "I do not know.": print("  外れ:", r["q"], "|", r["card"][:70], "=>", x[:70])
for f in ("s10k_test_newsubj.jsonl", "s10k_test_seen.jsonl"):
    t = [r for r in rd(f) if not r["idk"]][:400]
    gg = [gen(r["prompt"]) for r in t]
    print(f"{f}: 答えられる{len(t)}件のうちIDKと答えた(誤IDK)率 {sum(x == 'I do not know.' for x in gg) / len(t):.1%}")
