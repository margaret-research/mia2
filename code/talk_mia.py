# -*- coding: utf-8 -*-
"""Mia と会話してみる(会話SFT済みのモデルを読み込む)
使い方: MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python talk_mia.py [ckpt=ckpt/chat_sft_mix_B1024.pt] [温度=0.7]
入力: 英語で話しかける / 空行・quit で終了 / /reset で会話をはじめから
"""
import sys

import torch
from tokenizers import Tokenizer

from model import GPT, Config

ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt/chat_sft_mix_B1024.pt"
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
BLOCK, MAX_NEW, dev = 1024, 120, "cuda"
tok = Tokenizer.from_file(__import__("os").environ.get("MIA_TOK", "data/ts_v3/tokenizer_ext.json"))
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
model = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=BLOCK)).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
model.eval()

ids = [BOS]
print(f"Mia ({ckpt}, 温度 {temp})。英語で話しかけてください。")
while True:
    try:
        text = input("you> ").strip()
    except EOFError:
        break
    if text in ("", "quit"):
        break
    if text == "/reset":
        ids = [BOS]; print("(はじめから)"); continue
    ids += [USER] + tok.encode(text).ids + [BOT]
    new = []
    with torch.no_grad():
        for _ in range(MAX_NEW):
            lg, _ = model(torch.tensor([ids[-BLOCK:]], device=dev))
            nxt = int(torch.multinomial(torch.softmax(lg[0, -1] / temp, -1), 1))
            if nxt in (EOS, USER):
                break
            ids.append(nxt); new.append(nxt)
    ids.append(EOS)
    print("Mia>", tok.decode(new).strip())
    if len(ids) > BLOCK:
        print(f"(会話が {len(ids)} トークンになり、古い部分は読まれません)")
