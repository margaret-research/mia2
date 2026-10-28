# -*- coding: utf-8 -*-
"""T-026 基礎力向上: TinyStories(V2-GPT4)を拡張トークナイザーでトークン化し uint16 の連続列にする

正規化: 曲がった引用符・ダッシュ・省略記号をASCIIへ、空白を1つに畳む。正規化後も非ASCIIを含む話(中国語など)、
        短すぎる話、完全重複は除外。シャッフルして末尾 N_VAL 話を検証用に取り分ける。
入力は信頼できない外部データ: スクリプトはここ(code/)、データは別ディレクトリに置き、-I で実行する。
使い方: python -I prep_tinystories.py <入力.txt> <tokenizer_ext.json> <出力ディレクトリ>
"""
import random
import re
import sys

import numpy as np
from tokenizers import Tokenizer

src, tok_path, out = sys.argv[1:4]
N_VAL, CHUNK = 20000, 20000
tok = Tokenizer.from_file(tok_path)
BOS, EOS = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
assert tok.get_vocab_size() < 65536
MAP = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...", "\xa0": " ",
       "​": "", " ": " ", "´": "'"}
tr = str.maketrans(MAP)


def clean(s):
    return re.sub(r"\s+", " ", s.translate(tr)).strip()


stories, seen, drop_nonascii, drop_short, drop_dup = [], set(), 0, 0, 0
with open(src, encoding="utf-8", errors="replace") as f:
    for part in f.read().split("<|endoftext|>"):
        s = clean(part)
        if len(s.split()) < 20:
            drop_short += 1
        elif not s.isascii():
            drop_nonascii += 1
        elif s in seen:
            drop_dup += 1
        else:
            seen.add(s)
            stories.append(s)
del seen
print(f"採用 {len(stories)} 話 / 非ASCII除外 {drop_nonascii} / 短すぎ {drop_short} / 重複 {drop_dup}", flush=True)
random.Random(0).shuffle(stories)
splits = {"val": stories[:N_VAL], "train": stories[N_VAL:]}
for name, ss in splits.items():
    total = 0
    with open(f"{out}/{name}.bin", "wb") as fo:
        for i in range(0, len(ss), CHUNK):
            arrs = [np.array([BOS] + e.ids + [EOS], dtype=np.uint16) for e in tok.encode_batch(ss[i:i + CHUNK])]
            a = np.concatenate(arrs)
            a.tofile(fo)
            total += len(a)
            if (i // CHUNK) % 20 == 0:
                print(f"  {name} {i + CHUNK}/{len(ss)} 話  累計 {total / 1e6:.1f}Mトークン", flush=True)
    print(f"{name}: {len(ss)} 話 {total / 1e6:.1f}Mトークン", flush=True)
