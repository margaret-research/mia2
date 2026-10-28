# -*- coding: utf-8 -*-
"""MIA2: SimpleStories(MIT) を、prep_tinystories.py と同じ正規化で自作トークナイザーでトークン化し uint16 の連続列にする

正規化: 曲がった引用符・ダッシュ・省略記号をASCIIへ、改行を含む空白を1つに畳む(TinyStoriesと同じ)。
        正規化後も非ASCIIを含む話、20語未満の話、完全重複は除外。各話を [BOS] ... [EOS] で区切って連結。
        検証用は公式の test 分割(21,371本)。train と同じ文は train 側から除く(漏れ防止)。
入力は信頼できない外部データ: スクリプトはここ(code/)、データは別ディレクトリ。-I で実行する。
使い方: python -I prep_simplestories.py <入力parquetのディレクトリ> <tokenizer_ext.json> <出力ディレクトリ>
"""
import glob
import re
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

src, tok_path, out = sys.argv[1:4]
CHUNK = 20000
tok = Tokenizer.from_file(tok_path)
BOS, EOS = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
assert tok.get_vocab_size() < 65536
tr = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...", "\xa0": " ", "​": "", " ": " ", "´": "'"})


def clean(s):
    return re.sub(r"\s+", " ", s.translate(tr)).strip()


def load(files, seen, label):
    ss, nonascii, short, dup = [], 0, 0, 0
    for fn in files:
        for s in pq.read_table(fn, columns=["story"])["story"].to_pylist():
            s = clean(s or "")
            if len(s.split()) < 20:
                short += 1
            elif not s.isascii():
                nonascii += 1
            elif s in seen:
                dup += 1
            else:
                seen.add(s)
                ss.append(s)
        print(f"  {fn.split('/')[-1]} 読み込み後 {len(ss)} 本", flush=True)
    print(f"{label}: 採用 {len(ss)} 本 / 非ASCII除外 {nonascii} / 短すぎ {short} / 重複 {dup}", flush=True)
    return ss


def write(name, ss):
    total = 0
    with open(f"{out}/{name}.bin", "wb") as fo:
        for i in range(0, len(ss), CHUNK):
            a = np.concatenate([np.array([BOS] + e.ids + [EOS], dtype=np.uint16) for e in tok.encode_batch(ss[i:i + CHUNK])])
            a.tofile(fo)
            total += len(a)
            if (i // CHUNK) % 10 == 0:
                print(f"  {name} {min(i + CHUNK, len(ss))}/{len(ss)} 本  累計 {total / 1e6:.1f}Mトークン", flush=True)
    print(f"{name}: {len(ss)} 本 {total / 1e6:.1f}Mトークン", flush=True)


seen = set()
train = load(sorted(glob.glob(f"{src}/train-*.parquet")), seen, "train")
val = load(sorted(glob.glob(f"{src}/test-*.parquet")), seen, "val")   # train にある文は val から除く
write("val", val)
write("train", train)
