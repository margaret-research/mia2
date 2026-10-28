# -*- coding: utf-8 -*-
"""T-026 生成した会話(raw_f*.jsonl)を機械的に直して、会話SFT用に整形する

直す: 役割ラベル(USER:/BOT:)の除去 / 発言末尾の「. Mia」除去 / Bot発言の句読点補い(質問なら ?、それ以外は .)
除外: multiで2ターン / 1語だけの発言 / 同じ発言の繰り返し・連続 / ASCII外の文字・markdown記号 / 256トークン超
出力: data/chat_qa/chat_clean_{train,val}.jsonl と、トークン化した data/chat_qa/sft_{train,val}.npz
  ids: uint16 全会話をつなげたID列 ([BOS] <user> .. <bot> .. <eos> ...)、w: uint8 損失の重み(Botの発言と<eos>だけ1)、off: 各会話の開始位置(長さn+1)
使い方: python build_chat_sft.py
"""
import glob
import json
import os
import random
import re
import collections

import numpy as np
from eval_extra_items import EXTRA
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "chat_qa")
BLOCK, N_VAL = 256, 3000
QW = ("do", "does", "did", "is", "are", "was", "were", "can", "could", "will", "would", "should", "have", "has",
      "what", "why", "how", "who", "where", "when", "which", "whose")

tok = Tokenizer.from_file(os.environ.get("MIA_TOK", os.path.join(HERE, "data", "ts", "tokenizer_ext.json")))
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
LABEL = re.compile(r"^\s*(?:USER|BOT|User|Bot|user|bot)\s*:\s*")


def fix_end(t):
    t = t.strip()
    if t and t[-1] not in ".!?":
        last = re.split(r"(?<=[.!?])\s+", t)[-1]
        t += "?" if last.split()[0].lower().strip(",") in QW else "."
    return t


def clean(r):
    T = [LABEL.sub("", t).strip() for t in r["turns"]]
    T = [re.sub(r"(?<=[.!?])\s+Mia$", "", t) for t in T]
    T[1::2] = [fix_end(t) for t in T[1::2]]
    return T


EXTRA_USER = {re.sub(r"[^a-z0-9 ]", "", t.lower()).strip() for _, turns, _, _ in EXTRA for t in turns}


def reject(r, T):
    if any(re.sub(r"[^a-z0-9 ]", "", t.lower()).strip() in EXTRA_USER for t in T[0::2]):
        return "eval-extra-overlap"
    if r["cat"] == "multi" and len(T) < 4:
        return "multi-short"
    if any(len(t.split()) < 2 for t in (T[1::2] if r["cat"] == "greet" else T)):   # greetのユーザー発言は "Hi Mia" 等の短さを許す
        return "one-word"
    if len(set(T)) < len(T) or any(a.lower() == b.lower() for a, b in zip(T, T[1:])):
        return "repeat"
    if any(re.search(r"[^\x00-\x7f]|[*#_`\n]|^\s*[-•]\s", t) for t in T):
        return "symbol"
    return None


def encode(T):
    ids, w = [BOS], [0]
    for i, t in enumerate(T):
        e = tok.encode(t).ids
        if i % 2 == 0:
            ids += [USER] + e
            w += [0] * (len(e) + 1)
        else:
            ids += [BOT] + e + [EOS]
            w += [0] + [1] * (len(e) + 1)
    return ids, w


def main():
    rows = []
    # 追加生成(gen_chat_qa_extra.py): raw_x_recall.jsonl / raw_x_greet.jsonl (試作 *_pilot は含めない)
    for p in sorted(glob.glob(os.path.join(OUT, "raw_f[0-9]*.jsonl"))) + sorted(glob.glob(os.path.join(OUT, "raw_x_*[a-z].jsonl"))):
        if p.endswith("_pilot.jsonl"):
            continue
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    why, keep, changed = collections.Counter(), [], 0
    for r in rows:
        T = clean(r)
        changed += T != r["turns"]
        w = reject(r, T)
        if w:
            why[w] += 1
            continue
        ids, wt = encode(T)
        if len(ids) > BLOCK:
            why["too-long"] += 1
            continue
        keep.append({"cat": r["cat"], "turns": T, "ids": ids, "w": wt})
    random.Random(0).shuffle(keep)
    parts = {"val": keep[:N_VAL], "train": keep[N_VAL:]}
    for name, items in parts.items():
        with open(os.path.join(OUT, f"chat_clean_{name}.jsonl"), "w", encoding="utf-8") as fo:
            for r in items:
                fo.write(json.dumps({"cat": r["cat"], "turns": r["turns"]}, ensure_ascii=False) + "\n")
        off = np.cumsum([0] + [len(r["ids"]) for r in items]).astype(np.int64)
        np.savez(os.path.join(OUT, f"sft_{name}.npz"), ids=np.concatenate([r["ids"] for r in items]).astype(np.uint16),
                 w=np.concatenate([r["w"] for r in items]).astype(np.uint8), off=off)
        print(f"{name}: {len(items)} 会話 / {int(off[-1])} トークン (損失対象 {sum(sum(r['w']) for r in items)})")
    print(f"入力 {len(rows)} / 修正した会話 {changed} / 除外 {sum(why.values())} {dict(why)} / 採用 {len(keep)}")


if __name__ == "__main__":
    main()
