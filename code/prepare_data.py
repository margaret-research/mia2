# Claude Code により作成 (2026-10-04) — T-023: コーパスをトークン化して uint16 の連続ストリームに保存
import json, numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file("data/tokenizer_5k.json")
BOS, EOS = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
ids, buf = [], []


def flush():
    global buf
    for e in tok.encode_batch(buf):
        ids.append(np.array([BOS] + e.ids + [EOS], dtype=np.uint16))
    buf = []


with open("data/corpus_sentences.jsonl", encoding="utf-8") as f:
    for line in f:
        buf.append(json.loads(line)["sentence"])
        if len(buf) >= 50000:
            flush()
if buf:
    flush()
arr = np.concatenate(ids)
rng = np.random.default_rng(0)
# 文単位でシャッフルしてから val 分割 (テンプレート順の偏り対策)
lens = np.array([len(a) for a in ids])
order = rng.permutation(len(ids))
n_val = 20000
val = np.concatenate([ids[i] for i in order[:n_val]])
train = np.concatenate([ids[i] for i in order[n_val:]])
train.tofile("data/train.bin")
val.tofile("data/val.bin")
print("sentences", len(ids), "train tokens", len(train), "val tokens", len(val))
