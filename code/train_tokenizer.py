# Claude Code により作成 (2026-10-04) — T-023: BPE(vocab 5000) 学習と実トークン数の計測
import json, random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

CORPUS = "data/corpus_sentences.jsonl"
OUT = "data/tokenizer_5k.json"
VOCAB_SIZE = 5000
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def sentences():
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["sentence"]


tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(sentences(), trainer=trainer)
tok.save(OUT)
print("vocab size:", tok.get_vocab_size())

# 実トークン数の計測 (ランダム20万文でサンプリング→全文数へ外挿)
random.seed(0)
sample = []
n_total = 0
for s in sentences():
    n_total += 1
    if len(sample) < 200000:
        sample.append(s)
    else:
        j = random.randint(0, n_total - 1)
        if j < 200000:
            sample[j] = s
lens = [len(e.ids) for e in tok.encode_batch(sample)]
avg = sum(lens) / len(lens)
print(f"sentences: {n_total}, avg tokens/sentence (+bos/eos 2): {avg:.2f} (+2)")
print(f"total tokens est: {(avg + 2) * n_total / 1e6:.1f}M")
print(f"unique sentences in sample: {len(set(sample))}/{len(sample)}")
print("example:", ascii(tok.encode(sample[0]).tokens))
