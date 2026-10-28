# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-07) — readqa_colab.ipynb を作り、colab_readqa_upload/ にそろえる (T-026 読解SFT 1万問の通し確認)"""
import json
import os
import shutil

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)})


md("""# 読解SFT(動物に限らない) 1万問の通し確認 — T-026
Claude Code により作成 (2026-10-07)

**やること**: 会話LLM(`sft_5k_v2.pt`)に「Facts(カード) + 質問 → 答えの文」を追加学習します。分野は動物・植物・天気・体・食べ物・学校・乗り物など23分野、
約9,700問(学習)です。通常会話が壊れないよう、普通の会話も混ぜます。
**これは学習曲線(1万/3万/10万問)の最小点です。** 10万問が必要かは、この結果と次の3万問の結果を並べて決めます。

**評価セット(すべて学習に使っていない)**
- `s10k_test_newsubj`: **学習に出ていない題材**(128題材)の問い。名前は学習で使った種類から付けています。
- `s10k_test_newname`: 同じ題材で、**名前入りカードの名前を、学習に出ていない名前(300種)に替えたもの**(193件)。名前を正しく書き写せるかを見ます。
- `s10k_test_seen`: 学習した題材の問い(500件)。記憶の確認です(これが高くても、読解力の証拠にはなりません)。

**収束の確認(省略しません)**: ①5エポック、②**15エポック(3倍)**、③5エポックを別シード2つ、の順に回します。
学習の10%・25%・50%・100%時点の val_loss を出すので、下がり止まったか、エポックを増やすと結果が変わるかを見ます。

**Drive の `MyDrive/llm5k/` に置くファイル**
- 既存: `sft_5k_v2.pt`, `tokenizer_5k.json`, `dialogues_train_expanded.jsonl`, `dialogues_val.jsonl`, `model.py`
- ★新規(`colab_readqa_upload/` にまとめてあります): `train_readqa.py`, `s10k_train.jsonl`, `s10k_test_newsubj.jsonl`, `s10k_test_newname.jsonl`, `s10k_test_seen.jsonl`

T4で、①が約10分、②が約25分、③が約20分の見込みです(未計測)。結果は `readqa_*.json` と `readqa_*.pt` として Drive に保存します。""")
code("""from google.colab import drive
drive.mount('/content/drive')
import os, shutil, sys, json
DIR = '/content/drive/MyDrive/llm5k'
need = ['sft_5k_v2.pt', 'tokenizer_5k.json', 'dialogues_train_expanded.jsonl', 'dialogues_val.jsonl', 'model.py',
        'train_readqa.py', 's10k_train.jsonl', 's10k_test_newsubj.jsonl', 's10k_test_newname.jsonl', 's10k_test_seen.jsonl']
for f in need:
    p = os.path.join(DIR, f)
    print(f, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else '')
assert all(os.path.exists(os.path.join(DIR, f)) for f in need), 'MyDrive/llm5k/ に不足ファイルを置いてください'
WORK = '/content/work'
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
for f in need:
    shutil.copy(os.path.join(DIR, f), WORK)
os.chdir(WORK)
!pip -q install tokenizers""")
md("""## 1. 5エポック・シード0
途中経過(10/25/50/100%)と、3つの評価セットの結果が出ます。""")
code("""TESTS = 's10k_test_newsubj.jsonl s10k_test_newname.jsonl s10k_test_seen.jsonl'
import subprocess
def run(epochs, seed, data='s10k'):
    tag = f'e{epochs}_seed{seed}' + ('' if data == 's10k' else '_' + data)
    cmd = (f'python -u train_readqa.py --data /content/work --ckpt-in /content/work/sft_5k_v2.pt '
           f'--ckpt-out /content/drive/MyDrive/llm5k/readqa_{tag}.pt --train-file {data}_train.jsonl --tests {TESTS} '
           f'--epochs {epochs} --seed {seed} --log /content/drive/MyDrive/llm5k/readqa_{tag}.json')
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p, open(f'log_{tag}.txt', 'w') as lf:
        for line in p.stdout:
            print(line, end=''); lf.write(line)
run(5, 0)""")
md("""## 2. 15エポック(3倍)・シード0
5エポックと比べて、評価が大きく変わるなら、5エポックは学習不足です。""")
code("""run(15, 0)""")
md("""## 3. 5エポック・シード1、2
結果が安定しているか(シードで変わらないか)を見ます。""")
code("""run(5, 1)
run(5, 2)""")
md("""## 3b. 15エポック・シード1、2(収束確認の続き)
前回、15エポックはシード0だけでした。シード1、2を足して、15エポックの結果が安定しているかを見ます(各約14分)。""")
code("""run(15, 1)
run(15, 2)""")
md("""## 3c. 3万問・5エポック(件数の効果を見る)
`s30k_train.jsonl`(1万問の続きに2万問を足したもの)を使います。評価セットは1万問のときと**同じ**(`s10k_test_*`)で、評価用の題材は学習に出ていません。
3万問×5エポックは約11,400ステップで、1万問×15エポックとほぼ同じステップ数です。**ステップ数をそろえて、データ量の効果を見る**ための比較です。
先に `colab_readqa_upload/` の `s30k_train.jsonl` を Drive の `llm5k/` に置いてください(約15MB)。""")
code("""assert os.path.exists(os.path.join(DIR, 's30k_train.jsonl')), 's30k_train.jsonl を MyDrive/llm5k/ に置いてください'
shutil.copy(os.path.join(DIR, 's30k_train.jsonl'), WORK)
run(5, 0, 's30k')""")
md("""### 3万問・15エポック(任意)
3万問でも5エポックが学習不足かを見るための、3倍のエポックです(約40分)。""")
code("""run(15, 0, 's30k')""")
md("""## 3d. 3万問・15エポック・シード1、2(3万問の収束確認の続き)
3万問は15エポックがシード0だけです。シード1、2を足して、結果が安定しているかを見ます(各約45分)。""")
code("""run(15, 1, 's30k')
run(15, 2, 's30k')""")
md("""## 3e. 10万問・5エポック(件数の効果の3点目)
`s100k_train.jsonl`(約10.2万問、題材7,388)を使います。評価セットはこれまでと同じです。約34,000ステップで、約50分の見込みです(未計測)。
先に `colab_readqa_upload/` の `s100k_train.jsonl` を Drive の `llm5k/` に置いてください。""")
code("""assert os.path.exists(os.path.join(DIR, 's100k_train.jsonl')), 's100k_train.jsonl を MyDrive/llm5k/ に置いてください'
shutil.copy(os.path.join(DIR, 's100k_train.jsonl'), WORK)
run(5, 0, 's100k')""")
md("""## 3f. 10万問・15エポック(約2.5時間)
10万問でも5エポックが学習不足かを見る、3倍のエポックです(約100,000ステップ)。3eの後に続けて回します。Colab の接続が切れないよう、ブラウザのタブを開いたままにしてください。""")
code("""run(15, 0, 's100k')""")
md("""## 4. 結果の表(収束確認つき)
5エポック(3シード)と15エポックの、途中経過の val_loss と評価を並べます。""")
code("""import glob
rows = []
for p in sorted(glob.glob(DIR + '/readqa_e*_seed*.json')):
    d = json.load(open(p))
    a = d['args']
    print(f"\\n### {a['train_file']} epochs={a['epochs']} seed={a['seed']} (学習{d['train_n']}問, {d['steps']}step) 途中経過 val_loss:", d['marks'])
    for name, r in d['results'].items():
        print('  ', name.replace('.jsonl', ''), r)""")
md("""## 5. 試してみる(自由に)
`facts` を書き換えて、`question` を変えて試せます。使うのは、上で保存した `readqa_e5_seed0.pt` です。""")
code("""import torch
sys.path.insert(0, WORK)
from model import GPT, Config
from tokenizers import Tokenizer
tok = Tokenizer.from_file('tokenizer_5k.json'); tok.add_special_tokens(['<user>', '<bot>', '<memory>'])
BOS, EOS, USER, BOT = tok.token_to_id('<bos>'), tok.token_to_id('<eos>'), tok.token_to_id('<user>'), tok.token_to_id('<bot>')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=256)).to(dev)
model.load_state_dict(torch.load(os.path.join(DIR, 'readqa_e5_seed0.pt'), map_location=dev)['model']); model.eval()


@torch.no_grad()
def answer(card, question, max_new=60):
    prompt = 'Facts: ' + card + ' Question: ' + question
    ids = [BOS, USER] + tok.encode(prompt).ids + [BOT]; n0 = len(ids)
    for _ in range(max_new):
        logits, _ = model(torch.tensor([ids[-256:]], device=dev))
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS: break
        ids.append(nxt)
    return tok.decode(ids[n0:]).strip()


card = 'Mimi has a red kite. The kite flies high when the wind blows. Mimi likes to fly it at the park.'
for q in ['What color is the kite?', 'Where does Mimi fly the kite?', 'Does the kite fly high when there is no wind?', 'How old is Mimi?']:
    print('Q:', q, '\\nA:', answer(card, q), '\\n')""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
up = "colab_readqa_upload"
os.makedirs(up, exist_ok=True)
json.dump(nb, open(os.path.join(up, "readqa_colab.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for src in ["train_readqa.py", "model.py"] + [f"data/reading_qa/s10k_{n}.jsonl" for n in ("train", "test_newsubj", "test_newname", "test_seen")]:
    shutil.copy(src, os.path.join(up, os.path.basename(src)))
print(sorted(os.listdir(up)))
