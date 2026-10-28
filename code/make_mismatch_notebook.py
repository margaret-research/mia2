# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — 不一致IDK(検索で来たカードに答えが無い→わかりません)の追加学習用ノートブック"""
import json, os, shutil

cells = []
md = lambda t: cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
code = lambda t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)})

md("""# 読解SFT: 「カードに答えが無い→わかりません」の追加学習 (会話Bot用)
Claude Code により作成 (2026-10-08)

**目的**: 会話Botでは、検索で質問と関係の薄いカードが来ても、読解モデルがIDK(I do not know.)と言わず答えを作ってしまった(例: 火星の首都→意味不明な文、クジラは魚か→No, a whale is a fish.)。
その対策として、「質問に近いが答えの内容語がほぼ無い別カード」と質問の組を IDK の例として足して学習します。

**条件**: 10万問×15エポック(シード0)と同じ設定で、データだけ変えます。学習は `m120k_train.jsonl` = 元の `s100k_train`(101,828問)+ 不一致IDK 20,000問(計121,828問、IDK 30.9%)。
評価は従来の3セット(s10k_test_*)に、不一致IDKの評価 `m_test_mismatch.jsonl`(評価題材の質問 × 検索で来た別カード、300件)を足します。**比較の相手は、前の10万問×15エポックの結果です。**
見るもの: ①不一致IDKの再現率(これが上がる必要がある) ②答えられる問いの core 再現率が落ちていないか(前: 新題材 91.1%) ③IDKを言いすぎていないか(適合率)

**Drive の `MyDrive/llm5k/` に置くファイル**
- 既存: `sft_5k_v2.pt`, `tokenizer_5k.json`, `dialogues_train_expanded.jsonl`, `dialogues_val.jsonl`, `model.py`, `train_readqa.py`, `s10k_test_newsubj.jsonl`, `s10k_test_newname.jsonl`, `s10k_test_seen.jsonl`
- ★新規(`colab_mismatch_upload/`): `m120k_train.jsonl`(約58MB), `m_test_mismatch.jsonl`

T4で約2時間の見込みです(10万問×15エポックは約100分でした)。ブラウザのタブを開いたままにしてください。""")
code("""from google.colab import drive
drive.mount('/content/drive')
import os, shutil, sys, json
DIR = '/content/drive/MyDrive/llm5k'
need = ['sft_5k_v2.pt', 'tokenizer_5k.json', 'dialogues_train_expanded.jsonl', 'dialogues_val.jsonl', 'model.py', 'train_readqa.py',
        'm120k_train.jsonl', 'm_test_mismatch.jsonl', 's10k_test_newsubj.jsonl', 's10k_test_newname.jsonl', 's10k_test_seen.jsonl']
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
md("""## 学習(15エポック・シード0、約2時間)
学習の10/25/50/100%時点とエポックごとの val_loss(カードQA・通常会話)が出ます。最後に4つの評価セットの結果が出ます。""")
code("""import subprocess
TESTS = 's10k_test_newsubj.jsonl s10k_test_newname.jsonl s10k_test_seen.jsonl m_test_mismatch.jsonl'
def run(epochs, seed, data='m120k'):
    tag = f'e{epochs}_seed{seed}_{data}'
    cmd = (f'python -u train_readqa.py --data /content/work --ckpt-in /content/work/sft_5k_v2.pt '
           f'--ckpt-out /content/drive/MyDrive/llm5k/readqa_{tag}.pt --train-file {data}_train.jsonl --tests {TESTS} '
           f'--epochs {epochs} --seed {seed} --log /content/drive/MyDrive/llm5k/readqa_{tag}.json')
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p, open(f'log_{tag}.txt', 'w') as lf:
        for line in p.stdout:
            print(line, end=''); lf.write(line)
run(15, 0)""")
md("""## 結果の表(前の10万問×15エポックと並べる)""")
code("""import glob
for p in sorted(glob.glob(DIR + '/readqa_e15_seed0_s100k.json')) + sorted(glob.glob(DIR + '/readqa_e15_seed0_m120k.json')):
    d = json.load(open(p)); a = d['args']
    print(f"\\n### {a['train_file']} epochs={a['epochs']} seed={a['seed']} (学習{d['train_n']}問, {d['steps']}step) 途中経過 val_loss:", d['marks'])
    for name, r in d['results'].items():
        print('  ', name.replace('.jsonl', ''), r)""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 0}
up = "colab_mismatch_upload"
os.makedirs(up, exist_ok=True)
json.dump(nb, open(os.path.join(up, "mismatch_colab.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for src in ["data/reading_qa/m120k_train.jsonl", "data/reading_qa/m_test_mismatch.jsonl"]:
    shutil.copy(src, os.path.join(up, os.path.basename(src)))
print(sorted(os.listdir(up)))
