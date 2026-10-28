# Mia (MIA2) プログラム使い方マニュアル

小学校低学年向けの英語会話Bot「Mia」(約4,000万パラメータの自作GPT)の第2世代のコードです。
開発の経緯と結果は `reports/開発報告書_MIA2_2026-10-28.pdf` を見てください(第1世代の報告書は、この一式には含みません)。
作成: マーガレットリサーチ (2026-10-28)。コードの多くはClaude Codeが書きました。

> **ライセンスの確認結果 (2026-10-28、Hugging Faceのカードと規約で確認)**
> - TinyStories: `cdla-sharing-1.0`。データそのものを再配布するときだけ同じ条件が付く。学習に使って作ったモデルとコードの公開は可能。**この一式はデータを含まない**。
> - SimpleStories: `mit`。 / Qwen3-Embedding-0.6B: `apache-2.0`。
> - DeepSeek (項目・質問・採点): API規約 4.2 で、出力を他のモデルの学習に使うことを認めている。公開時は、DeepSeekの名前を推薦のように使わず、出力がAI生成で誤りを含みうると明記する (規約 8.1)。
> - 会話データの生成に使ったQwen (`RedHatAI/Qwen3.8-27B-INT4`、基はQwen/Qwen3.8-27B): `apache-2.0`。カードに出力の利用制限の記載なし。法的な助言ではなく、規約の読み取り。
> - 結果は単一の乱数の種、採点はLLM(DeepSeek)。数値は比較用の目安。
> - 重み (`.pt`) と生成データ (項目など) は Git に入れず別の置き場に置く。この一式は**コードだけ**。

## 1. 全体の構成

```
ユーザーの発言
   └─ ルーターMia v6 ──┬─ calc 式 → 電卓 (calc_tool.solve) → 答え役Mia (answer_mia_v1) が文にする
                       ├─ card 題名 → 項目の検索 (Qwen3-Embedding-0.6B + 採用の下限 0.50/0.70) → 現行Miaが項目を読んで答える
                       └─ chat → 現行Mia (mia_current_read.pt) が会話
```

3つのMiaはどれも同じ形のモデル(10層 / 8ヘッド / 512次元、文脈長1,024、語彙16,212語)です。重みだけが違います。

| 役 | 重み | 作り方 |
|---|---|---|
| 現行Mia (会話・項目を読む) | `ckpt/mia_current_read.pt` | 基礎学習 → 会話SFT → 読解SFT |
| ルーターMia v6 | `ckpt/router_mia_v6.pt` | 現行Miaのコピーを微調整 |
| 答え役Mia | `ckpt/answer_mia_v1.pt` | 現行Miaのコピーを微調整 |

## 2. 準備

### 2.1 必要なもの
- GPU 1枚 (24GB。RTX 4090で動作確認)。Botの実行だけならずっと小さくて済みます。
- Python 3.11系。使うパッケージ: `torch`, `tokenizers`, `numpy`, `sentence-transformers`, `transformers`。動作確認の版は sentence-transformers 6.1.0 / transformers 5.19.0。
- 学習・採点用: `nltk`, `wordfreq`, `pyarrow` (データ作りだけ)、`vllm` (Qwenでデータ生成するときだけ)。
- DeepSeek APIキー (環境変数 `DEEPSEEK_API_KEY`): 会話テスト・実機会話テスト・項目作成・強化学習でだけ必要。**Botの実行には不要**です。

### 2.2 置くファイル
1. `code/` を作業フォルダにします (以下、ここで実行)。
2. 重みとデータの束 `mia2_runtime_files.tar` (約500MB) を、次のリンクから取得します。
   **取得先: (準備中。Hugging Face に置いたら、ここにURLを書く)** — `<ここにURL>`
   (md5 `0e9dad7c9944bd3d150c6ff593b85f47`。取得後に `md5sum` で確認)
   取得した tar を `code/` の中で展開します。
   ```
   cd code && tar xf mia2_runtime_files.tar
   ```
   これで次が置かれます。
   - `ckpt/mia_current_read.pt`, `ckpt/router_mia_v6.pt`, `ckpt/answer_mia_v1.pt` (各約161MB)
   - `data/ts_v4/tokenizer_ext.json` (語彙16,212語のトークナイザー)
   - `data/reading_entry/` (項目2,226件と質問)
   - `data/card_emb_entries.npy` (項目の埋め込みの保存。なければ初回に作り直されます)
3. 初回はQwen3-Embedding-0.6Bが自動でダウンロードされます (Hugging Face)。

### 2.3 環境変数 (どのコマンドにも必要)
```
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
export MIA_TOK=data/ts_v4/tokenizer_ext.json
export PYTHONPATH=.
```
`MIA_NLAYER` などを付け忘れると、モデルの大きさが合わず重みが読めません。

> 旧実験用のスクリプト(カード網羅の実験、初見の質問の作成)は、この一式から外してあります。`run_*.sh` はスクリプトのある場所で実行する形に直してあり、Qwenの置き場所は環境変数 `QWEN_DIR` で指定します。Pythonは使う環境の `python` が呼ばれます。

## 3. Botを動かす

### 3.1 対話 (おすすめ)
```
python bot_try.py
```
`You:` に英語で話しかけます。`exit` で終了。返事の前の `[calc]` `[card]` `[chat]` は、どの経路で答えたかです。
会話役の重みを差し替えるとき: `CHAT=ckpt/別のモデル.pt python bot_try.py`

試す例:
- 計算: `I have 12 apples. I give 5 to Tom. How many do I have left?`
- カード: `Tell me about the moon.` / `What do grasshoppers eat?`
- 会話: `Hi! How are you?`

### 3.2 コマンドで1回ずつ
```
python bot_mia2.py "What is 12 plus 5?" "Tell me about Mars."
```

### 3.3 Pythonから使う
```python
import bot_mia2 as B
b = B.Bot()                      # router, chat, answer の重みを指定可
way, reply = b.say("What is 3 plus 4?")
```
履歴はBotが覚えます (ルーターと会話には直近3往復を渡す)。

### 3.4 もとの会話モデルだけと話す
```
python talk_mia.py ckpt/mia_current_read.pt 0.7     # 重み・温度。空行で終了、/reset でやり直し
```

### 3.5 仕組みの注意
- 経路の判断は、ルーターの出力の**最初の語**が `calc` / `card` / それ以外 (chat) かで決まります。
- カードの採用の下限は `card_retrieve.py` の `LO, HI = 0.50, 0.70`。上げると作り話は減り、正しい項目も落ちます (報告書の表を参照)。
- 項目にない話題は `I do not know.` と答えます。ただし似た項目から作り話をすることが約3割残っています (既知の弱点)。

## 4. 評価 (基準テスト5つ)

問題と採点基準は凍結してあります。変えるときは旧版と並べて記録します。詳細は `benchmark/README.md`。
`<ckpt>` はテストするモデルの重みです。

| # | テスト | 実行 | 時間・費用 |
|---|---|---|---|
| 1 | 読解 (931問の正解率) | `python benchmark/eval_read.py <ckpt>` | 約1分・無料 |
| 2 | 会話 (DeepSeek採点、100点満点) | `COST_CAP=1 python benchmark/eval_chat5.py <ckpt> <タグ>` | 約2分・約$0.07 |
| 3 | 計算 | `TEST=data/router/test_v2aug.jsonl python eval_router.py <ルーターckpt>` と `python eval_answer.py <答え役ckpt>` | 各1〜2分・無料 |
| 4 | カード全体 | `python eval_card_e2e.py <ルーターckpt> [読解ckpt]` | 数分・無料 |
| 5 | 実機会話 (24場面120発言、合格80点) | `COST_CAP=1 python benchmark/eval_live.py <ルーターckpt> <タグ>` | 数分・約$0.03 |

- DeepSeekを使うもの (2, 5) は、キーを環境変数に入れて実行します。キーをファイルやログに書かないこと。
- テスト3・4は `data/router/` の凍結ファイルが要ります (この一式の外。`gen_router_data.py` / `build_router_data.py` / `build_card_router3.py` で作り直せますが、乱数が同じでないと数値は一致しません)。
- 実機会話テストの会話役は `CHAT=<ckpt>` で差し替えられます。結果は `logs/live_<タグ>.jsonl` に出ます。
- GPUは1枚なので、vLLM (Qwen) と学習・評価は同時に動かせません。

現在の値 (2026-10-28): 読解80%、会話84.6、計算の答え0.976、カード全体70.4% (項目なし話題の断り65.6%)、実機会話77.9。

## 5. 作り直す (学習の流れ)

重みを一から作る場合の順序です。各スクリプトの冒頭に使い方が書いてあります。

| 段 | 内容 | 主なスクリプト |
|---|---|---|
| 0 | 語彙・トークナイザー | `build_tokenizer_v4.py`, `extend_emb.py` |
| 1 | 基礎学習データ (TinyStories, SimpleStories) | `prep_tinystories.py`, `prep_simplestories.py` (入力は信頼できない外部データ。`python -I` で実行) |
| 2 | 基礎学習 (文脈長1,024、3エポック、約35億トークン、約3時間) | `run_pre_stories_B1024.sh` → `pretrain_stories.py` |
| 3 | 会話データ (Qwenで作成。vLLMを先に起動) | `gen_long_chat_run.py`, `gen_chat_qa_local.py`, `build_chat_mix.py` |
| 4 | 会話SFT (約10分) | `run_sft_mix_B1024.sh` → `sft_chat.py` |
| 5 | 項目の作成 (DeepSeek。費用あり) | `gen_entry_topics*.py`, `gen_entry_qa*.py` |
| 6 | 読解SFT | `build_read_mix4.py` → `run_sft_read_v5.sh` |
| 7 | ルーターv6 | `gen_card_data3.py` → `build_card_router3.py` → `RV=v6 python build_router_sft.py` → `sft_chat.py --ckpt-in ckpt/mia_current_read.pt --data data/router_sft_v6 --out ckpt/router_mia_v6.pt --epochs 3 --lr 1e-4 --freeze-emb --block 1024 --max-tok 32768` |
| 8 | 答え役 | `gen_answer_data.py`, `build_answer_sft.py` → `sft_chat.py` |
| 9 | 強化学習 (試験的・不採用) | `rl_live.py` |

注意:
- 基礎学習の埋め込みは Qwen3-Embedding を主成分分析で512次元にして**固定**します (`build_emb_init.py`、`--freeze-emb`)。
- 強化学習 (`rl_live.py`) は DeepSeek 採点で100ステップ約$1.0です。報告書のとおり、読解が下がったため採用していません。実機会話テストは報酬や学習データに**使わない**決まりです。
- 費用の数字はプログラムが使用量から見積もった値です。`COST_CAP` (ドル) で上限を設定できます。

## 6. 困ったとき

| 症状 | 原因と対処 |
|---|---|
| 重みを読むときに形のエラー | `MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512` を付ける |
| トークナイザーのキーエラー | `MIA_TOK=data/ts_v4/tokenizer_ext.json` を付ける |
| `data/reading_entry` が見つからない | `code/` 直下で実行しているか確認。`mia2_runtime_files.tar` を展開したか確認 |
| GPUメモリ不足 | vLLM (Qwen) など他の処理を止める (`pkill -f "[v]llm serve"`) |
| DeepSeekが `DEEPSEEK_API_KEY` を要求 | 環境変数にキーを入れる (Botの実行だけなら不要) |
| 返事が `The answer is X.` の固定文 | 答え役の出力が使えなかったときの代わりの文。正常な挙動 |

## 7. ファイル一覧 (主なもの)

- `bot_mia2.py` Bot本体 / `bot_try.py` 対話ランチャー / `talk_mia.py` 会話だけ
- `model.py` GPT / `calc_tool.py` 電卓 (`eval` を使わず、Fractionの自前パーサ)
- `card_retrieve.py` 項目の検索と採用の下限 / `card_stem.py` 語の正規化
- `sft_chat.py` SFT / `pretrain_stories.py` 基礎学習 / `rl_live.py` 強化学習
- `gen_*.py` データ生成 / `build_*.py` データ組み立て / `prep_*.py` 外部データ前処理
- `eval_*.py`, `benchmark/` 評価
- 内部メモ・実験用 (`exp_*`, `probe_*`, `raft_*`, `bot.py` 系の旧版) は、経緯を残すために入っています。使う必要はありません。
