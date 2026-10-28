#!/bin/bash
# MIA2 読解SFT 第5版: 質問を約2倍(追加質問を含む学習71,081問、読解25,020サンプル)+会話の復習108,000、会話SFT済み(新語の行つき)から5エポック、lr 2e-4
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 MIA_TOK=data/ts_v4/tokenizer_ext.json
python sft_chat.py --ckpt-in ckpt/chat_sft_mix_B1024_v4ext.pt --data data/read_mix_v5 --out ckpt/read_sft_B1024_v5.pt --epochs 5 --lr 2e-4 --freeze-emb --block 1024 --max-tok 32768 --val-split 100 > logs/read_sft_B1024_v5.log 2>&1
touch logs/read_sft_B1024_v5.finished
