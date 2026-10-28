#!/bin/bash
# MIA2 読解SFT(新トークナイザー ts_v4・項目2,226): 会話SFT済み(埋め込みに新語の行を足した)から5エポック。読解12,510サンプル+会話の復習108,000
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 MIA_TOK=data/ts_v4/tokenizer_ext.json
python sft_chat.py --ckpt-in ckpt/chat_sft_mix_B1024_v4ext.pt --data data/read_mix_v4 --out ckpt/read_sft_B1024_v4.pt --epochs 5 --lr 2e-4 --freeze-emb --block 1024 --max-tok 32768 --val-split 100 > logs/read_sft_B1024_v4.log 2>&1
touch logs/read_sft_B1024_v4.finished
