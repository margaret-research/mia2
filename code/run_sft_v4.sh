#!/bin/bash
# MIA2 新トークナイザー(ts_v4: 読解にだけ出る145語を足した)での会話SFT → 読解SFT。基礎は埋め込みに行を足した pre_stories_B1024_v4.pt
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 MIA_TOK=data/ts_v4/tokenizer_ext.json
PY=python
$PY sft_chat.py --ckpt-in ckpt/pre_stories_B1024_v4.pt --data data/chat_mix_v4 --out ckpt/chat_sft_mix_B1024_v4.pt --epochs 3 --freeze-emb --block 1024 --max-tok 32768 > logs/chat_sft_mix_B1024_v4.log 2>&1
$PY sft_chat.py --ckpt-in ckpt/chat_sft_mix_B1024_v4.pt --data data/read_mix_v4 --out ckpt/read_sft_B1024_v4.pt --epochs 2 --lr 2e-4 --freeze-emb --block 1024 --max-tok 32768 --val-split 100 > logs/read_sft_B1024_v4.log 2>&1
touch logs/sft_v4.finished
