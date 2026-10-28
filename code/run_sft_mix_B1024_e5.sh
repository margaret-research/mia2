#!/bin/bash
# MIA2 会話SFT(混合)を5エポックで。3エポック版(run_sft_mix_B1024.sh)との比較用。ほかは同じ
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
python sft_chat.py --ckpt-in ckpt/pre_stories_B1024.pt --data data/chat_mix --out ckpt/chat_sft_mix_B1024_ep5.pt --epochs 5 --freeze-emb --block 1024 --max-tok 32768 --val-split 3000 > logs/chat_sft_mix_B1024_ep5.log 2>&1
touch logs/chat_sft_mix_B1024_ep5.finished
