#!/bin/bash
# MIA2 読解SFT 第2版: 読解6,330サンプル+会話の復習108,000(短90,000/長18,000)、2エポック、lr 2e-4、会話SFT済みから
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
python sft_chat.py --ckpt-in ckpt/chat_sft_mix_B1024.pt --data data/read_mix2 --out ckpt/read_sft2_B1024.pt --epochs 2 --lr 2e-4 --freeze-emb --block 1024 --max-tok 32768 --val-split 100 > logs/read_sft2_B1024.log 2>&1
touch logs/read_sft2_B1024.finished
