#!/bin/bash
# MIA2 読解SFT: 会話SFT済み chat_sft_mix_B1024.pt から。読解6,330サンプル+会話の復習36,000(短30,000/長6,000)を混ぜて3エポック、lr 2e-4、埋め込み固定
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
python sft_chat.py --ckpt-in ckpt/chat_sft_mix_B1024.pt --data data/read_mix --out ckpt/read_sft_B1024.pt --epochs 3 --lr 2e-4 --freeze-emb --block 1024 --max-tok 32768 --val-split 100 > logs/read_sft_B1024.log 2>&1
touch logs/read_sft_B1024.finished
