#!/bin/bash
# MIA2 会話SFT(短い会話+長い会話の混合): pre_stories_B1024.pt から。設定は前と同じ(3エポック、lr 3e-4、埋め込み固定)。長い会話のため1バッチ最大32,768トークン
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
python sft_chat.py --ckpt-in ckpt/pre_stories_B1024.pt --data data/chat_mix --out ckpt/chat_sft_mix_B1024.pt --epochs 3 --freeze-emb --block 1024 --max-tok 32768 > logs/chat_sft_mix_B1024.log 2>&1
touch logs/chat_sft_mix_B1024.finished
