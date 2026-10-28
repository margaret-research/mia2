#!/bin/bash
# MIA2 会話SFT: 新しい基礎モデル(pre_stories_B1024.pt、B・文脈長1,024)から。設定は前のB(chat_sft_qB)と同じ: 3エポック、lr 3e-4、埋め込み固定
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
python sft_chat.py --ckpt-in ckpt/pre_stories_B1024.pt --out ckpt/chat_sft_B1024.pt --epochs 3 --freeze-emb --block 1024 > logs/chat_sft_B1024.log 2>&1
touch logs/chat_sft_B1024.finished
