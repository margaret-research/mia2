#!/bin/bash
# MIA2 基礎学習: SimpleStories+TinyStories をランダムに混ぜ、お話ごとに学習。Bモデル(10層/8ヘッド/512次元)、文脈長1,024、Qwen埋め込み固定、3エポック、開始位置を50%の話でずらす
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
python pretrain_stories.py --out ckpt/pre_stories_B1024.pt --epochs 3 --emb-init ckpt/emb_init_v3_512.pt --freeze-emb "$@" > logs/pre_stories_B1024.log 2>&1
touch logs/pre_stories_B1024.done
