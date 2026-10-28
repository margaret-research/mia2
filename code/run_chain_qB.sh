#!/bin/bash
# Bモデル: 10層 / 8ヘッド / 512次元(約3,977万)。Qwen由来の埋め込みを固定。データ・手順はv3qと同じ
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
T="s10k_test_newsubj.jsonl s10k_test_newname.jsonl s10k_test_seen.jsonl m_test_mismatch.jsonl"
python pretrain_ts.py --data data/ts_v3 --out ckpt/ts_pre_v3qB.pt --tokens 1.64e9 --emb-init ckpt/emb_init_v3_512.pt --freeze-emb > logs/ts_pre_v3qB.log 2>&1
python sft_chat.py --ckpt-in ckpt/ts_pre_v3qB.pt --out ckpt/chat_sft_qB.pt --epochs 3 --freeze-emb > logs/chat_sft_qB.log 2>&1
python train_readqa_ext.py --freeze-emb --ckpt-in ckpt/chat_sft_qB.pt --ckpt-out ckpt/readqa_qB_s1.pt --train-file m120k_train.jsonl --tests $T --epochs 15 --lr 1.5e-4 --log logs/readqa_qB_s1.json > logs/readqa_qB_s1.log 2>&1
python train_readqa_ext.py --freeze-emb --ckpt-in ckpt/readqa_qB_s1.pt --ckpt-out ckpt/readqa_qB.pt --train-file m_para_train.jsonl --tests $T --epochs 5 --lr 1e-4 --log logs/readqa_qB.json > logs/readqa_qB.log 2>&1
touch logs/chain_qB.done
