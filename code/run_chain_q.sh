#!/bin/bash
cd "$(dirname "$0")"
T="s10k_test_newsubj.jsonl s10k_test_newname.jsonl s10k_test_seen.jsonl m_test_mismatch.jsonl"
python sft_chat.py --ckpt-in ckpt/ts_pre_v3q.pt --out ckpt/chat_sft_q.pt --epochs 3 --freeze-emb > logs/chat_sft_q.log 2>&1
python train_readqa_ext.py --freeze-emb --ckpt-in ckpt/chat_sft_q.pt --ckpt-out ckpt/readqa_q_s1.pt --train-file m120k_train.jsonl --tests $T --epochs 15 --lr 1.5e-4 --log logs/readqa_q_s1.json > logs/readqa_q_s1.log 2>&1
python train_readqa_ext.py --freeze-emb --ckpt-in ckpt/readqa_q_s1.pt --ckpt-out ckpt/readqa_q.pt --train-file m_para_train.jsonl --tests $T --epochs 5 --lr 1e-4 --log logs/readqa_q.json > logs/readqa_q.log 2>&1
touch logs/chain_q.done
