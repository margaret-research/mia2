#!/bin/bash
cd "$(dirname "$0")"
T="s10k_test_newsubj.jsonl s10k_test_newname.jsonl s10k_test_seen.jsonl m_test_mismatch.jsonl"
python train_readqa_ext.py --freeze-emb --ckpt-in ckpt/chat_sft_3ep_x.pt --ckpt-out ckpt/readqa_v3f_s1.pt --train-file m120k_train.jsonl --tests $T --epochs 15 --lr 1.5e-4 --log logs/readqa_v3f_s1.json > logs/readqa_v3f_s1.log 2>&1
python train_readqa_ext.py --freeze-emb --ckpt-in ckpt/readqa_v3f_s1.pt --ckpt-out ckpt/readqa_v3f.pt --train-file m_para_train.jsonl --tests $T --epochs 5 --lr 1e-4 --log logs/readqa_v3f.json > logs/readqa_v3f.log 2>&1
touch logs/readqa_v3f.done
