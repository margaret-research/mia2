#!/bin/bash
cd "$(dirname "$0")"
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
T="raft_test.jsonl s10k_test_newsubj.jsonl s10k_test_newname.jsonl s10k_test_seen.jsonl m_test_mismatch.jsonl"
python train_readqa_ext.py --freeze-emb --ckpt-in ckpt/readqa_qB_s1.pt --ckpt-out ckpt/readqa_qB_raft.pt --train-file raft_train.jsonl --tests $T --epochs 5 --lr 1e-4 --log logs/readqa_qB_raft.json > logs/readqa_qB_raft.log 2>&1
touch logs/raft.done
