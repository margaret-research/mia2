#!/bin/bash
cd "$(dirname "$0")"
kill $(pgrep -f "[b]in/vllm serve"); sleep 25
export MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512
QA=ckpt/readqa_qB_raft.pt TAG=_qBraft python eval_reader_k.py > logs/eval_reader_k_qBraft.log 2>&1
QA=ckpt/readqa_qB_raft.pt TAG=_qBraft_gold python -c "print(1)" >/dev/null
cd "$QWEN_DIR" && VLLM_USE_FLASHINFER_SAMPLER=0 nohup vllm serve models/Qwen3.8-27B-INT4 --served-model-name qwen --host 127.0.0.1 --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.92 --language-model-only --max-num-seqs 16 --max-num-batched-tokens 2048 --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 > logs/vllm_raft.log 2>&1 &
until curl -s -m 3 127.0.0.1:8000/v1/models | grep -q qwen; do sleep 5; done
cd "$(dirname "$0")"
TAG=_qBraft python eval_reader_judge.py > logs/judge_qBraft.log 2>&1
touch logs/eval_raft1.done
