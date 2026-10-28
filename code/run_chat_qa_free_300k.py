# -*- coding: utf-8 -*-
"""T-026 制約なしの会話生成 (gen_chat_qa_free.py) を、区切り(shard)ごとに回して30万会話まで積み上げる

1 shard = 2,500 回の呼び出し (約1.4万会話, 約50分)。shard ごとに保存するので、途中で止めてもそこまでの分は使える。
  data/chat_qa/raw_f<N>.jsonl / rej_f<N>.jsonl
重複排除: 前の shard までの最初の発言を読み込んで、同じ最初の発言は捨てる (完全一致のみ。似た質問は残る)。
止め方: data/chat_qa/STOP というファイルを置く (今の shard が終わってから止まる)。再開は同じコマンド(完了済み shard は飛ばす)。
使い方: python run_chat_qa_free_300k.py [目標の会話数=300000]     (先に vLLM サーバーを起動しておく)
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "chat_qa")
PY = sys.executable
SHARD_CALLS = 2500
K = 8
BASE_TAGS = ["free_pilot", "free_pilot2"]


def n_lines(path):
    return sum(1 for _ in open(path, encoding="utf-8"))


def main(goal):
    n = 1
    while True:
        total = sum(n_lines(p) for p in glob.glob(os.path.join(OUT, "raw_f[0-9]*.jsonl")))
        print(f"[shard前] 採用累計 {total} / 目標 {goal}", flush=True)
        if total >= goal or os.path.exists(os.path.join(OUT, "STOP")):
            print("終了 (目標到達 or STOP)", flush=True)
            return
        tag = f"f{n}"
        if os.path.exists(os.path.join(OUT, f"raw_{tag}.jsonl")):
            n += 1
            continue
        prev = BASE_TAGS + [f"f{i}" for i in range(1, n)]
        env = dict(os.environ, SEED=str(5000 + n), SEEN_FROM=",".join(prev), WORKERS="16")
        # gen_chat_qa_free.py は 呼び出し数 = target / K * 1.15 なので、呼び出し数 SHARD_CALLS になるよう割り戻す
        subprocess.run([PY, "gen_chat_qa_free.py", tag, str(int(SHARD_CALLS * K / 1.15))], cwd=HERE, env=env, check=True)
        n += 1


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300000)
