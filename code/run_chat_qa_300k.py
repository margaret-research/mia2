# -*- coding: utf-8 -*-
"""T-026 会話データを30万件(OK判定後)まで、区切り(shard)ごとに 生成 → OK/BAD判定 しながら積み上げる

1 shard = 20,000 回の生成 (採用は約1万弱、判定でさらに半分ほど残る)。shard ごとに保存するので、途中で止めてもそこまでの分は使える。
  data/chat_qa/raw_s<N>.jsonl / rej_s<N>.jsonl / scored_s<N>.jsonl / kept_s<N>.jsonl
重複排除: 前の shard までの最初の発言(raw_/rej_)を読み込んで、同じ最初の発言は作った時点で捨てる。
止め方: data/chat_qa/STOP というファイルを置く (今の shard が終わってから止まる)。再開は同じコマンド(完了済み shard は飛ばす)。
使い方: python run_chat_qa_300k.py [目標のOK件数=300000]     (先に vLLM サーバーを起動しておく)
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "chat_qa")
PY = sys.executable
SHARD_CALLS = 20000
BASE_TAGS = ["gbnf_pilot", "gbnf_pilot2", "gbnf_test", "gbnf_3k", "gbnf_ext_test", "gbnf_3k_b"]


def n_lines(path):
    return sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0


def main(goal):
    n = 1
    while True:
        kept = sum(n_lines(p) for p in glob.glob(os.path.join(OUT, "kept_s*.jsonl")))
        print(f"[shard前] OK累計 {kept} / 目標 {goal}", flush=True)
        if kept >= goal or os.path.exists(os.path.join(OUT, "STOP")):
            print("終了 (目標到達 or STOP)", flush=True)
            return
        tag = f"s{n}"
        if os.path.exists(os.path.join(OUT, f"kept_{tag}.jsonl")):
            n += 1
            continue
        prev = BASE_TAGS + [f"s{i}" for i in range(1, n)]
        env = dict(os.environ, SEED=str(1000 + n), EXT="1", WORDSEED="1", SEEN_FROM=",".join(prev))
        # gen_chat_qa_gbnf.py の target は「呼び出し数 = target*1.2」なので、呼び出し数 SHARD_CALLS になるよう割り戻す
        subprocess.run([PY, "gen_chat_qa_gbnf.py", tag, str(int(SHARD_CALLS / 1.2))], cwd=HERE, env=env, check=True)
        subprocess.run([PY, "score_chat_qa_local.py", tag], cwd=HERE, env=env, check=True)
        n += 1


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300000)
