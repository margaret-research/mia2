# -*- coding: utf-8 -*-
"""MIA2 項目6: 長い会話(10/20/30往復を3:2:1)を Qwen(ローカルvLLM)で作り、採用した会話の自作トークン合計が目標に届くまで続ける

gen_long_chat.py の one()/parse() をそのまま使う(制約なし生成。「10歳の子の言葉で」と頼むだけ)。
採用の条件(最小限): 形式が読める / ターン数が偶数で4以上(BOTで終わる) / 自作トークン(+役割トークン)が 1,024 に収まる /
                    評価の固定62問と同じユーザー発言を含まない / 最初の発言が既存と完全に同じではない。
出力: data/long_chat/run/raw_lf<N>.jsonl (JOBS_PER_FILE件の生成ごとに1ファイル。採用した会話だけ) と rej_lf<N>.jsonl (理由つき)
停止: data/long_chat/STOP というファイルを置く。再開: そのまま再実行(既存のファイルは数えて続きから)。
使い方: python gen_long_chat_run.py [目標の自作トークン数=20e6]    (先に vLLM サーバーを起動)
生成テキストは信頼できない外部由来として扱う(整形と保存だけ)。
"""
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import gen_chat_qa as base
import gen_chat_qa_local as local
import gen_long_chat as gl

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 20e6
OUT = os.path.join(gl.OUT, "run")
STOP = os.path.join(gl.OUT, "STOP")
JOBS_PER_FILE = 1000
PATTERN = [10, 10, 10, 20, 20, 30]          # 会話数の比 3:2:1
os.makedirs(OUT, exist_ok=True)


def own_tokens(turns):
    return sum(len(local.tok.encode(t).ids) + 2 for t in turns) + 1     # 役割トークンと<eos>、先頭の<bos>


def accept(r, seen):
    turns = r["turns"]
    if not turns:
        return "unreadable", None
    if len(turns) % 2 or len(turns) < 4:
        return f"turn-count:{'odd' if len(turns) % 2 else 'short'}", None
    n = own_tokens(turns)
    if n > 1024:
        return "over-1024", None
    if any(base.norm(t) in base.EVAL_USER for t in turns[0::2]):
        return "eval-overlap", None
    return None, n          # 最初の発言の重複は不採用にしない(オーナー指示 2026-10-24。会話全体は別物のため)


def main():
    done_files = sorted(glob.glob(os.path.join(OUT, "raw_lf*.jsonl")))
    seen, tokens, convs = set(), 0, 0
    for f in done_files:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            seen.add(base.norm(r["turns"][0])); tokens += r["own_tokens"]; convs += 1
    next_file = len(done_files)
    job_idx = next_file * JOBS_PER_FILE            # 再開しても同じ番号から(乱数も同じにならないよう、ファイル番号で種をずらす)
    print(f"開始: 既存 {len(done_files)} ファイル {convs} 会話 {tokens / 1e6:.2f}Mトークン / 目標 {TARGET / 1e6:.0f}Mトークン", flush=True)
    t0, batch_ok, batch_rej, batch_jobs, fails, inflight = time.time(), [], [], 0, 0, {}
    stats = {}

    def submit(ex):
        nonlocal job_idx
        exch = PATTERN[job_idx % len(PATTERN)]
        seed = random.Random(job_idx * 7919 + 2026).randrange(10**9)
        inflight[ex.submit(gl.one, (job_idx, exch, seed))] = exch
        job_idx += 1

    def flush():
        nonlocal next_file, batch_ok, batch_rej, batch_jobs
        for name, data in ((f"raw_lf{next_file}.jsonl", batch_ok), (f"rej_lf{next_file}.jsonl", batch_rej)):
            with open(os.path.join(OUT, name), "w", encoding="utf-8") as fo:
                for r in data:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        next_file += 1; batch_ok, batch_rej, batch_jobs = [], [], 0

    with ThreadPoolExecutor(local.WORKERS) as ex:
        while len(inflight) < local.WORKERS:
            submit(ex)
        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in finished:
                exch = inflight.pop(f)
                try:
                    r = f.result()
                    fails = 0
                except Exception as e:
                    fails += 1
                    print(f"[エラー] {e}", file=sys.stderr, flush=True)
                    if fails >= 20:
                        print("[停止] 連続エラー", flush=True); open(STOP, "w").close()
                    batch_jobs += 1
                    continue
                why, n = accept(r, seen)
                batch_jobs += 1
                stats[why or "ok"] = stats.get(why or "ok", 0) + 1
                if why:
                    batch_rej.append({"why": why, "exch": exch, "turns": r["turns"]})
                else:
                    seen.add(base.norm(r["turns"][0]))
                    batch_ok.append({"exch": exch, "scenario": r["scenario"], "topic": r["topic"], "turns": r["turns"], "own_tokens": n})
                    tokens += n; convs += 1
                if batch_jobs >= JOBS_PER_FILE:
                    flush()
                    print(f"[保存] raw_lf{next_file - 1}: 採用累計 {convs} 会話 {tokens / 1e6:.2f}Mトークン ({tokens / TARGET:.0%}) 経過 {time.time() - t0:.0f}s  内訳 {stats}", flush=True)
                if tokens < TARGET and not os.path.exists(STOP):
                    submit(ex)
    if batch_jobs:
        flush()
    print(f"終了: 採用 {convs} 会話 {tokens / 1e6:.2f}Mトークン 経過 {time.time() - t0:.0f}s 内訳 {stats}", flush=True)


if __name__ == "__main__":
    main()
