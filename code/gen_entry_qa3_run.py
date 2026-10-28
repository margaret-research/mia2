# -*- coding: utf-8 -*-
"""MIA2 項目3(読解): 題材リスト(topics_themes_v2.json)の全題材について、説明文+質問20問を DeepSeek で作る (オーナー承認 2026-10-25)

gen_entry_qa3.py の build() を使う。題材は1つずつ全部(テーマ順にばらして)。失敗した題材は最大2回までやり直す。
100題材ごとに data/reading_entry/full/raw3_<N>.jsonl に保存(質問まで完成した項目だけ)、失敗は fail_<N>.jsonl。
停止: data/reading_entry/full/STOP を置く。再開: そのまま再実行(保存済みの題材は飛ばす)。費用上限 COST_CAP(USD、既定 5)。
使い方: COST_CAP=5 python gen_entry_qa3_run.py   (先にvLLMサーバーを起動、DEEPSEEK_API_KEY が必要)
"""
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gen_chat_qa as ds
import gen_entry_qa as G
import gen_entry_qa3 as Q3

OUT = os.path.join(G.OUT, os.environ.get("OUT_SUB", "full"))
STOP = os.path.join(OUT, "STOP")
CHUNK = 100
os.makedirs(OUT, exist_ok=True)
ds.COST_CAP = float(os.environ.get("COST_CAP", 5.0))


def work(job):
    title, domain, seed = job
    last = None
    for k in range(3):
        if os.path.exists(STOP):
            return {"title": title, "domain": domain, "fail": "stopped"}
        try:
            r = Q3.build(title, domain, random.Random(seed + k))
        except Exception as e:
            if "COST_CAP" in str(e):
                raise
            r = {"title": title, "domain": domain, "fail": f"error:{str(e)[:60]}"}
        if "qas" in r:
            r["tries"] = k + 1
            return r
        last = r
    return last


def main():
    topics = json.load(open(os.path.join(G.OUT, os.environ.get("TOPICS_FILE", "topics_themes_v2.json")), encoding="utf-8"))
    allj = [(t, d) for d, v in topics.items() for t in v]
    random.Random(7).shuffle(allj)
    done = set()
    for f in glob.glob(os.path.join(OUT, "raw3_*.jsonl")):
        done |= {json.loads(l)["title"] + "|" + json.loads(l)["domain"] for l in open(f, encoding="utf-8")}
    for f in glob.glob(os.path.join(OUT, "fail_*.jsonl")):
        done |= {json.loads(l)["title"] + "|" + json.loads(l)["domain"] for l in open(f, encoding="utf-8")}
    todo = [(t, d, 1000 + i * 7) for i, (t, d) in enumerate(allj) if t + "|" + d not in done]
    n_files = len(glob.glob(os.path.join(OUT, "raw3_*.jsonl")))
    print(f"題材 {len(allj)} 個 / 保存済み {len(done)} / これから {len(todo)}  費用上限 ${ds.COST_CAP}", flush=True)
    t0, ok_n, fail_n = time.time(), 0, 0
    for c in range(0, len(todo), CHUNK):
        if os.path.exists(STOP):
            break
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(work, todo[c:c + CHUNK]))
        res = [r for r in res if r.get("fail") != "stopped"]
        ok = [r for r in res if "qas" in r]
        bad = [r for r in res if "qas" not in r]
        k = n_files + c // CHUNK
        with open(os.path.join(OUT, f"raw3_{k}.jsonl"), "w", encoding="utf-8") as f:
            for r in ok:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(OUT, f"fail_{k}.jsonl"), "w", encoding="utf-8") as f:
            for r in bad:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok_n += len(ok); fail_n += len(bad)
        print(f"[保存] {k}: 完成 {ok_n} / 失敗 {fail_n} / 経過 {time.time() - t0:.0f}s / DeepSeek費用の見積もり ${ds.cost:.3f}", flush=True)
    print(f"終了: 完成 {ok_n} 失敗 {fail_n} 経過 {time.time() - t0:.0f}s 費用 ${ds.cost:.3f}", flush=True)


if __name__ == "__main__":
    main()
