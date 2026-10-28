# -*- coding: utf-8 -*-
"""MIA2 項目3(読解): 同じ項目(本文)に、質問を20問追加する(オーナー承認 2026-10-25)。配分: wh 9 / yes 2 / no 4 / count 3 / unk 2(noが弱いので増やし、unkは既存と重なりやすいので減らす)。
DeepSeekに本文と既存の質問を見せ、「重ならない新しい20問」を作らせる。プログラムが型・形式をそろえ、既存と同じ質問(正規化して一致)を除き、Qwenが各ペアを検証する。
出力: data/reading_entry/extra/qa_<N>.jsonl {title, domain, qas:[{type,q,a}], verdict}  / 失敗は fail_<N>.jsonl。停止: extra/STOP。再開: そのまま再実行。費用上限 COST_CAP。
使い方: COST_CAP=6 python gen_entry_qa4.py [項目数の上限=全部]   (先にvLLMサーバー、DEEPSEEK_API_KEY が必要)"""
import glob, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import gen_chat_qa as ds
import gen_chat_qa_local as local
import gen_entry_qa as G
import gen_entry_qa2 as Q2

OUT = os.path.join(G.OUT, "extra"); STOP = os.path.join(OUT, "STOP"); CHUNK = 200
NEED2 = {"wh": 9, "yes": 2, "no": 4, "count": 3, "unk": 2}
MIN2 = {"wh": 6, "yes": 1, "no": 3, "count": 0, "unk": 1}     # 本文の数字は少なくcountは足りないことが多い。足りない分は少なくてよい(合計15問以上)
NUMW = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|half|both|once|twice|pair|dozen"
os.makedirs(OUT, exist_ok=True)
ds.COST_CAP = float(os.environ.get("COST_CAP", 6.0))
PROMPT = """Below is a short encyclopedia entry for young children (grade 1-2) and the questions that were already asked about it.
Title: {title}
Entry: {entry}
Already asked (do NOT repeat these, and do not ask the same thing in other words):
{old}
Write exactly 20 NEW questions about the entry, with short answers in one full sentence that repeat the words of the question. Every question is clear and complete by itself, in correct easy English, and has exactly one answer that is written in the entry. Ask about different facts than the ones already asked when you can; if you must reuse a fact, ask about a different part of it. Make the new questions different from each other.
- 9 "wh" questions (what, who, where, which, how, why), spread over the first, middle and last parts of the entry.
- 2 "yes" questions whose answer is "Yes, ...".
- 4 "no" questions whose answer is "No, ..." : each says something that is FALSE according to the entry (a wrong detail, a wrong place, a wrong number, a thing that is not true), and the answer corrects it with the true fact from the entry. Make them different kinds of mistakes.
- 3 "count" questions whose answer is a number written in the entry; the answer sentence contains that number.
- 2 "unk" questions: believable questions about real facts of the title that a curious child could ask, but that the entry does NOT answer anywhere (not even partly), about a different thing than the "already asked" unknown questions. They assume nothing false. The answer is exactly: I do not know.
Output only one JSON object on one line:
{{"qas": [{{"type": "wh" or "yes" or "no" or "count" or "unk", "q": "...", "a": "..."}}, ...]}}"""
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def work(r):
    out = {"title": r["title"], "domain": r["domain"]}
    old = {norm(q["q"]) for q in r["qas"]}
    for k in range(2):
        if os.path.exists(STOP): out["fail"] = "stopped"; return out
        try:
            d = Q2.j(ds.ask(PROMPT.format(title=r["title"], entry=r["entry"], old="\n".join("- " + q["q"] for q in r["qas"])), max_tokens=2500, temperature=0.7 + 0.1 * k))
        except Exception as e:
            if "COST_CAP" in str(e): raise
            d = None
        if not d or not isinstance(d.get("qas"), list): out["fail"] = "parse"; continue
        qas = [{"type": x.get("type"), "q": str(x.get("q", "")).strip(), "a": str(x.get("a", "")).strip()} for x in d["qas"] if isinstance(x, dict)]
        pick, bad = [], None
        for t, n in NEED2.items():
            got = [x for x in qas if x["type"] == t and x["q"] and x["a"] and norm(x["q"]) not in old]
            if t == "yes": got = [x for x in got if x["a"].lower().startswith("yes")]
            if t == "no": got = [x for x in got if x["a"].lower().startswith("no")]
            if t == "unk": got = [x for x in got if x["a"] == "I do not know."]
            if t == "count": got = [x for x in got if re.search(r"\b(" + NUMW + r")\b|\d", x["a"].lower())]
            if len(got) < MIN2[t]: bad = f"short:{t}"; break
            pick += got[:n]
        if not bad and len(pick) < 15: bad = "short:total"
        if bad: out["fail"] = bad; continue
        pairs = "\n".join(f"{i + 1}. Q: {x['q']} A: {x['a']}" for i, x in enumerate(pick))
        v = Q2.j("{\"v\": " + local.ask(G.VERIFY.format(entry=r["entry"], pairs=pairs), max_tokens=250, temperature=0.0) + "}")
        out["qas"] = pick; out["verdict"] = v["v"] if v and isinstance(v.get("v"), list) and len(v["v"]) == len(pick) else None
        out.pop("fail", None); return out
    return out


def main():
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(G.OUT, "full*/raw3_*.jsonl"))) for l in open(f, encoding="utf-8")]
    done = set()
    for f in glob.glob(os.path.join(OUT, "qa_*.jsonl")) + glob.glob(os.path.join(OUT, "fail_*.jsonl")):
        done |= {json.loads(l)["title"] + "|" + json.loads(l)["domain"] for l in open(f, encoding="utf-8")}
    todo = [r for r in R if r["title"] + "|" + r["domain"] not in done][: int(sys.argv[1]) if len(sys.argv) > 1 else None]
    n0 = len(glob.glob(os.path.join(OUT, "qa_*.jsonl")))
    print(f"項目 {len(R)} / 保存済み {len(done)} / これから {len(todo)}  費用上限 ${ds.COST_CAP}", flush=True)
    t0, okn, fn = time.time(), 0, 0
    for c in range(0, len(todo), CHUNK):
        if os.path.exists(STOP): break
        with ThreadPoolExecutor(8) as ex: res = [x for x in ex.map(work, todo[c:c + CHUNK]) if x.get("fail") != "stopped"]
        ok, bad = [x for x in res if "qas" in x], [x for x in res if "qas" not in x]
        k = n0 + c // CHUNK
        with open(os.path.join(OUT, f"qa_{k}.jsonl"), "w", encoding="utf-8") as f:
            for x in ok: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        with open(os.path.join(OUT, f"fail_{k}.jsonl"), "w", encoding="utf-8") as f:
            for x in bad: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        okn += len(ok); fn += len(bad)
        print(f"[保存] {k}: 完成 {okn} / 失敗 {fn} / 経過 {time.time() - t0:.0f}s / DeepSeek費用の見積もり ${ds.cost:.3f}", flush=True)
    print(f"終了: 完成 {okn} 失敗 {fn} 経過 {time.time() - t0:.0f}s 費用 ${ds.cost:.3f}", flush=True)


if __name__ == "__main__":
    main()
