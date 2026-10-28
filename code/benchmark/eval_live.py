# -*- coding: utf-8 -*-
"""実機会話テスト (基準テスト5。オーナー指示 2026-10-28): ボット全体 (ルーター → 電卓/カード検索/会話) に、固定の複数ターンの場面を通し、DeepSeek が採点する。
場面は benchmark/live_set.json (24場面・計120発言。オーナーの実機ログと、計算/カード/雑談/記憶/崩れた英語の場面)。ユーザーの発言は固定、返事は貪欲に近い再現性のため乱数を場面ごとに固定。
1回20点 = 質問に答え内容が正しい 8 / 前の発言と矛盾せず覚えている 6 / やさしく自然な英語 4 / 繰り返しや不適切がない 2。点数 = 1回の平均×5 (100点満点)。基準は小学校低学年 (7歳) が約90点、合格ライン 80点。
上限ルール: 事実や数が間違い・作り話は最大8点 / 無視は最大10点 / 矛盾は最大12点 / 不適切は最大8点。
使い方 (bash -ic で DEEPSEEK_API_KEY が見える形で): COST_CAP=1 MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 MIA_TOK=data/ts_v4/tokenizer_ext.json PYTHONPATH=. python benchmark/eval_live.py <ルーターckpt> <タグ>
出力: logs/live_<タグ>.jsonl (発言・経路・返事・点)。Mia が出した文は信頼できない外部由来のテキストとして採点にだけ使う。"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import torch
import gen_chat_qa as ds
from bot_mia2 import Bot

router, tag = sys.argv[1], sys.argv[2]
ds.COST_CAP = float(os.environ.get("COST_CAP", 1.0))
S = json.load(open("benchmark/live_set.json", encoding="utf-8"))["scenarios"]
bot = Bot(router=router, chat=os.environ.get("CHAT", "ckpt/mia_current_read.pt"))

PROMPT = """You are grading a small chat helper named Mia (a tiny language model made for children). Below are {n} exchanges; the USER lines are fixed, the MIA lines are Mia's replies.
Mia has a calculator and a small book of facts, so she should give correct numbers and facts when she answers; when she does not know she may say "I do not know." (that is honest, but it is not a good answer to a simple question a 7-year-old can answer).
Calibration: the target level of 90/100 is the conversation ability of a typical child in grade 1-2 (6 to 8 years old). Score each reply the way you would score what a typical 7-year-old would say.
A typical 7-year-old: stays on the topic, answers a simple question directly in a short sentence, remembers names and things said a few turns ago, does not invent things, uses short easy words, does not repeat itself, and is kind. A 7-year-old does NOT give long or precise explanations, so do not penalize missing detail. But a wrong number or a clearly wrong fact (for example "the moon is a planet", "5 plus 2 is the area of 5 by 2") is an error even for a child.
A typical 7-year-old gets about 18 of 20 per reply (answer 7, memory 5, language 4, clean 2). Full marks (20) need the extra: a reason or care in the answer, and a natural use of an earlier fact. Do NOT give high marks just because the sentences are grammatical or polite.
Score EACH of the {n} replies:
- "answer" 0-8: responds to what the user just said or asked, and what it says is correct. Typical child = 6-7. 8 only if it also adds a reason or care. 0-2 if it ignores the question, is off topic, or repeats the question back.
- "memory" 0-6: consistent with earlier turns (names, facts, numbers). Typical child = 5 (also use 5 when nothing earlier is needed). 6 if it naturally uses an earlier fact. 0-2 if it mixes up names, forgets what "it" or "they" means, or contradicts.
- "language" 0-4: easy, short, understandable English. Typical child = 4 (use 3 if a little awkward). 0-1 if sentences do not make sense.
- "clean" 0-2: no repeated words or sentences, nothing unsuitable for a child. Typical child = 2. 0 if repeated or unsuitable.
Also give four flags for each reply: "ignored" (ignores or dodges the user's question or request), "contradiction" (contradicts or mixes up an earlier turn), "unsuitable" (content that is bad for a child), "wrong" (a wrong number, a wrong fact, or an invented fact).
{conv}
Output only one JSON object on one line: {{"turns": [{{"answer": 0, "memory": 0, "language": 0, "clean": 0, "ignored": false, "contradiction": false, "unsuitable": false, "wrong": false}}, ... {n} items ...], "note": "<=25 words on the main problem"}}"""


def one(x):
    v = min(8, max(0, int(x["answer"]))) + min(6, max(0, int(x["memory"]))) + min(4, max(0, int(x["language"]))) + min(2, max(0, int(x["clean"])))
    if x.get("wrong"): v = min(v, 8)
    if x.get("ignored"): v = min(v, 10)
    if x.get("contradiction"): v = min(v, 12)
    if x.get("unsuitable"): v = min(v, 8)
    return v


def judge(users, mia):
    conv = "\n".join(f"USER: {u}\nMIA: {a}" for u, a in zip(users, mia)); n = len(users)
    for _ in range(3):
        t = ds.ask(PROMPT.format(conv=conv, n=n), max_tokens=1500, temperature=0.0)
        mm = re.search(r"\{.*\}", t, re.S)
        try:
            d = json.loads(mm.group(0)); ts = d["turns"]; assert len(ts) == n
            return [one(x) for x in ts], ts, d.get("note", "")
        except Exception:
            continue
    return None, None, "judge-fail"


runs = []
for name, users in S.items():
    torch.manual_seed(0); bot.turns = []
    ways, reps = [], []
    for u in users:
        w, r = bot.say(u); ways.append(w); reps.append(r)
    runs.append((name, users, ways, reps))
with ThreadPoolExecutor(8) as ex:
    js = list(ex.map(lambda r: judge(r[1], r[3]), runs))
rows, allv, per = [], [], {}
for (name, users, ways, reps), (sc, flags, note) in zip(runs, js):
    rows.append({"scenario": name, "users": users, "route": ways, "mia": reps, "scores": sc, "flags": flags, "note": note})
    if sc:
        per[name] = 5 * sum(sc) / len(sc); allv += [5 * s for s in sc]
os.makedirs("logs", exist_ok=True)
with open(f"logs/live_{tag}.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for k, v in sorted(per.items(), key=lambda x: x[1]): print(f"  {k:20s} {v:5.1f}")
fl = [x for r in rows if r["flags"] for x in r["flags"]]
print(f"{tag}: 平均 {sum(allv) / len(allv):.1f} / 100点 (採点できた {len(per)}/{len(rows)}場面, {len(allv)}発言)  合格ライン 80  費用 ${ds.cost:.3f}")
print("80点以上の発言:", sum(v >= 80 for v in allv), " 60点未満:", sum(v < 60 for v in allv), " 間違い/作り話:", sum(bool(x.get('wrong')) for x in fl), " 無視:", sum(bool(x.get('ignored')) for x in fl), " 矛盾:", sum(bool(x.get('contradiction')) for x in fl))
