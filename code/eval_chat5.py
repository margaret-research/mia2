# -*- coding: utf-8 -*-
"""会話テスト(オーナー指示 2026-10-25): 会話SFTの検証用会話(学習に使っていない)から、短い会話50本(data/chat_qa/chat_clean_val.jsonl、1〜4回の会話をそのまま)と
長い会話50本(data/chat_mix/long_val.jsonl、最初の5発言)を選ぶ。ユーザーの発言を固定で与え、Mia に貪欲生成で答えさせ(自分の返事は会話に残す)、DeepSeek が採点する。
点数 = 1回あたりの平均点(20点満点)×5 = 100点満点。短い会話・長い会話・全体で平均を出す。1回20点 = 質問・話しかけに答えている 8 / 前の発言と矛盾せず覚えている 6 / やさしく自然な英語 4 / 繰り返しや不適切がない 2。
使い方: COST_CAP=1 python eval_chat5.py ckpt.pt tag [本数=100]   (MIA_NLAYER等のenv、DEEPSEEK_API_KEY、MIA_TOK が必要)
出力: logs/chat5_<tag>.jsonl (会話と採点)、平均点を表示。生成テキストは信頼できない外部由来として扱う(採点に使うだけ)。"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
import torch
from tokenizers import Tokenizer
from model import GPT, Config
import gen_chat_qa as ds

ck, tag = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
tok = Tokenizer.from_file(os.environ.get("MIA_TOK", "data/ts_v3/tokenizer_ext.json"))
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
sd = torch.load(ck, map_location="cuda")["model"]
m = GPT(Config(vocab_size=sd["wte.weight"].shape[0], block_size=1024)).cuda(); m.load_state_dict(sd); m.eval()
ds.COST_CAP = float(os.environ.get("COST_CAP", 1.0))
def pick(f, k, cut):
    X = [json.loads(l)["turns"] for l in open(f, encoding="utf-8")]
    random.Random(11).shuffle(X); return [t[0:cut:2] for t in X[:k]]      # ユーザーの発言(長い会話は最初の5つ)
FROZEN = "benchmark/chat5_set.json"      # 基準テスト: 凍結した100本(短50+長50)。あれば必ずこれを使う
if os.path.exists(FROZEN):
    fz = json.load(open(FROZEN, encoding="utf-8")); L = fz["short"] + fz["long"]; N = len(L)
else:
    L = pick("data/chat_qa/chat_clean_val.jsonl", N // 2, 10) + pick("data/chat_mix/long_val.jsonl", N - N // 2, 10)
NS = N // 2

@torch.no_grad()
def talk(users):
    ids, out = [BOS], []
    for u in users:
        ids += [USER] + tok.encode(u).ids + [BOT]; new = []
        for _ in range(80):
            lg, _ = m(torch.tensor([ids[-1024:]], device="cuda")); nx = int(lg[0, -1].argmax())
            if nx in (EOS, USER): break
            ids.append(nx); new.append(nx)
        ids.append(EOS); out.append(tok.decode(new).strip())
    return out

PROMPT = """You are grading a small chat helper named Mia (a tiny language model made for children). Below are {n} exchanges; the USER lines are fixed, the MIA lines are Mia's replies.
Calibration: the target level of 90/100 is the conversation ability of a typical child in grade 1-2 (6 to 8 years old). Score each reply the way you would score what a typical 7-year-old would say.
A typical 7-year-old: stays on the topic, answers a simple question directly in a short sentence, remembers names and things said a few turns ago, says "I do not know" or asks back instead of inventing things, uses short easy words, does not repeat itself, and is kind. A 7-year-old does NOT give long or precise explanations, so do not penalize missing detail or small factual slips that a child could make.
A typical 7-year-old gets about 18 of 20 per reply (answer 7, memory 5, language 4, clean 2). Full marks (20) need the extra: a reason or care in the answer, and a natural use of an earlier fact. Do NOT give high marks just because the sentences are grammatical or polite.
Score EACH of the {n} replies:
- "answer" 0-8: responds to what the user just said or asked. Typical child = 6-7. 8 only if it also adds a reason or care. 0-2 if it ignores the question or is off topic.
- "memory" 0-6: consistent with earlier turns (names, facts the user or Mia said). Typical child = 5 (also use 5 when nothing earlier is needed). 6 if it naturally uses an earlier fact. 0-2 if it mixes up names, forgets, or contradicts.
- "language" 0-4: easy, short, understandable English. Typical child = 4 (use 3 if a little awkward). 0-1 if sentences do not make sense.
- "clean" 0-2: no repeated words or sentences, nothing unsuitable for a child. Typical child = 2. 0 if repeated or unsuitable.
Also give three flags for each reply: "ignored" (true if it ignores or dodges the user's question or request), "contradiction" (true if it contradicts or mixes up an earlier turn), "unsuitable" (true if it gives advice or content that is bad for a child).
{conv}
Output only one JSON object on one line: {{"turns": [{{"answer": 0, "memory": 0, "language": 0, "clean": 0, "ignored": false, "contradiction": false, "unsuitable": false}}, ... {n} items ...], "note": "<=20 words on the main problem"}}"""

def one(x):          # 上限ルール: 無視は最大10点 / 矛盾・取り違えは最大12点 / 不適切な助言は最大8点
    v = min(8, max(0, int(x["answer"]))) + min(6, max(0, int(x["memory"]))) + min(4, max(0, int(x["language"]))) + min(2, max(0, int(x["clean"])))
    if x.get("ignored"): v = min(v, 10)
    if x.get("contradiction"): v = min(v, 12)
    if x.get("unsuitable"): v = min(v, 8)
    return v


def judge(users, mia):
    n = len(users)
    conv = "\n".join(f"USER: {u}\nMIA: {a}" for u, a in zip(users, mia))
    for _ in range(3):
        t = ds.ask(PROMPT.format(conv=conv, n=n), max_tokens=900, temperature=0.0)
        mm = re.search(r"\{.*\}", t, re.S)
        try:
            d = json.loads(mm.group(0)); ts = d["turns"]; assert len(ts) == n
            sc = [one(x) for x in ts]
            return sc, d.get("note", "")
        except Exception:
            continue
    return None, "judge-fail"

res = [(u, talk(u)) for u in L]
with ThreadPoolExecutor(8) as ex:
    js = list(ex.map(lambda r: judge(*r), res))
rows, tot, grp = [], [], {"短い": [], "長い": []}
for k, ((u, a), (sc, note)) in enumerate(zip(res, js)):
    rows.append({"users": u, "mia": a, "scores": sc, "note": note})
    if sc:
        x = 5 * sum(sc) / len(sc); tot.append(x); grp["短い" if k < NS else "長い"].append(x)
os.makedirs("logs", exist_ok=True)
with open(f"logs/chat5_{tag}.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{tag}: 平均 {sum(tot) / len(tot):.1f} / 100点 (採点できた {len(tot)}/{len(rows)}本)  " + "  ".join(f"{g}会話 {sum(v) / len(v):.1f} ({len(v)}本)" for g, v in grp.items()) + f"  費用 ${ds.cost:.3f}")
print("80点以上の本数:", sum(t >= 80 for t in tot), " 60点未満:", sum(t < 60 for t in tot))
