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

PROMPT = """You are a strict but fair teacher grading a children's chat helper named Mia (a small language model). Below are {n} exchanges; the USER lines are fixed, the MIA lines are Mia's replies.
Mia should talk like a kind, simple, correct helper for children in grade 1-2. Score EACH of the {n} replies:
- "answer" 0-8: the reply really answers or responds to what the user just said or asked (8 = fully and helpfully; 4 = partly or vague; 0 = off topic or nonsense)
- "memory" 0-6: consistent with the earlier turns and what the user said before (names, facts); no contradictions or wrong claims; 6 if nothing earlier is needed and nothing is wrong
- "language" 0-4: easy, natural, correct English, complete sentences (0 = broken)
- "clean" 0-2: no repeating the same words or sentences, nothing odd or unsuitable for a child
{conv}
Output only one JSON object on one line: {{"turns": [{{"answer": 0, "memory": 0, "language": 0, "clean": 0}}, ... {n} items ...], "note": "<=20 words on the main problem"}}"""

def judge(users, mia):
    n = len(users)
    conv = "\n".join(f"USER: {u}\nMIA: {a}" for u, a in zip(users, mia))
    for _ in range(3):
        t = ds.ask(PROMPT.format(conv=conv, n=n), max_tokens=500, temperature=0.0)
        mm = re.search(r"\{.*\}", t, re.S)
        try:
            d = json.loads(mm.group(0)); ts = d["turns"]; assert len(ts) == n
            sc = [min(8, max(0, int(x["answer"]))) + min(6, max(0, int(x["memory"]))) + min(4, max(0, int(x["language"]))) + min(2, max(0, int(x["clean"]))) for x in ts]
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
