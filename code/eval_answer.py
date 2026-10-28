# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — 答える側の凍結テスト: 正しい式と結果を渡したとき、自然な文で正しい数を答えるか
対象: data/router/test_v2aug.jsonl の計算行 (履歴つき) から固定シードで800件。
 正解 = 結果の数をちょうど1回言う / 質問・履歴にない数を言わない / 25語以内 / 質問を返さない。
 固定文率 = "The answer is X." 型の割合 (低いほど自然)。使い方: python eval_answer.py ckpt/answer_mia_v1.pt"""
import json, os, random, re, sys, collections
import torch
from tokenizers import Tokenizer
from model import GPT, Config
from gen_router_para import nums_of
tok = Tokenizer.from_file(os.environ["MIA_TOK"])
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
m = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=1024)).cuda()
m.load_state_dict(torch.load(sys.argv[1])["model"]); m.eval()
@torch.no_grad()
def answer(r):
    ids = [BOS]
    for i, t in enumerate(r["hist"]):
        ids += ([USER] + tok.encode(t).ids) if i % 2 == 0 else ([BOT] + tok.encode(t).ids + [EOS])
    ids += [USER] + tok.encode(f"Facts: {r['target'][5:]} = {r['result']}. Question: {r['user']}").ids + [BOT]
    n0 = len(ids)
    for _ in range(50):
        lg, _ = m(torch.tensor([ids], device="cuda")); nx = int(lg[0, -1].argmax())
        if nx in (EOS, USER): break
        ids.append(nx)
    return tok.decode(ids[n0:]).strip()
rows = [json.loads(l) for l in open("data/router/test_v2aug.jsonl") if '"CALC' in l]
random.seed(0); random.shuffle(rows); rows = rows[:800]
st, fixed, bad, samples = collections.Counter(), 0, [], []
for r in rows:
    a = answer(r); want = r["result"].replace(",", "")
    allowed = {k.replace(",", "") for k in nums_of(" ".join(r["hist"] + [r["user"]]))}
    got = collections.Counter(k.replace(',', '') for k in nums_of(a).elements())
    ok = got.get(want, 0) == 1 and all(k == want or k in allowed for k in got) and len(a.split()) <= 25 and "?" not in a
    st[r["type"]] += 1; st["ok_" + r["type"]] += ok; st["ok"] += ok
    fixed += bool(re.fullmatch(r"(The answer is|It is|That is) [\d,]+\.", a))
    if not ok and len(bad) < 8: bad.append((r["user"], r["result"], a))
    if len(samples) < 14 and r["type"] != "expr" or (r["type"] == "expr" and len(samples) < 14 and random.random() < .1): samples.append((r["user"][:80], r["result"], a))
print(f"答え正解 {st['ok']}/{len(rows)} = {st['ok']/len(rows):.3f} / 固定文率 {fixed/len(rows):.3f}")
for t in sorted(k for k in st if not k.startswith("ok")): print(f"  {t:12} {st['ok_'+t]}/{st[t]}")
print("--- 見本"); [print("  ", s) for s in samples]
print("--- 失敗"); [print("  ", b) for b in bad]
