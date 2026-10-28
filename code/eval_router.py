# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — ルーターMiaの凍結テスト (data/router/test.jsonl)。問題と採点基準は変えない。
 判断 = calc/chat の一致 / 式 = 式の文字列が完全一致 / 答え = 出した式を電卓で解いた値が正解と一致 (a*b と b*a は可)
 chat_num・chat_plain の誤呼び出し率も出す。使い方: python eval_router.py ckpt/router_mia_v1.pt"""
import collections, json, os, sys
import torch
from tokenizers import Tokenizer
from model import GPT, Config
from calc_tool import solve
tok = Tokenizer.from_file(os.environ["MIA_TOK"])
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
m = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=1024)).cuda()
m.load_state_dict(torch.load(sys.argv[1])["model"]); m.eval()
@torch.no_grad()
def route(r):
    ids = [BOS]
    for i, t in enumerate(r["hist"]):
        ids += ([USER] + tok.encode(t).ids) if i % 2 == 0 else ([BOT] + tok.encode(t).ids + [EOS])
    ids += [USER] + tok.encode(r["user"]).ids + [BOT]
    n0 = len(ids)
    for _ in range(40):
        lg, _ = m(torch.tensor([ids], device="cuda")); nx = int(lg[0, -1].argmax())
        if nx in (EOS, USER): break
        ids.append(nx)
    return tok.decode(ids[n0:]).strip()
rows = [json.loads(l) for l in open(os.environ.get("TEST", "data/router/test.jsonl"))]
st = collections.defaultdict(lambda: collections.Counter()); bad = []
kind = lambda x: x.split()[0].lower() if x.split() else ""
for r in rows:
    o = route(r); wk = kind(r["target"]); gk = kind(o)
    s = st[r["type"]]; s["n"] += 1; s["judge"] += (gk == wk)
    if wk == "calc":
        e = r["target"][5:]; s["expr"] += (o[5:].strip() == e)
        ok = gk == "calc" and solve(o[5:].strip()) == r["result"]; s["answer"] += ok
        if not ok and len(bad) < 12: bad.append((r["type"], r["user"], r["target"], o))
    elif wk == "card":
        ok = gk == "card" and o[4:].strip().lower() == r["target"][5:].strip().lower(); s["expr"] += ok; s["answer"] += ok
        if not ok and len(bad) < 24: bad.append((r["type"], r["hist"][-2:], r["user"], r["target"], o))
    elif gk != "chat" and len(bad) < 24: bad.append((r["type"], r["user"], r["target"], o))
print(f"{'型':14}{'件数':>6}{'判断':>7}{'式/題名':>8}{'答え':>7}")
tc = collections.Counter()
for t, s in sorted(st.items()):
    c = t not in ("chat_num", "chat_plain", "card_neg")
    print(f"{t:14}{s['n']:6}{s['judge']/s['n']:7.2f}" + (f"{s['expr']/s['n']:8.2f}{s['answer']/s['n']:7.2f}" if c else "   (誤呼び出し率 %.3f)" % (1 - s['judge']/s['n'])))
    k = ("card" if t.startswith("card") else "calc") if c else "chat"; tc[k + "_n"] += s["n"]; tc[k + "_judge"] += s["judge"]; tc[k + "_expr"] += s["expr"]; tc[k + "_ans"] += s["answer"]
print(f"計算 {tc['calc_n']}件: 判断 {tc['calc_judge']/tc['calc_n']:.3f} 式 {tc['calc_expr']/tc['calc_n']:.3f} 答え {tc['calc_ans']/tc['calc_n']:.3f}")
if tc["card_n"]: print(f"カード {tc['card_n']}件: 判断 {tc['card_judge']/tc['card_n']:.3f} 題名一致 {tc['card_expr']/tc['card_n']:.3f}")
print(f"会話 {tc['chat_n']}件: 誤呼び出し率 {1 - tc['chat_judge']/tc['chat_n']:.4f}")
for b in bad: print(b)
