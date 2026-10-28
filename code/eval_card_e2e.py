# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — カード拡張の全体テスト: 読解テストの50項目(931問)を、ルーターMia → 題名索引 → 項目 → 現行Miaの読解、の順に通す。
 題名はルーターの学習から外してある(新しい題材への一般化)。比較対象: 項目を最初から渡す読解テスト(eval_read.py)=80%。
 使い方: python eval_card_e2e.py ckpt/router_mia_v4.pt [読解ckpt=ckpt/mia_current_read.pt]"""
import collections, json, os, re, sys
import torch
from tokenizers import Tokenizer
from model import GPT, Config
from card_retrieve import Index, EmbIndex
tok = Tokenizer.from_file(os.environ["MIA_TOK"])
BOS, EOS, USER, BOT = (tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
def load(p):
    m = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=1024)).cuda(); m.load_state_dict(torch.load(p)["model"]); return m.eval()
router, reader = load(sys.argv[1]), load(sys.argv[2] if len(sys.argv) > 2 else "ckpt/mia_current_read.pt")
@torch.no_grad()
def gen(m, ids, n=60):
    n0 = len(ids)
    for _ in range(n):
        lg, _ = m(torch.tensor([ids], device="cuda")); nx = int(lg[0, -1].argmax())
        if nx in (EOS, USER): break
        ids.append(nx)
    return tok.decode(ids[n0:]).strip()
STOP = set("the and are was were that this with for from have has does did not what how who which where why".split())
words = lambda s: {w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) >= 3 and w not in STOP}
dump = []
RET = os.environ.get('RET', 'name')       # name=題名索引 / emb=本文の埋め込み+題名 上位1枚 / emb3=上位3枚を順に読み、IDKでない最初の答え
idx = EmbIndex() if RET != 'name' else Index(); E = json.load(open("benchmark/read_val_entries.json"))
hit, tot, route_ok, kinds, bad = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), []
for e in E:
    for q in e["qs"]:
        t = q["type"]; tot[t] += 1
        o = gen(router, [BOS, USER] + tok.encode(q["q"]).ids + [BOT], 40)
        kinds[o.split()[0] if o.split() else ""] += 1; dump.append({"q": q["q"], "gold": e["title"], "routed": o})
        routed = o[4:].strip() if o.startswith("card") else None
        if RET == "name": cands = [idx.find(routed, q["q"])] if routed else []
        elif routed is None: cands = []
        else: cands = idx.top(routed, q["q"], 1 if RET == "emb" else 3)
        cands = [c for c in cands if c]
        title = cands[0] if cands else None
        route_ok[t] += (title == e["title"])
        a = "I do not know."
        for c in cands:
            a = gen(reader, [BOS, USER] + tok.encode(f"Title: {c}. {idx.ents[c]['entry']} Question: {q['q']}").ids + [BOT])
            if a != "I do not know.": break
        if t == "unk": ok = a == "I do not know."
        else:
            g = words(q["a"]); ok = a != "I do not know." and len(g & words(a)) >= 0.6 * len(g)
            if t in ("yes", "no"): ok = ok and a.lower().startswith(q["a"].lower().split(",")[0])
        hit[t] += ok
        if not ok and title != e["title"] and len(bad) < 14: bad.append((q["q"], e["title"], o, title))
n = sum(tot.values())
print("ルーターの判断:", dict(kinds))
for t in tot: print(f"{t}: 正解 {hit[t]}/{tot[t]} = {hit[t]/tot[t]:.2f} / 正しい項目を引けた率 {route_ok[t]/tot[t]:.2f}")
print(f"全体: 正解 {sum(hit.values())}/{n} = {sum(hit.values())/n:.3f} / 正しい項目を引けた率 {sum(route_ok.values())/n:.3f}   (項目を最初から渡した場合 = 0.80)")
for b in bad: print(b)

json.dump(dump, open("logs/card_routed.json", "w"), ensure_ascii=False)

# 項目にない話題の質問 (card2_test の card_nokb): ルーター → 検索 → 読解。"I do not know." で答えられた率 (高いほどよい)
nk = [json.loads(l) for l in open("data/router/card2_test.jsonl") if '"card_nokb"' in l]
idk = rej = 0
for r in nk:
    o = gen(router, [BOS, USER] + tok.encode(r["user"]).ids + [BOT], 40)
    c = (idx.top(o[4:].strip(), r["user"], 1) if o.split()[:1] == ["card"] and RET != "name" else [])
    if not c: rej += 1; idk += 1; continue
    a = gen(reader, [BOS, USER] + tok.encode(f"Title: {c[0]}. {idx.ents[c[0]]['entry']} Question: {r['user']}").ids + [BOT])
    idk += a == "I do not know."
print(f"項目にない話題 {len(nk)}問: 検索で項目なし {rej/len(nk):.3f} / I do not know. で答えた率 {idk/len(nk):.3f}")
