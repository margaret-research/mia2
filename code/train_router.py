# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-06) — T-025 方式C: ルーター頭の訓練と評価 (LLMは凍結)
使い方: python train_router.py            (特徴量抽出→訓練→評価。特徴量は ckpt/router_feats.pt にキャッシュ)
        python train_router.py --refit    (特徴量キャッシュを使って頭だけ再訓練)
"""
import json, random, sys, time, collections
import torch
import torch.nn.functional as F
from fractions import Fraction
from calc_tool import find_expression, solve
from router import (OPS, Heads, load_backbone, build_context, hidden, number_spans, decode, op_base, op_logits)

random.seed(0)
torch.manual_seed(0)
dev = "cpu"   # GPUはT-024(CoAtNet)が使用中。凍結LLMの順伝播のみなのでCPUで足りる
torch.set_num_threads(8)
NONCALC = ("neg", "negh", "plain")


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def label_call(final_text, expr):
    """式 expr が final_text の どの数 を どの順で使うか。返り値 (op添字, 1つ目の数の区間, 2つ目の数の区間) / None(判定不能)。"""
    toks, _ = find_expression(expr)
    if not toks or len(toks) != 3:
        return None
    x, op, y = Fraction(toks[0].replace(",", "")), toks[1], Fraction(toks[2].replace(",", ""))
    sp = number_spans(final_text)
    for rev in (False, True):
        a, b = (y, x) if rev else (x, y)
        for i in range(len(sp)):
            for j in range(i + 1, len(sp)):
                if sp[i][2] == a and sp[j][2] == b:
                    if op == "-":
                        k = "sub_rev" if rev else "sub"
                    elif op == "/":
                        if rev:
                            return None
                        k = "div"
                    else:
                        k = "add" if op == "+" else "mul"
                    return OPS.index(k), sp[i][:2], sp[j][:2]
    return None


def make_examples(calc_rows, plain_rows, n_plain):
    ex, dropped = [], collections.Counter()
    for r in calc_rows:
        t, cat = r["turns"], r["cat"]
        if r["calls"]:
            ctx, want = t[:-1], list(r["calls"].values())[0]
            lab = label_call(ctx[-1], want["expr"])
            if lab is None:
                dropped[cat] += 1
                continue
            ex.append(dict(ctx=ctx, op=lab[0], s1=lab[1], s2=lab[2], cat=cat, want=want["result"]))
        else:
            ctx = t[:1] if cat == "neg" else t[:-1]
            ex.append(dict(ctx=ctx, op=0, s1=None, s2=None, cat=cat, want=None))
    for _ in range(n_plain):          # 普通の会話の途中 (計算ではない発言)
        c = random.choice(plain_rows)["turns"]
        k = random.randrange(0, len(c), 2)
        ex.append(dict(ctx=c[:k + 1], op=0, s1=None, s2=None, cat="plain", want=None))
    return ex, dropped


@torch.no_grad()
def extract(tok, model, ex, bs=48, verbose=True):
    """各例について 最後の位置の隠れ状態(op用)と 最後の発言トークンの隠れ状態+タグ(tag用)を取り出す。"""
    ctxs = [build_context(tok, e["ctx"]) for e in ex]
    order = sorted(range(len(ex)), key=lambda i: len(ctxs[i][0]))
    feats = [None] * len(ex)
    t0 = time.time()
    for bi in range(0, len(order), bs):
        idxs = order[bi:bi + bs]
        L = max(len(ctxs[i][0]) for i in idxs)
        x = torch.zeros(len(idxs), L, dtype=torch.long)
        for r, i in enumerate(idxs):
            x[r, :len(ctxs[i][0])] = torch.tensor(ctxs[i][0])
        h = hidden(model, x)
        for r, i in enumerate(idxs):
            ids, start, offs, text = ctxs[i]
            e = ex[i]
            tags = torch.zeros(len(offs), dtype=torch.long)
            for tid, sp in ((1, e["s1"]), (2, e["s2"])):
                if sp:
                    for k, (a, b) in enumerate(offs):
                        if a < sp[1] and b > sp[0]:
                            tags[k] = tid
            emb = model.wte(torch.tensor(ids[start:start + len(offs)]))
            feats[i] = dict(uid=torch.tensor(ids[start:start + len(offs)]), last=h[r, len(ids) - 1].half(), utt=h[r, start:start + len(offs)].half(), emb=emb.half(), tags=tags,
                            offs=offs, text=text)
        if verbose and (bi // bs) % 100 == 0:
            print(f"  特徴量 {bi}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    return feats


def fit(feats_tr, ex_tr, epochs=40):
    heads = Heads()
    Xo = torch.stack([op_base(f["last"].float(), f["utt"].float(), f["emb"].float()) for f in feats_tr])
    Uo = [f["uid"] for f in feats_tr]
    yo = torch.tensor([e["op"] for e in ex_tr])
    Xt = torch.cat([f["utt"] for f in feats_tr]).float()
    yt = torch.cat([f["tags"] for f in feats_tr])
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"訓練: op {len(yo)} 件 / tag {len(yt)} トークン (O:{(yt == 0).sum().item()} N1:{(yt == 1).sum().item()} N2:{(yt == 2).sum().item()})")
    for ep in range(epochs):
        po, pt = torch.randperm(len(yo)), torch.randperm(len(yt))
        lo = lt = 0.0
        for b in range(0, len(yo), 256):
            ix = po[b:b + 256]
            loss = F.cross_entropy(op_logits(heads, Xo[ix], [Uo[i] for i in ix.tolist()]), yo[ix])
            opt.zero_grad(); loss.backward(); opt.step(); lo += loss.item() * len(ix)
        for b in range(0, len(yt), 1024):
            ix = pt[b:b + 1024]
            loss = F.cross_entropy(heads.tag(Xt[ix]), yt[ix])
            opt.zero_grad(); loss.backward(); opt.step(); lt += loss.item() * len(ix)
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep}: op loss {lo / len(yo):.4f} tag loss {lt / len(yt):.4f}", flush=True)
    return heads.eval()


@torch.no_grad()
def predict(heads, f):
    base = op_base(f["last"].float(), f["utt"].float(), f["emb"].float())[None]
    op = op_logits(heads, base, [f["uid"]]).argmax().item()
    tags = heads.tag(f["utt"].float()).argmax(-1).tolist()
    return op, decode(op, tags, f["offs"], f["text"])


def evaluate(heads, feats, ex, title):
    print(f"\n===== {title} =====")
    by = collections.defaultdict(lambda: [0, 0, 0, 0])   # cat -> [n, 答え正解, op正解, 呼ばない正解]
    bylen = collections.defaultdict(lambda: [0, 0])
    shown = 0
    for f, e in zip(feats, ex):
        op, expr = predict(heads, f)
        s = by[e["cat"]]
        s[0] += 1
        s[2] += (op == e["op"])
        if e["want"] is not None:
            ok = expr not in (None, "missing") and solve(expr) == e["want"]
            s[1] += ok
            L = bylen[len(e["ctx"])]; L[0] += 1; L[1] += ok
            if not ok and shown < 10:
                shown += 1
                print("NG", e["cat"], "|", f["text"][:70], "| want", e["want"], "| op", OPS[op], "| expr", expr)
        else:
            s[3] += (expr is None or expr == "missing")
    for c, (n, a, o, z) in sorted(by.items()):
        if c in NONCALC:
            print(f"{c:13s} 呼ばない正解 {z}/{n} = {z / n:.1%}  (誤呼び出し {n - z})")
        else:
            print(f"{c:13s} 答え正解 {a}/{n} = {a / n:.1%}   (op正解 {o / n:.1%})")
    tot = sum(v[0] for k, v in by.items() if k not in NONCALC)
    cor = sum(v[1] for k, v in by.items() if k not in NONCALC)
    print(f"計算ありの合計: {cor}/{tot} = {cor / max(tot, 1):.1%}")
    print("履歴の長さ別(ctxターン数): " + "  ".join(f"{k}:{v[1]}/{v[0]}" for k, v in sorted(bylen.items())))


def hand_test(heads, tok, model, vals, nohist=False):
    """手入力の失敗例を、本物の対話の履歴 (0〜2往復) の後で試す。Colabの表と同じ入力。"""
    QA = [("What is twelve plus five?", "17"), ("What is 144 divided by 12?", "12"), ("What is 7 x 8?", "56"),
          ("What is 100 divided by 3?", "about 33.3333"),
          ("I have 14 cookies. I give 6 to my friend. How many cookies do I have left?", "8"),
          ("What is 10 minus 3?", "7"), ("what will be 10 minus 3?", "7"), ("What is 10 - 3?", "7"),
          ("Subtract 3 from 10.", "7"), ("What is 10 take away 3?", "7"), ("I will stay for 3-4 days.", None),
          ("Hi! How are you today?", None)]
    print("\n===== 手入力の失敗例 (本物の対話の履歴0〜2往復、各20回) =====")
    for q, want in QA:
        ok = 0
        for _ in range(20):
            base = random.choice(vals)
            k = random.randrange(0, 3)
            ex1 = [dict(ctx=([q] if nohist else base[:2 * k] + [q]), op=0, s1=None, s2=None, cat="hand", want=want)]
            op, expr = predict(heads, extract(tok, model, ex1, verbose=False)[0])
            if want is None:
                ok += expr in (None, "missing")
            else:
                ok += expr not in (None, "missing") and solve(expr) == want
        print(f"{q[:48]:48s} {ok}/20")


if __name__ == "__main__":
    NOHIST = "--nohist" in sys.argv
    SUF = "_nohist" if NOHIST else ""
    print("履歴なし(最後の発言だけで判断)" if NOHIST else "履歴あり")
    tok, model = load_backbone(dev)
    train = read("data/calc_train.jsonl")
    test = read("data/calc_test.jsonl")
    plain_tr = read("data/dialogues_train_expanded.jsonl")
    plain_va = read("data/dialogues_val.jsonl")
    ex_tr, d1 = make_examples(train, plain_tr, 12000)
    ex_te, d2 = make_examples(test, plain_va, 600)
    print("ラベル付け不能で除外: train", dict(d1), " test", dict(d2))
    if NOHIST:
        for e in ex_tr + ex_te:
            e["ctx"] = e["ctx"][-1:]
    cache = f"ckpt/router_feats{SUF}.pt"
    if "--refit" in sys.argv:
        feats_tr, feats_te = torch.load(cache)
        for f, e in zip(feats_tr + feats_te, ex_tr + ex_te):   # 旧キャッシュにはトークンidが無いので本文から作る
            if "uid" not in f:
                f["uid"] = torch.tensor(tok.encode(f["text"]).ids)
    else:
        print("特徴量抽出 (train)", len(ex_tr)); feats_tr = extract(tok, model, ex_tr)
        print("特徴量抽出 (test)", len(ex_te)); feats_te = extract(tok, model, ex_te)
        torch.save((feats_tr, feats_te), cache)
    heads = fit(feats_tr, ex_tr)
    torch.save(heads.state_dict(), f"ckpt/router_heads{SUF}.pt")
    evaluate(heads, feats_tr[:3000], ex_tr[:3000], "訓練データの一部 (参考。汎化の指標ではない)")
    evaluate(heads, feats_te, ex_te, "テスト (calc_test + 普通の会話val)")
    hand_test(heads, tok, model, [r["turns"] for r in plain_va if len(r["turns"]) >= 4], NOHIST)
