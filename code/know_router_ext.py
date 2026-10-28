# (新トークナイザー版 2026-10-12: know_router.py から router_ext・会話データ(新データ)・保存名だけ変更)
# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — T-026 会話Bot: 「知識の質問 / 普通の会話」の判定器

凍結した会話LLM (ckpt/sft_5k_v2.pt) の隠れ状態の上に、小さな2クラスの頭だけを載せて訓練する (計算ルーター router.py と同じ方式)。
  入力: 最後の位置(<bot>) + 発言トークンの max / mean プーリング (履歴は見ない)
  知識(1): 読解データの質問 (カード付きの質問から質問文だけ取り出したもの) + 「Tell me about X」型
  会話(0): 会話データ (dialogues_train_expanded) のユーザー発言
訓練: python know_router.py     -> ckpt/know_head.pt と精度を出力
"""
import json
import random
import re

import torch
import torch.nn as nn

from router_ext import build_context, hidden, load_backbone

HEAD_PATH = "ckpt/know_head_ext.pt"
D = "data/reading_qa/"
TEMPLATES = ["What is a {s}?", "What is {s}?", "Tell me about {s}.", "Tell me about the {s}.", "Can you tell me about {s}?",
             "What do you know about {s}?", "what is a {s}", "Do you know what a {s} is?", "What are {s}s?", "What does a {s} do?"]


def features(tok, model, text, dev="cpu"):
    ids, start, offs, _ = build_context(tok, [text])
    h = hidden(model, torch.tensor([ids], device=dev))[0]
    utt = h[start:start + len(offs)]
    return torch.cat([h[-1], utt.max(0).values, utt.mean(0)])


def make_head(d=384):
    return nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 2))


class KnowRouter:
    def __init__(self, tok, model, dev="cpu", path=HEAD_PATH):
        self.tok, self.model, self.dev = tok, model, dev
        self.head = make_head().to(dev)
        self.head.load_state_dict(torch.load(path, map_location=dev))
        self.head.eval()

    @torch.no_grad()
    def prob(self, text):
        """知識の質問である確率"""
        f = features(self.tok, self.model, text, self.dev)
        return torch.softmax(self.head(f[None]), -1)[0, 1].item()


def load_data(seed=0):
    rnd = random.Random(seed)
    held = {json.loads(l)["subject"] for l in open(D + "s10k_test_newsubj.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for f in ("raw2_s10k", "raw2_s20k", "raw2_s100k") for l in open(D + f + ".jsonl", encoding="utf-8")]
    rows = [r for r in rows if not r["named"]]
    tr_q = [r["q"] for r in rows if r["subject"] not in held]
    te_q = [r["q"] for r in rows if r["subject"] in held]
    subs_tr = sorted({r["subject"] for r in rows if r["subject"] not in held})
    subs_te = sorted(held)

    def aug(q):                                  # 小文字・記号なしの入力に強くする
        return q.lower().rstrip("?") if rnd.random() < 0.2 else q

    know_tr = [aug(q) for q in rnd.sample(tr_q, 5000)] + [rnd.choice(TEMPLATES).format(s=s) for s in rnd.sample(subs_tr, 1500)]
    know_te = rnd.sample(te_q, 300) + [rnd.choice(TEMPLATES).format(s=s) for s in subs_te]
    # 会話側(0)の例: 設計方針は「事実の質問はカードで答える」。会話データの qa / multi / cmd には「Where do fish live?」のような
    # 事実の質問が混ざっていて、これを会話(0)として学習すると知識の質問が会話に回ってしまった。
    # そこで 会話側は ①greet/self/react 型 ②話し手自身(I/my/me/we/you/your)に関わる発言や、質問でない発言 だけにする
    PRON = re.compile(r"\b(i|my|me|mine|we|our|you|your|yours|mia)\b", re.I)

    def keep(cat, t):
        return cat in ("greet", "self") or bool(PRON.search(t)) or not t.rstrip().endswith("?")
    users = lambda f: [t for l in open("data/" + f, encoding="utf-8") for r in [json.loads(l)] for t in r["turns"][0::2] if keep(r.get("cat"), t)]
    chat_tr = rnd.sample(users("chat_qa/chat_clean_train.jsonl"), 6500)
    chat_te = rnd.sample(users("chat_qa/chat_clean_val.jsonl"), 428)
    return know_tr, know_te, chat_tr, chat_te


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_backbone(dev)
    know_tr, know_te, chat_tr, chat_te = load_data()
    feat = lambda xs: torch.stack([features(tok, model, x, dev) for x in xs])
    Xtr, Xte = feat(know_tr + chat_tr), feat(know_te + chat_te)
    ytr = torch.tensor([1] * len(know_tr) + [0] * len(chat_tr), device=dev)
    yte = torch.tensor([1] * len(know_te) + [0] * len(chat_te), device=dev)
    torch.manual_seed(0)
    head = make_head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=0.01)
    for ep in range(30):
        head.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = nn.functional.cross_entropy(head(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            head.eval()
            with torch.no_grad():
                acc = lambda X, y: (head(X).argmax(-1) == y).float().mean().item()
                print(f"epoch {ep + 1}: train loss {loss.item():.3f} 学習正答率 {acc(Xtr, ytr):.4f} 評価正答率 {acc(Xte, yte):.4f}")
    head.eval()
    with torch.no_grad():
        pred = head(Xte).argmax(-1)
    n = len(know_te)
    print(f"評価(学習に使っていない題材・会話): 知識の再現率 {(pred[:n] == 1).float().mean():.3f} ({n}件) | 会話の再現率 {(pred[n:] == 0).float().mean():.3f} ({len(chat_te)}件)")
    torch.save(head.state_dict(), HEAD_PATH)
    print("saved", HEAD_PATH)
