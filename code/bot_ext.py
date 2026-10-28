# (新トークナイザー版 2026-10-12: bot.py から router_ext / know_router_ext / 新ckpt に差し替えただけ)
# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — T-026 会話Bot: 入力を「普通の会話 / 計算 / 知識」に振り分けて返す

  1. 計算ルーター (router.Router, 凍結LLM+op/tag頭) が「計算」と判断 -> 電卓 (calc_tool.solve) の結果を返す
  2. 知識ルーター (know_router.KnowRouter) が「知識の質問」と判断 -> 知識カードを検索 (knowledge_kb.KB) し、
     読解モデル (ckpt/readqa_*.pt) に「Facts: カード Question: 質問」を渡して答えさせる。上位3枚を順に試し、
     "I do not know." でない最初の答えを返す。カードが見つからない/全部IDKなら "I do not know."
  3. それ以外 -> 会話モデル (ckpt/chat_sft_3ep_x.pt) が履歴つきで返事
使い方: python bot.py                       (対話。空行で終了。--card でカードも表示)
        python bot.py "text1" "text2" ...
読解モデルは Drive の llm5k/readqa_ext_e15_m120k.pt を ckpt/ に置く (--qa で別のckptも指定可)。
"""
import argparse
import os
import re
import sys

import torch

from knowledge_kb import KB
from know_router_ext import KnowRouter
from model import GPT, Config
from router_ext import Router

IDK = "I do not know."


class Bot:
    def __init__(self, qa_ckpt="ckpt/readqa_ext_e15_m120k.pt", know_threshold=0.5, min_score=0.25, dev=None):
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.calc = Router(self.dev, "ckpt/router_heads_ext_nohist.pt", nohist=True)
        self.tok, self.chat_model = self.calc.tok, self.calc.model
        self.know = KnowRouter(self.tok, self.chat_model, self.dev)
        self.kb = KB()
        self.know_threshold, self.min_score = know_threshold, min_score
        t = self.tok
        self.BOS, self.EOS, self.USER, self.BOT = (t.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
        self.qa = None
        if os.path.exists(qa_ckpt):
            self.qa = GPT(Config(vocab_size=t.get_vocab_size(), block_size=256)).to(self.dev)
            self.qa.load_state_dict(torch.load(qa_ckpt, map_location=self.dev)["model"])
            self.qa.eval()
        self.hist = []

    def reset(self):
        self.hist = []

    @torch.no_grad()
    def answer_from_card(self, card, question, max_new=60):
        """読解モデルの貪欲生成。"""
        ids = [self.BOS, self.USER] + self.tok.encode(f"Facts: {card} Question: {question}").ids + [self.BOT]
        n0 = len(ids)
        for _ in range(max_new):
            logits, _ = self.qa(torch.tensor([ids[-256:]], device=self.dev))
            nxt = int(logits[0, -1].argmax())
            if nxt == self.EOS:
                break
            ids.append(nxt)
        return self.tok.decode(ids[n0:]).strip()

    def knowledge(self, q):
        """返り値: (答え, 使ったカード or None, 試したカードの数)"""
        if self.qa is None:
            return "(読解モデルの ckpt が見つかりません)", None, 0
        hits = [h for h in self.kb.search(q, 3) if h[0] >= self.min_score]
        self.last_searched = [c for _, c, _ in hits]
        for _, card, _ in hits:
            a = self.answer_from_card(card, q)
            if a and a != IDK:
                return a, card, len(hits)
        return IDK, None, len(hits)

    @torch.no_grad()
    def say(self, u, temperature=0.7):
        """返り値: dict(who='calc'|'know'|'chat', reply, ...)"""
        t = self.tok
        out = self.calc.route([u])
        if out is not None:
            expr, res = out
            reply = ("I can not divide by zero." if "/" in expr else "I can not calculate that.") if res == "error" else f"The answer is {res}."
            info = dict(who="calc", expr=expr)
        else:
            p = self.know.prob(u)
            # 「あなたの名前は?」のようにBot自身への質問(you/your)は、「知っている?/教えて」型を除いて会話に回す
            about_bot = bool(re.search(r"\b(you|your|yours)\b", u.lower())) and not re.search(r"\b(know|tell)\b", u.lower())
            if p >= self.know_threshold and not about_bot:
                reply, card, n = self.knowledge(u)
                info = dict(who="know", p=round(p, 3), card=card, tried=n, searched=self.last_searched)
            else:
                info = dict(who="chat", p=round(p, 3))
                ctx = ([self.BOS] + self.hist + [self.USER] + t.encode(u).ids + [self.BOT])[-200:]
                if ctx[0] != self.BOS:
                    ctx = [self.BOS] + ctx
                gen = self.chat_model.generate(torch.tensor([ctx], device=self.dev), 40, temperature=temperature, top_k=40, eos=self.EOS)[0].tolist()[len(ctx):]
                reply = t.decode([i for i in gen if i != self.EOS]).strip()
        # 履歴には全部の種類のやりとりを残す (会話の流れを保つため)
        self.hist += [self.USER] + t.encode(u).ids + [self.BOT] + t.encode(reply).ids + [self.EOS]
        return dict(reply=reply, **info)


def show(r, card=False):
    tag = {"calc": f"計算 式: {r.get('expr')}", "know": f"知識 p={r.get('p')} カード{r.get('tried')}枚検索", "chat": f"会話 p={r.get('p')}"}[r["who"]]
    s = f"Bot[{tag}]: {r['reply']}"
    if r["who"] == "know":
        searched = r.get("searched") or []
        if not searched:
            s += "\n      (該当するカードが見つかりませんでした)"
        for i, c in enumerate(searched, 1):
            mark = "★根拠" if c == r.get("card") else "  "
            s += f"\n      {mark} 検索カード{i}: {c}"
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--qa", default="ckpt/readqa_ext_e15_m120k.pt")
    ap.add_argument("--card", action="store_true", help="(互換用) 根拠カードは常に表示されます")
    a = ap.parse_args()
    bot = Bot(a.qa)
    if bot.qa is None:
        print(f"[注意] {a.qa} がありません。知識の質問は答えられません (計算と会話は動きます)", file=sys.stderr)
    if a.texts:
        for u in a.texts:
            print(f"You: {u}\n{show(bot.say(u), a.card)}")
    else:
        print("(終了するには exit と入力)")
        while True:
            try:
                u = input("You: ").strip()
            except EOFError:
                break
            if u.lower() == "exit":
                break
            if not u:
                continue
            print(show(bot.say(u), a.card))
