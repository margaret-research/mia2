# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — MIA2 の会話Bot(計算つき): ルーターMia → 電卓 → 固定文、または現行Miaの会話
  1. ルーターMia (ckpt/router_mia_v6.pt) が履歴と今の発言を見て "calc 式" か "chat" を出す
  2. calc なら電卓 (calc_tool.solve) で計算し、答え役Mia (ckpt/answer_mia_v1.pt) が "Facts: 式 = 結果. Question: 発言" を読んで自然な文で答える
     (答え役が使えない/数が合わないときは固定文 "The answer is X.")
  3. card 題名 なら、項目(題名+本文)を本文の埋め込み検索(card_retrieve.EmbIndex、上位1枚)で選び、現行Miaが読解で答える。見つからなければ "I do not know."
  4. chat なら現行Mia (ckpt/mia_current_read.pt) が履歴つきで返事
使い方: MIA_TOK=data/ts_v4/tokenizer_ext.json MIA_NLAYER=10 MIA_NHEAD=8 MIA_NEMBD=512 python bot_mia2.py ["発言1" "発言2" ...]"""
import os, sys
import torch
from tokenizers import Tokenizer
from model import GPT, Config
from calc_tool import solve


class Bot:
    def __init__(self, router="ckpt/router_mia_v6.pt", chat="ckpt/mia_current_read.pt", answer="ckpt/answer_mia_v1.pt", dev="cuda"):
        self.dev = dev
        self.tok = Tokenizer.from_file(os.environ["MIA_TOK"])
        self.BOS, self.EOS, self.USER, self.BOT = (self.tok.token_to_id(x) for x in ("<bos>", "<eos>", "<user>", "<bot>"))
        load = lambda p: self._load(p)
        self.router, self.chat = load(router), load(chat)
        self.answer = load(answer) if answer and os.path.exists(answer) else None
        self.turns = []                      # [発言, 返事, 発言, 返事, ...]
        self.kb = None                       # 項目の索引 (初めてカードが出たときに読み込む)

    def _load(self, p):
        m = GPT(Config(vocab_size=self.tok.get_vocab_size(), block_size=1024)).to(self.dev)
        m.load_state_dict(torch.load(p, map_location=self.dev)["model"]); return m.eval()

    def ctx(self, user, hist_n=6):
        ids = [self.BOS]
        for i, t in enumerate(self.turns[-hist_n:] if hist_n % 2 == 0 else self.turns[-hist_n:]):
            ids += ([self.USER] + self.tok.encode(t).ids) if i % 2 == 0 else ([self.BOT] + self.tok.encode(t).ids + [self.EOS])
        return ids + [self.USER] + self.tok.encode(user).ids + [self.BOT]

    @torch.no_grad()
    def gen(self, model, ids, n, sample=False):
        n0 = len(ids)
        for _ in range(n):
            lg, _ = model(torch.tensor([ids[-1024:]], device=self.dev)); lg = lg[0, -1]
            nx = int(torch.multinomial(torch.softmax(lg / 0.7, -1), 1)) if sample else int(lg.argmax())
            if nx in (self.EOS, self.USER): break
            ids.append(nx)
        return self.tok.decode(ids[n0:]).strip()

    def say(self, user):
        h = self.turns[-6:]
        if len(h) % 2: h = h[1:]              # 履歴は 発言/返事 の組で切る (ルーターの学習と同じ)
        saved, self.turns = self.turns, h
        out = self.gen(self.router, self.ctx(user), 40)
        if out.startswith("calc"):
            expr = out[4:].strip(); res = solve(expr)
            if res == "error":
                reply = "I can not calculate that."
            else:
                reply = f"The answer is {res}."
                if self.answer is not None:
                    a = self.gen(self.answer, self.ctx(f"Facts: {expr} = {res}. Question: {user}"), 50)
                    if a and a.replace(",", "").count(res.replace(",", "")) == 1 and "?" not in a:   # 結果をちょうど1回含む答えだけ採用
                        reply = a
            who = f"計算 {expr}"
        elif out.split()[:1] == ["card"]:
            if self.kb is None:
                from card_retrieve import EmbIndex
                self.kb = EmbIndex(self.dev)
            title = (self.kb.top(out[4:].strip(), user, 1) or [None])[0]
            if title is None:                # 項目なし (検索の下限に満たない) → 作り話をしない
                reply = "I do not know."
            else:
                ids = [self.BOS, self.USER] + self.tok.encode(f"Title: {title}. {self.kb.ents[title]['entry']} Question: {user}").ids + [self.BOT]
                reply = self.gen(self.chat, ids, 60)
            who = f"カード {out[4:].strip()} → 項目「{title}」"
        else:
            reply = self.gen(self.chat, self.ctx(user), 60, sample=True); who = "会話"
        self.turns = saved + [user, reply]
        return who, reply


if __name__ == "__main__":
    bot = Bot()
    if len(sys.argv) > 1:
        for u in sys.argv[1:]:
            w, r = bot.say(u); print(f"You: {u}\nMia[{w}]: {r}")
    else:
        while True:
            try: u = input("You: ").strip()
            except EOFError: break
            if u == "exit": break
            if u: w, r = bot.say(u); print(f"Mia[{w}]: {r}")
