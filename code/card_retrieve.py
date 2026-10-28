# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — 項目(カード)の題名索引: ルーターMiaが出した題名 + 質問から、項目を1つ選ぶ
  1) 正規化(小文字・複数形をそろえる)で題名に完全一致 → それ
  2) なければ、題名の文字列の近さ + 質問に題名が出ているか で採点し、しきい値以上の最良を採用 (なければ None = I do not know)"""
import difflib, glob, json, re
from card_stem import stem_words
from gen_card_data import load_entries, mentions

def norm(s): return " ".join(stem_words(s))

class Index:
    def __init__(self):
        self.ents, _ = load_entries()
        self.by_norm = {}
        for t in self.ents: self.by_norm.setdefault(norm(t), t)
        self.titles = list(self.ents)

    def find(self, routed, question="", thr=0.75):
        n = norm(routed)
        if n in self.by_norm: return self.by_norm[n]
        best, bs = None, 0.0
        rw = set(n.split())
        for t in self.titles:
            tn = norm(t); tw = set(tn.split())
            sc = difflib.SequenceMatcher(None, n, tn).ratio()
            if rw and tw: sc = max(sc, 0.9 * len(rw & tw) / max(len(rw), len(tw)) + 0.1 * (rw <= tw or tw <= rw))
            sc += 0.15 * mentions(t, question)        # 質問に題名が出ていれば加点
            if sc > bs: best, bs = t, sc
        return best if bs >= thr else None


class EmbIndex(Index):
    """本文の埋め込み検索 (Qwen3-Embedding-0.6B、項目=題名+本文) + ルーターの題名の手がかり (w=0.05)。cache: data/card_emb_entries.npy"""
    TASK = "Given a question from a child, retrieve the encyclopedia entry that answers it"

    def __init__(self, device="cuda", w=0.05):
        super().__init__()
        import numpy as np, torch
        from sentence_transformers import SentenceTransformer
        self.np, self.w, self.ti = np, w, {t: i for i, t in enumerate(self.titles)}
        self.st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device, model_kwargs={"torch_dtype": torch.float16})
        self.st.max_seq_length = 512; self.st.tokenizer.padding_side = "left"
        self.E = np.load("data/card_emb_entries.npy")

    LO, HI = 0.50, 0.70        # 採用の下限: 類似度>=HI、または 類似度>=LO かつ ルーターの話題の語が選ばれた項目の本文に出る (exp_card_thr2.py で決定)

    def lex(self, topic, title):
        body = set(stem_words(title + " " + self.ents[title]["entry"])); w = stem_words(topic)
        return bool(w) and all(x in body for x in w)

    def top(self, routed, question, k=3, gate=True):
        """返り値: [題名 ...] 上位k件。routed はルーターの題名 (無ければ "")。gate=True なら、1位が採用基準に満たないとき [] (= 項目なし → I do not know.)。"""
        np = self.np
        q = self.st.encode([f"Instruct: {self.TASK}\nQuery:{question}"], normalize_embeddings=True, convert_to_numpy=True)[0]
        s = self.E @ q
        if routed:
            n = norm(routed)
            if n in self.by_norm: s[self.ti[self.by_norm[n]]] += self.w * 1.0
            for j, t in enumerate(self.titles):
                if mentions(t, question): s[j] += self.w * 0.5
        order = np.argsort(-s)
        if gate:
            raw = float(self.E[order[0]] @ q)           # 手がかりの加点を除いた類似度
            if not (raw >= self.HI or (raw >= self.LO and routed and self.lex(routed, self.titles[order[0]]))):
                return []
        return [self.titles[j] for j in order[:k]]
