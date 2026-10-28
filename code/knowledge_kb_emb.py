# -*- coding: utf-8 -*-
"""知識カードの検索を、文の埋め込み(Qwen3-Embedding-0.6B)+コサイン類似度にした版 (RAG)。
KB(TF-IDF)と同じカードを読み込み、検索だけを置き換える。埋め込みは data/kb_emb_*.npy にキャッシュ(カード一覧のハッシュが変われば作り直す)。
質問側は推奨の指示文つき、カード側はカード文だけを埋め込む (評価: 正解カードが上位3枚に入る率 TF-IDF 49% → 82%)。
sentence-transformers が要る: python で実行する。"""
import hashlib
import os

import numpy as np

from knowledge_kb import KB

MODEL = "Qwen/Qwen3-Embedding-0.6B"
TASK = "Given a question from a child, retrieve the fact card that answers it"


class KBE(KB):
    def __init__(self, *a, device="cpu", cache="data/kb_emb_qwen3_0.6b", **kw):
        super().__init__(*a, **kw)
        from sentence_transformers import SentenceTransformer
        import torch
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.st = SentenceTransformer(MODEL, device=device, model_kwargs={"torch_dtype": dtype})
        self.st.max_seq_length = 128
        self.st.tokenizer.padding_side = "left"
        h = hashlib.md5("\n".join(self.cards).encode()).hexdigest()
        if os.path.exists(cache + ".npy") and os.path.exists(cache + ".md5") and open(cache + ".md5").read() == h:
            self.emb = np.load(cache + ".npy")
        else:
            self.emb = self.st.encode(self.cards, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            np.save(cache + ".npy", self.emb)
            open(cache + ".md5", "w").write(h)

    def search(self, query, k=3):
        """返り値: [(コサイン類似度, カード, 題材)] 上位k件。"""
        q = self.st.encode([f"Instruct: {TASK}\nQuery:{query}"], normalize_embeddings=True, convert_to_numpy=True)[0]
        s = self.emb @ q
        top = s.argsort()[::-1][:k]
        return [(float(s[i]), self.cards[i], self.subjects[i]) for i in top]
