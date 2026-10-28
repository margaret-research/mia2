# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-08) — T-026 会話Bot: 知識カードの検索 (TF-IDF)

知識カード = 読解SFTの生成データ (data/reading_qa/raw2_*.jsonl) のうち、名前入りの作り話を除いた「カード」。
質問から、題材名(subject)を重く見て TF-IDF で上位カードを探す。生成カードは外部(DeepSeek)由来のテキスト。
使い方: python knowledge_kb.py "What is a deck?"
"""
import glob
import json
import re
import sys

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# 質問の枠の語 (検索の手がかりにならない)
QWORDS = {"what", "who", "where", "when", "which", "how", "many", "much", "does", "do", "did", "is", "are", "can", "tell", "know",
          "about", "kind", "kinds", "usually", "get", "make"}
DROP = set(ENGLISH_STOP_WORDS) | QWORDS


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def analyzer(text):
    return [stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in DROP]


class KB:
    def __init__(self, pattern="data/reading_qa/raw2_s*.jsonl", extra="data/kb_lowgrade/facts_final.jsonl"):
        cards = {}
        for f in sorted(glob.glob(pattern)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if not r.get("named"):
                    cards.setdefault(r["card"], r["subject"])
        # 低学年向けに追加した事実カード。題材名は outline の括弧の崩れを直して使う
        for f in sorted(glob.glob(extra)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                cards.setdefault(r["fact"], r["subject"].strip(" ()"))
        self.cards = list(cards)
        self.subjects = [cards[c] for c in self.cards]
        docs = [(s + " ") * 2 + c for c, s in zip(self.cards, self.subjects)]
        self.vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.mat = self.vec.fit_transform(docs)

    def search(self, query, k=3):
        """返り値: [(スコア, カード, 題材)] 上位k件。語がカードに1つもなければ空。"""
        q = self.vec.transform([query])
        if q.nnz == 0:
            return []
        sims = (self.mat @ q.T).toarray().ravel()
        top = sims.argsort()[::-1][:k]
        return [(float(sims[i]), self.cards[i], self.subjects[i]) for i in top if sims[i] > 0]


if __name__ == "__main__":
    kb = KB()
    print(f"カード {len(kb.cards)} 枚 / 題材 {len(set(kb.subjects))}")
    for q in sys.argv[1:] or ["What is a deck?", "Do alpacas have wool?", "How many legs does a spider have?"]:
        print("Q:", q)
        for s, c, sub in kb.search(q):
            print(f"   {s:.3f} [{sub}] {c}")
