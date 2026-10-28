# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-27) — 英語の複数形を簡単にそろえる (題名の索引用。完全でなくてよい)"""
import re
def stem_word(w):
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ves"): return w[:-3] + "f"
    if len(w) > 4 and re.search(r"(ch|sh|ss|x|z|o)es$", w): return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"): return w[:-1]
    return w
def stem_words(s): return [stem_word(w) for w in re.findall(r"[a-z0-9']+", s.lower())]
