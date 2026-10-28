# -*- coding: utf-8 -*-
"""実験: 曲がったアポストロフィ(’)も許す文法にすると、Its / Don / His の癖が減るか (gen_chat_qa_gbnf.py は変更しない)

文法の短縮形の部分だけ ' に加えて ’ も許し、保存前に ’ を ' に直す。出現数(生のまま)を数えて表示する。
使い方: SEED=555 EXT=1 WORDSEED=1 python exp_curly_apostrophe.py exp_curly 250      (出力は raw_<tag>.jsonl。kept_ は作らないので30万件の集計には混ざらない)
"""
import sys

import gen_chat_qa_gbnf as g

CURLY = "’"
stats = {"calls": 0, "with_curly": 0, "curly_chars": 0}
_orig_build, _orig_one = g.build_grammar, g.one


def build(n_pairs):
    gr, n = _orig_build(n_pairs)
    old = 'word ::= base ("\'" ('
    assert old in gr
    return gr.replace(old, 'word ::= base (("\'" | "' + CURLY + '") ('), n


def one(job):
    cat, turns = _orig_one(job)
    if turns:
        c = sum(t.count(CURLY) for t in turns)
        stats["calls"] += 1
        stats["with_curly"] += bool(c)
        stats["curly_chars"] += c
        turns = [t.replace(CURLY, "'") for t in turns]
    return cat, turns


g.build_grammar, g.one = build, one
if __name__ == "__main__":
    g.main(sys.argv[1], int(sys.argv[2]))
    print("曲がったアポストロフィの出現:", stats)
