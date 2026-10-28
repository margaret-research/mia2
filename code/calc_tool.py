# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-05) — T-025: 四則演算ツール (LLMとは別の「計算エージェント」)

route(text) -> 返事の文字列 or None
  - 入力に四則演算の式 (12 + 5 / twelve plus five / what is the sum of 3 and 4 など) があれば計算して返事を返す
  - なければ None を返す (呼び出し側がLLMに回す)
計算は eval を使わず、Fraction による自前のパーサで行う (誤差なし・安全)。
"""
import re
from fractions import Fraction

UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6}
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES)

OP_PHRASES = [  # 長い句から先に置換する
    (r"\bmultiplied\s+by\b", " * "), (r"\bdivided\s+by\b", " / "), (r"\btimes\b", " * "),
    (r"\bplus\b", " + "), (r"\bminus\b", " - "), (r"\btake\s+away\b", " - "),
]
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+"


class CalcError(Exception):
    pass


def words_to_number(words):
    """['one','hundred','and','five'] -> 105 / ['twenty','one'] -> 21。解釈できなければ None。"""
    words = [w for w in words if w != "and"]
    if not words:
        return None
    total, cur = 0, 0
    for w in words:
        if w in UNITS:
            cur += UNITS[w]
        elif w in TENS:
            cur += TENS[w]
        elif w == "hundred":
            cur = (cur or 1) * 100
        elif w in ("thousand", "million"):
            total += (cur or 1) * SCALES[w]
            cur = 0
        else:
            return None
    return total + cur


def normalize(text, tight_ops=True):
    """英語の数詞・演算語を数字と記号に直し、トークン列 (数値は文字列, 演算子は記号) を返す。"""
    t = text.lower()
    for pat, sym in OP_PHRASES:
        t = re.sub(pat, sym, t)
    t = t.replace("×", " * ").replace("÷", " / ")
    if not tight_ops:
        # "3-4 days" "10/5" のように数字に密着した - / は式とみなさない (日付・範囲・番号の誤検出を避ける)
        t = re.sub(r"(?<=\d)[-/](?=\d)", " ~ ", t)
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", t)           # 3x4
    t = re.sub(r"(?<![a-z])x(?![a-z])", " * ", t)           # 3 x 4
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)             # twenty-one -> twenty one
    raw = re.findall(rf"{NUM}|[a-z']+|[+\-*/()=?]", t)
    out, i = [], 0
    while i < len(raw):
        w = raw[i]
        if w in NUMBER_WORDS:
            j = i
            while j < len(raw) and (raw[j] in NUMBER_WORDS or
                                    (raw[j] == "and" and j + 1 < len(raw) and raw[j + 1] in NUMBER_WORDS
                                     and j > i and raw[j - 1] in ("hundred", "thousand", "million"))):
                j += 1
            n = words_to_number(raw[i:j])
            if n is None:
                out.append(w)
                i += 1
            else:
                out.append(str(n))
                i = j
        else:
            out.append(w)
            i += 1
    return rewrite_verbs(out)


def rewrite_verbs(toks):
    """命令形 "subtract 3 from 10" / "multiply 6 by 7" / "divide 12 by 3" を 数 演算子 数 の並びにする (Claude Code による追加 2026-10-05)。"""
    out, i = [], 0
    while i < len(toks):
        w = toks[i]
        if i + 3 < len(toks) and w == "add" and is_num(toks[i + 1]) and toks[i + 2] == "and" and is_num(toks[i + 3]):
            out += [toks[i + 1], "+", toks[i + 3]]          # add 3 and 4
            i += 4
        elif (i + 3 < len(toks) and is_num(toks[i]) and toks[i + 1] in ("more", "less") and toks[i + 2] == "than"
                and is_num(toks[i + 3])):
            a, b = toks[i], toks[i + 3]                      # 5 more than 3 = 3 + 5 / 3 less than 10 = 10 - 3
            out += [b, "+" if toks[i + 1] == "more" else "-", a]
            i += 4
        elif (i + 3 < len(toks) and is_num(toks[i + 1]) and is_num(toks[i + 3])
                and ((w == "subtract" and toks[i + 2] == "from") or (w in ("multiply", "divide") and toks[i + 2] == "by"))):
            a, b = toks[i + 1], toks[i + 3]
            out += [b, "-", a] if w == "subtract" else [a, "*" if w == "multiply" else "/", b]
            i += 4
        else:
            out.append(w)
            i += 1
    return out


def is_num(tok):
    return re.fullmatch(NUM, tok) is not None


def to_fraction(tok):
    return Fraction(tok.replace(",", ""))


# ---- 式の評価 (再帰下降パーサ、四則と括弧) ----
class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self):
        v = self.peek()
        self.i += 1
        return v

    def expr(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.eat()
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self):
        v = self.factor()
        while self.peek() in ("*", "/"):
            op = self.eat()
            r = self.factor()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise CalcError("div0")
                v = v / r
        return v

    def factor(self):
        tok = self.eat()
        if tok == "-":
            return -self.factor()
        if tok == "(":
            v = self.expr()
            if self.eat() != ")":
                raise CalcError("paren")
            return v
        if tok is not None and is_num(tok):
            return to_fraction(tok)
        raise CalcError("syntax")


def evaluate(toks):
    p = Parser(toks)
    v = p.expr()
    if p.i != len(toks):
        raise CalcError("syntax")
    return v


def fmt_number(v):
    if v.denominator == 1:
        return f"{v.numerator:,}" if abs(v.numerator) >= 1000 else str(v.numerator)
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    exact = Fraction(s) == v
    return s if exact else f"about {s}"


# ---- 入力文から式を探す ----
def find_expression(text):
    """文中から 数 (演算子 数)+ の最初の連なりを探す。返り値は (トークン列, 純粋な式か)。"""
    # 1) 全体が式のみ ("12-5", "(3+4)*2 =" など)。この場合だけ密着した - / も演算子として扱う
    core = [t for t in normalize(text, True) if t not in ("=", "?")]
    if core and all(is_num(t) or t in "+-*/()" for t in core) and any(t in "+-*/" for t in core):
        return core, True
    toks = normalize(text, False)
    # 2) "sum/product/difference of A and B"
    m = re.search(r"\b(sum|product|difference|total)\s+(?:of|between)\s+(\S+)\s+and\s+(\S+)", " ".join(toks))
    if m and is_num(m.group(2)) and is_num(m.group(3)):
        op = {"sum": "+", "total": "+", "product": "*", "difference": "-"}[m.group(1)]
        return [m.group(2), op, m.group(3)], False
    # 3) 文中の "数 演算子 数 (演算子 数)*"。演算子は前後に空白がある記号か、語から変換した記号
    s = " ".join(toks)
    pat = rf"(?<![\w.])(?:{NUM})(?:\s+[+\-*/]\s+(?:{NUM}))+(?![\w.])"
    m = re.search(pat, s)
    if m:
        return m.group(0).split(), False
    return None, False


def route(text):
    toks, _ = find_expression(text)
    if not toks:
        return None
    try:
        v = evaluate(toks)
    except CalcError as e:
        return "I can not divide by zero." if str(e) == "div0" else None
    shown = " ".join(toks)
    return f"{shown} = {fmt_number(v)}"


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        print(route(line.strip()))


def solve(expr_text):
    """モデルが <calc> の後に書いた式の文字列を計算する。返り値は結果の文字列。
    解釈できなければ "error"、ゼロ除算は "error" ("div0" ではなく汎用エラー)。"""
    toks, _ = find_expression(expr_text)
    if not toks:
        return "error"
    try:
        return fmt_number(evaluate(toks))
    except CalcError:
        return "error"
