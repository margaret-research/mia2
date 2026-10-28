# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-26) — ルーターMia(Plan B)用データ: 文章題の型の見本生成器 (API費用なし)

ルーターMiaの出力は3種類: "<chat>" / "<calc> 式" / "<card> 検索語" (今回は chat と calc のみ)。
式は calc_tool.solve で検算する。見本の確認用: python gen_router_data.py [件数]
"""
import json
import random
import sys

from calc_tool import solve

NOUNS = ["pencils", "apples", "stickers", "cards", "books", "balls", "cookies", "stones", "flowers", "cups", "coins", "shells"]
NAMES = ["Tom", "Ann", "Ben", "Sue", "Mia's friend Sam", "my sister", "my brother", "Dad", "Mom"]
ONES = "zero one two three four five six seven eight nine ten eleven twelve".split()


def w(x, p=0.25):
    return ONES[x] if x <= 12 and random.random() < p else str(x)


def pair(op):
    big = random.random() < 0.3      # 3桁・4桁の数 (v2)
    if big and op == "+":
        return random.randint(10, 2500), random.randint(10, 2500)
    if big and op == "-":
        a = random.randint(100, 3000)
        return a, random.randint(10, a - 1)
    if big and op == "*":
        return random.randint(12, 120), random.randint(2, 12)
    if big and op == "/":
        b = random.randint(2, 12)
        return b * random.randint(10, 120), b
    if op == "+":
        return random.randint(2, 40), random.randint(2, 40)
    if op == "-":
        a = random.randint(5, 60)
        return a, random.randint(1, a - 1)
    if op == "*":
        return random.randint(2, 12), random.randint(2, 12)
    b = random.randint(2, 9)
    return b * random.randint(2, 12), b


# 各型: (型名, 作る関数)。関数は (発言, 式) を返す。式は数字のみ (発言の数詞は w() で語になることがある)
def t_join():
    a, b = pair("+"); n = random.choice(NOUNS); p = random.choice(NAMES)
    u = random.choice([f"I have {w(a)} {n}. {p.capitalize()} gives me {w(b)} more. How many {n} do I have now?",
                       f"I have {w(a)} {n} and {p} has {w(b)}. How many {n} do we have?",
                       f"{w(a).capitalize()} {n} are in the box. I put in {w(b)} more. How many {n} are in the box now?"])
    return u, f"{a} + {b}"


def t_total_plain():   # 演算語なし: 「鉛筆2本と鉛筆3本で何本」型
    a, b = pair("+"); n = random.choice(NOUNS)
    u = random.choice([f"{w(a).capitalize()} {n} and {w(b)} {n}. How many {n} is that?",
                       f"I see {w(a)} {n} here and {w(b)} {n} there. How many {n} are there in all?",
                       f"There are {w(a)} red {n} and {w(b)} blue {n}. How many {n} are there?"])
    return u, f"{a} + {b}"


def t_left():
    a, b = pair("-"); n = random.choice(NOUNS); p = random.choice(NAMES)
    u = random.choice([f"I have {w(a)} {n}. I give {w(b)} to {p}. How many {n} do I have left?",
                       f"There are {w(a)} {n}. {w(b).capitalize()} of them are lost. How many are left?",
                       f"I had {w(a)} {n} and I used {w(b)}. How many {n} do I have now?"])
    return u, f"{a} - {b}"


def t_more():          # 差: 何個多い
    a, b = pair("-"); n = random.choice(NOUNS); p = random.choice(NAMES)
    u = random.choice([f"I have {w(a)} {n} and {p} has {w(b)}. How many more {n} do I have?",
                       f"{p.capitalize()} has {w(b)} {n}. I have {w(a)}. How many more do I have than {p}?"])
    return u, f"{a} - {b}"


def t_groups():
    a, b = pair("*"); n = random.choice(NOUNS)
    u = random.choice([f"Each bag has {w(a)} {n}. I have {w(b)} bags. How many {n} do I have in all?",
                       f"There are {w(b)} boxes. Each box has {w(a)} {n}. How many {n} are there?",
                       f"I buy {w(b)} {n}. Each one costs {w(a)} dollars. How much do I pay?"])
    return u, f"{a} * {b}"


def t_share():
    a, b = pair("/"); n = random.choice(NOUNS)
    u = random.choice([f"I have {w(a)} {n} and {w(b)} friends. I share them equally. How many does each friend get?",
                       f"I have {w(a)} {n}. I put {w(b)} in each box. How many boxes do I need?"])
    return u, f"{a} / {b}"


def t_two_step():
    n = random.choice(NOUNS)
    k = random.choice(["add_sub", "mul_add", "mul_sub"])
    if k == "add_sub":
        a, b = pair("+"); c = random.randint(2, a + b - 1)
        return (f"I have {a} {n}. I get {b} more. Then I give {c} away. How many {n} do I have now?", f"{a} + {b} - {c}")
    a, b = pair("*"); c = random.randint(2, 20)
    if k == "mul_add":
        return (f"There are {b} bags with {a} {n} in each. I find {c} more {n}. How many {n} do I have?", f"{b} * {a} + {c}")
    c = random.randint(1, a * b - 1)
    return (f"There are {b} bags with {a} {n} in each. I eat {c} of them. How many {n} are left?", f"{b} * {a} - {c}")


def t_distractor():    # 式に使わない数字 (年齢・時刻・番号) が混ざる
    a, b = pair("+"); n = random.choice(NOUNS)
    extra = random.choice([f"I am {random.randint(6, 9)} years old.", f"It is {random.randint(1, 11)} o'clock.",
                           f"My room is number {random.randint(2, 30)}.", f"Today is the {random.randint(2, 28)}th."])
    u = random.choice([f"{extra} I have {a} {n} and I get {b} more. How many {n} do I have?",
                       f"I have {a} {n}. {extra} My friend gives me {b} more. How many {n} do I have now?"])
    return u, f"{a} + {b}"


def t_followup():      # 前の発言の数を受ける (履歴つきの型; 見本では直前の1文を前置)
    a, b = pair("+"); n = random.choice(NOUNS)
    c = random.randint(2, 9)
    return (f"I have {a} {n}.", f"My friend has {b} {n}.", f"How many {n} do we have together?"), f"{a} + {b}"


def t_three():         # 3項の足し算
    a, b, c = random.randint(2, 30), random.randint(2, 30), random.randint(2, 30)
    n1, n2, n3 = random.sample(["dogs", "cats", "birds", "fish", "apples", "pears", "pens", "books", "toys"], 3)
    u = random.choice([f"I have {w(a)} {n1}, {w(b)} {n2} and {w(c)} {n3}. How many things do I have?",
                       f"There are {w(a)} {n1} and {w(b)} {n2} and {w(c)} {n3}. How many are there in all?",
                       f"I bought {w(a)} {n1}, {w(b)} {n2} and {w(c)} {n3}. How many did I buy?"])
    return u, f"{a} + {b} + {c}"


def t_change():        # おつり・のこりのお金
    a = random.randint(5, 100); b = random.randint(1, a - 1); it = random.choice(["pen", "toy", "book", "ball", "cake", "hat"])
    u = random.choice([f"I have {a} dollars. A {it} costs {b} dollars. How much money is left if I buy it?",
                       f"I pay {a} dollars for a {it} that costs {b} dollars. How much change do I get?",
                       f"Mom gives me {a} dollars. I buy a {it} for {b} dollars. How many dollars do I have now?"])
    return u, f"{a} - {b}"


def t_older():         # 年上・年下
    a = random.randint(5, 40); b = random.randint(2, 4); r = random.choice(["brother", "sister", "friend", "cousin"])
    if random.random() < 0.5:
        return random.choice([f"I am {a} years old. My {r} is {b} years older. How old is my {r}?",
                              f"My {r} is {b} years older than me. I am {a}. How old is my {r}?"]), f"{a} + {b}"
    a = a + b
    return random.choice([f"I am {a} years old. My {r} is {b} years younger. How old is my {r}?",
                          f"My {r} is {b} years younger than me. I am {a}. How old is my {r}?"]), f"{a} - {b}"


def t_half():          # 半分・2倍
    k = random.choice(["half", "half2", "double", "twice"]); n = random.choice(NOUNS); a = random.randint(2, 50)
    if k == "half":
        return random.choice([f"Half of {2 * a} is how many?", f"What is half of {2 * a}?"]), f"{2 * a} / 2"
    if k == "half2":
        return random.choice([f"I have {2 * a} {n}. Half of them are red. How many are red?",
                              f"There are {2 * a} {n}. I give half of them to a friend. How many do I give?"]), f"{2 * a} / 2"
    if k == "double":
        return random.choice([f"What is double {a}?", f"What is {a} doubled?"]), f"{a} * 2"
    return random.choice([f"I have {a} {n}. My friend has twice as many. How many does my friend have?",
                          f"Tom has {a} {n}. Ann has two times as many. How many does Ann have?"]), f"{a} * 2"


def t_expr():          # 式だけ・「何ですか」型 (para なしでテンプレートのまま使う)
    op = random.choice(["+", "-", "*", "/"]); a, b = pair(op)
    if random.random() < 0.15 and op in "+*":      # 3項
        c = random.randint(2, 20)
        sym = op; word = {"+": "plus", "*": "times"}[op]
        e = f"{a} {sym} {b} {sym} {c}"
        return random.choice([f"What is {a} {word} {b} {word} {c}?", f"{a} {sym} {b} {sym} {c} = ?", f"Please calculate {a} {word} {b} {word} {c}."]), e
    word = {"+": ["plus", "+"], "-": ["minus", "-"], "*": ["times", "x", "multiplied by"], "/": ["divided by", "/"]}[op]
    wd = random.choice(word); A, B = (w(a, .3), w(b, .3)) if wd.isalpha() or " " in wd else (str(a), str(b))
    sumof = {"+": "sum", "*": "product", "-": "difference"}.get(op)
    forms = [f"What is {A} {wd} {B}?", f"{A} {wd} {B}", f"Please calculate {A} {wd} {B}.", f"How much is {A} {wd} {B}?",
             f"what is {A} {wd} {B}", f"Can you tell me {A} {wd} {B}?", f"{A} {wd} {B} = ?"]
    if sumof and op != "-":
        forms.append(f"What is the {sumof} of {A} and {B}?")
    if op == "+": forms.append(f"Add {A} and {B}.")
    if op == "-": forms.append(f"Subtract {B} from {A}.")
    f = random.choice(forms)
    e = f"{a} {op} {b}"
    if f.startswith("Subtract"): e = f"{a} - {b}"
    return f, e


CALC_TYPES = [t_join, t_total_plain, t_left, t_more, t_groups, t_share, t_two_step, t_distractor, t_three, t_change, t_older, t_half, t_expr]

CHAT_NUM = [  # 数字はあるが計算ではない発言 (chatに回す)
    "I am 7 years old.", "I have 3 cats at home.", "My phone number is 5 5 5 0 1 2 3.", "I go to bed at 9 o'clock.",
    "My favorite number is 7.", "I have two brothers. Do you have any?", "I saw 4 birds in the park today.",
    "My room is number 12.", "Do you like the number 8?", "I eat 3 times a day. What about you?",
    "We have 20 kids in my class.", "I read 2 books last week.",
]


def sample(n):
    out = []
    for i in range(n):
        if i % 4 == 3:
            out.append({"user": random.choice(CHAT_NUM), "route": "<chat>", "type": "chat_num"})
            continue
        f = random.choice(CALC_TYPES)
        u, e = f()
        out.append({"user": u, "route": f"<calc> {e}", "result": solve(e), "type": f.__name__[2:]})
    return out


if __name__ == "__main__":
    random.seed(1)
    for r in sample(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
        print(json.dumps(r, ensure_ascii=False))
