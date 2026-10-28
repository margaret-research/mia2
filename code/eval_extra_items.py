# -*- coding: utf-8 -*-
"""追加評価: 想起型(recall) と あいさつ型(greet2)。eval_chat.py の固定62問は変えない (旧結果との比較を保つため)。
形式は eval_chat.ITEMS と同じ (cat, ユーザー発言のリスト, route, 合格の正規表現)。判定は judge_chat.py (DeepSeek)。
ここの発言は学習に混ぜない (build_chat_sft.py が同じユーザー発言を含む会話を除外する)。
"""
EXTRA = [
    ("recall", ["My name is Lucy.", "Do you know my name?"], "chat", "lucy"),
    ("recall", ["I have a pet turtle.", "What pet do I have?"], "chat", "turtle"),
    ("recall", ["I am eight years old.", "How old am I?"], "chat", "eight|8"),
    ("recall", ["My favorite fruit is mango.", "Which fruit do I like best?"], "chat", "mango"),
    ("recall", ["I am going to the library.", "Where am I going now?"], "chat", "library"),
    ("recall", ["My best friend is Joe.", "Who is my best friend?"], "chat", "joe"),
    ("recall", ["I lost my green coat.", "What did I lose?"], "chat", "coat"),
    ("recall", ["I live in a small village.", "Where do I live?"], "chat", "village"),
    ("recall", ["I play the drums.", "Do you remember what I play?"], "chat", "drum"),
    ("recall", ["My brother is called Ken.", "Tell me my brother's name."], "chat", "ken"),
    ("recall", ["I ate a banana for lunch.", "What did I eat?"], "chat", "banana"),
    ("recall", ["I have a yellow bike.", "What color is my bike?"], "chat", "yellow"),
    ("recall", ["My name is Pam.", "I like the rain.", "What is my name?"], "chat", "pam"),
    ("recall", ["I am reading a book about space.", "What am I reading about?"], "chat", "space"),
    ("greet2", ["Hi Mia, how is your day going?"], "chat", "."),
    ("greet2", ["Thanks a lot for your help, Mia."], "chat", "."),
    ("greet2", ["Good evening, Mia."], "chat", "."),
    ("greet2", ["See you tomorrow, Mia."], "chat", "."),
    ("greet2", ["I am sorry I forgot."], "chat", "."),
    ("greet2", ["Nice to see you again."], "chat", "."),
]
