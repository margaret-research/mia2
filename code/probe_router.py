# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-06) — 方式Cルーターの手書きプローブ (生成器とは別に自分で書いた言い回し)"""
import sys
from router import Router
P = [("What is 6*7?", "42"), ("6*7", "42"), ("12+5", "17"), ("what's 20 / 4?", "5"), ("whats 9 times 9", "81"),
     ("how much is 45 minus 17", "28"), ("Can you tell me what fifteen plus twenty-seven is?", "42"),
     ("what is 1,000 plus 2,500?", "3,500"), ("what is 3.5 plus 1.25?", "4.75"),
     ("I bought 3 books for 12 dollars each. How much did I spend?", "36"),
     ("I have 20 stickers and give 8 away. How many are left?", "12"),
     ("There are 9 birds. 4 fly away. How many birds are left?", "5"),
     ("Each pack has 6 pens. I have 7 packs. How many pens do I have?", "42"),
     ("I have 24 cookies and 6 friends. How many cookies does each friend get?", "4"),
     ("Add 12 and 30.", "42"), ("Please multiply 8 by 7.", "56"), ("Divide 81 by 9.", "9"),
     ("What is the sum of 100 and 250?", "350"), ("What is 7 less than 50?", "43"),
     ("How much is 8×9?", "72"), ("what is 56 ÷ 7", "8"), ("72/9", "8"), ("6 * 7 =", "42"),
     ("I had 30 candies and gave 12 away. How many remain?", "18"),
     ("There are 15 apples in a box and 7 are taken out. How many are left?", "8"),
     ("20 candies are shared by 4 kids. How many does each kid get?", "5"),
     ("A bag holds 8 oranges. How many oranges do 5 bags hold?", "40"),
     ("I read 12 pages a day. How many pages do I read in 6 days?", "72"),
     ("What is 9 more than 14?", "23"), ("What is the difference between 50 and 8?", "42"), ("Please subtract 15 from 40.", "25"),
     ("Hi! How are you today?", None), ("I will stay for 3-4 days.", None), ("I have 3 cats and 2 dogs.", None),
     ("My phone number is 555-1234.", None), ("Meet me at 5 o'clock.", None), ("I am 12 years old.", None),
     ("Let us meet on the 5th.", None), ("The movie starts at 7:30.", None), ("I like number 7.", None)]
nohist = "--nohist" in sys.argv
r = Router(heads="ckpt/router_heads_nohist.pt" if nohist else "ckpt/router_heads.pt", nohist=nohist)
ok = 0
for q, want in P:
    out = r.route(["Hi! How are you today?", "I'm good, thanks. How about you?", q])
    got = None if out is None else out[1]
    good = got == want
    ok += good
    print("OK " if good else "NG ", f"{q[:62]:62s}", "->", out)
print(f"{'履歴なし' if nohist else '履歴あり'}: {ok}/{len(P)}")
