# (新トークナイザー版 2026-10-12: router.py からトークナイザーと既定ckptだけ変更)
# -*- coding: utf-8 -*-
"""Claude Code により作成 (2026-10-06) — T-025 方式C: LLMは凍結し「判断だけ」を訓練する計算ルーター

凍結した LLM (ckpt/sft_5k_v2.pt) の隠れ状態の上に、小さな2つの頭だけを載せて訓練する (LLMの重みは一切変えない)。
  op頭 : 会話の最後の位置 (<bot>) から 計算か/どの演算か を分類  none / add / sub / sub_rev / mul / div
         sub_rev = 「3 less than 10」「subtract 3 from 10」のように 出現順と引く順が逆
  tag頭: ユーザー発言の各トークンを  O / N1(式の1つ目の数) / N2(2つ目の数)  に分類
式は入力文の部分文字列をそのまま取り出して組み立てる (数字の書き写しミスが起きない)。計算は calc_tool.solve。
"""
import re
import torch
import torch.nn as nn
from fractions import Fraction
from tokenizers import Tokenizer
from model import GPT, Config
from calc_tool import NUMBER_WORDS, words_to_number, solve

OPS = ["none", "add", "sub", "sub_rev", "mul", "div"]
TAGS = ["O", "N1", "N2"]
BLOCK = 256
SCALES = ("hundred", "thousand", "million")


def load_tok():
    tok = Tokenizer.from_file("data/ts/tokenizer_ext.json")
    tok.add_special_tokens(["<user>", "<bot>", "<memory>"])
    return tok


def load_backbone(dev="cpu", ckpt="ckpt/chat_sft_3ep_x.pt"):
    tok = load_tok()
    model = GPT(Config(vocab_size=tok.get_vocab_size(), block_size=BLOCK)).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def hidden(model, idx):
    x = model.wte(idx) + model.wpe(torch.arange(idx.shape[1], device=idx.device))
    for b in model.blocks:
        x = b(x)
    return model.lnf(x)


def build_context(tok, turns):
    """turns = [user, bot, user, ..., user(最後)]。返り値: ids ([BOS] ... <user> 発言 <bot>), 最後の発言の (トークン開始位置, offsets, 本文)。"""
    BOS, EOS = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    USER, BOT = tok.token_to_id("<user>"), tok.token_to_id("<bot>")
    while True:
        ids = [BOS]
        for i, t in enumerate(turns):
            enc = tok.encode(t)
            if i % 2 == 0:
                ids.append(USER)
                if i == len(turns) - 1:
                    start, offs = len(ids), enc.offsets
                ids += enc.ids
            else:
                ids += [BOT] + enc.ids + [EOS]
        ids.append(BOT)
        if len(ids) <= BLOCK or len(turns) < 3:
            return ids[:BLOCK], start, offs, turns[-1]
        turns = turns[2:]


_WORD = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|[A-Za-z']+")


def number_spans(text):
    """文中の数 (数字 / 数詞の連なり) を [(開始, 終了, 値)] で返す。"""
    ms = list(_WORD.finditer(text))
    out, i = [], 0
    while i < len(ms):
        w = ms[i].group().lower()
        if ms[i].group()[0].isdigit():
            out.append((ms[i].start(), ms[i].end(), Fraction(ms[i].group().replace(",", ""))))
            i += 1
        elif w in NUMBER_WORDS:
            run, k = [i], i
            while k + 1 < len(ms):
                nxt = ms[k + 1]
                if text[ms[k].end():nxt.start()].strip(" -") != "":
                    break
                nw = nxt.group().lower()
                if nw in NUMBER_WORDS:
                    run.append(k + 1); k += 1
                elif (nw == "and" and k + 2 < len(ms) and ms[k + 2].group().lower() in NUMBER_WORDS
                      and ms[k].group().lower() in SCALES and text[ms[k + 1].end():ms[k + 2].start()].strip(" -") == ""):
                    run += [k + 1, k + 2]; k += 2
                else:
                    break
            v = words_to_number([ms[j].group().lower() for j in run])
            if v is not None:
                out.append((ms[run[0]].start(), ms[run[-1]].end(), Fraction(v)))
            i = k + 1
        else:
            i += 1
    return out


def op_base(h_last, h_utt, e_utt):
    """op頭の入力: 最後の位置 + 発言トークンの max/mean プーリング + 生のトークン埋め込みの max/min
    (演算子の記号のトークンが文脈に埋もれないように)。"""
    return torch.cat([h_last, h_utt.max(0).values, h_utt.mean(0), e_utt.max(0).values, e_utt.min(0).values])


class Heads(nn.Module):
    def __init__(self, d=384, vocab=16000, k=32):
        super().__init__()
        # 凍結LLMが見ていない稀なトークン (×=Ã+Ĺ, ÷=Ã+· など) を区別できるよう、op頭自身の小さなトークン埋め込みを持つ
        self.tokemb_mean = nn.EmbeddingBag(vocab, k, mode="mean")
        self.tokemb_max = nn.EmbeddingBag(vocab, k, mode="max")
        self.op = nn.Sequential(nn.Linear(5 * d + 2 * k, 256), nn.GELU(), nn.Linear(256, len(OPS)))
        self.tag = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, len(TAGS)))


def op_logits(heads, base, uids):
    """base: [N, 5d], uids: 各発言のトークンid列(tensor)のリスト。"""
    flat = torch.cat(uids)
    offs = torch.tensor([0] + [len(u) for u in uids[:-1]], device=flat.device).cumsum(0)
    return heads.op(torch.cat([base, heads.tokemb_mean(flat, offs), heads.tokemb_max(flat, offs)], 1))


def decode(op, tags, offs, text):
    """op: OPS の添字, tags: 発言トークンごとの TAGS 添字。返り値: 式の文字列 / None(計算しない) / "missing"(数が取れない)。"""
    if OPS[op] == "none":
        return None

    nums = number_spans(text)
    cls = lambda c: "d" if c.isdigit() else ("a" if c.isalpha() else "o")

    def span(tid):
        ks = [k for k, t in enumerate(tags) if t == tid]
        runs = []
        for k in ks:                       # 連続したトークンの塊に分ける
            if runs and k == runs[-1][-1] + 1:
                runs[-1].append(k)
            else:
                runs.append([k])
        for run in runs:
            a, b = min(offs[k][0] for k in run), max(offs[k][1] for k in run)
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            if a >= b:
                continue
            # 因果的な LLM は語の最初のサブワードの時点では数だと分からず取りこぼすので、同じ種類の文字(数字/英字)の境界まで広げる
            sep = lambda i: 0 < i < len(text) - 1 and ((text[i] in ",." and text[i - 1].isdigit() and text[i + 1].isdigit())
                                                       or (text[i] == "-" and text[i - 1].isalpha() and text[i + 1].isalpha()))
            while a > 0 and (cls(text[a - 1]) == cls(text[a]) != "o" or sep(a - 1) or sep(a)):   # 1,000 / 3.5 / twenty-one の区切りは語の一部
                a -= 1
            while b < len(text) and (cls(text[b]) == cls(text[b - 1]) != "o" or sep(b) or sep(b - 1)):
                b += 1
            # 数として読める塊と重なるものだけ採用 (「i」などの誤タグを捨てる)
            if any(x[0] < b and x[1] > a for x in nums):
                return text[a:b]
        return None

    s1, s2 = span(1), span(2)
    if not s1 or not s2:
        return "missing"
    return {"add": f"{s1} + {s2}", "sub": f"{s1} - {s2}", "sub_rev": f"{s2} - {s1}",
            "mul": f"{s1} * {s2}", "div": f"{s1} / {s2}"}[OPS[op]]


class Router:
    def __init__(self, dev="cpu", heads="ckpt/router_heads.pt", nohist=False):
        self.dev, self.nohist = dev, nohist   # nohist=True なら会話履歴を見ず最後の発言だけで判断する
        self.tok, self.model = load_backbone(dev)
        self.heads = Heads().to(dev)
        self.heads.load_state_dict(torch.load(heads, map_location=dev))
        self.heads.eval()

    @torch.no_grad()
    def route(self, turns):
        """turns: 最後がユーザー発言の会話。返り値: (式, 結果) か None (LLMに回す)。"""
        ids, start, offs, text = build_context(self.tok, turns[-1:] if self.nohist else turns)
        h = hidden(self.model, torch.tensor([ids], device=self.dev))[0]
        utt = h[start:start + len(offs)]
        emb = self.model.wte(torch.tensor(ids[start:start + len(offs)], device=self.dev))
        uid = torch.tensor(ids[start:start + len(offs)], device=self.dev)
        op = op_logits(self.heads, op_base(h[-1], utt, emb)[None], [uid]).argmax().item()
        tags = self.heads.tag(utt).argmax(-1).tolist()
        expr = decode(op, tags, offs, text)
        if expr is None or expr == "missing":
            return None
        return expr, solve(expr)
