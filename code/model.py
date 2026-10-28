# Claude Code により作成 (2026-10-04) — T-023: 小型GPT (環境変数 MIA_NLAYER / MIA_NHEAD / MIA_NEMBD で大きさを指定。MIA2は 10 / 8 / 512)
import os
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 5000
    block_size: int = 128
    n_layer: int = int(os.environ.get("MIA_NLAYER", 8))     # 環境変数で大きさを切り替える(Bモデル: 10 / 8 / 512)
    n_head: int = int(os.environ.get("MIA_NHEAD", 6))
    n_embd: int = int(os.environ.get("MIA_NEMBD", 384))
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.out = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        self.nh, self.dp = c.n_head, c.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.nh, C // self.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dp if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.lnf = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("out.weight"):
                nn.init.normal_(p, std=0.02 / (2 * c.n_layer) ** 0.5)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, pos=None):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device) if pos is None else pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40, eos=None):
        for _ in range(n):
            logits, _ = self(idx[:, -self.c.block_size:])
            logits = logits[:, -1] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if eos is not None and nxt.item() == eos:
                break
        return idx
