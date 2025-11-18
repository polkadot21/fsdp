import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):  # x: [B,T,D]
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.hd).transpose(1, 2)  # [B,3,T,H,hd]
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B,T,H,hd]
        q = q.transpose(1, 2)  # [B,H,T,hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.hd**0.5)  # [B,H,T,T]
        att = att.softmax(dim=-1)
        out = att @ v  # [B,H,T,hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, ff_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyModel(nn.Module):
    """
    Canonical base model API:

      __init__(in_dim, dim, n_heads, ff_dim, n_layers)
      attributes: inp, blocks, out
      forward(x: [B,T,in_dim]) -> [B,T,in_dim]
    """

    def __init__(self, in_dim=512, dim=512, n_heads=8, ff_dim=2048, n_layers=4):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim, bias=False)
        self.blocks = nn.ModuleList([Block(dim, n_heads, ff_dim) for _ in range(n_layers)])
        self.out = nn.Linear(dim, in_dim, bias=False)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)
