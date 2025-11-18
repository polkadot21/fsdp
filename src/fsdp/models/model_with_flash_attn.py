#############################
# Flash-attn-backed TinyModel drop-in.
#
# Requirements:
#   - flash-attn installed (Linux + CUDA + supported PyTorch/Python)
#   - GPU (Ampere+)
#
# API is identical to TinyModel:
#   FlashTinyModel(in_dim, dim, n_heads, ff_dim, n_layers)
#   .inp, .blocks, .out
#   forward(x: [B,T,in_dim]) -> [B,T,in_dim]
#############################

import torch
from torch import nn

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func  # ignore
except ImportError as e:
    raise ImportError(
        "flash-attn is not installed. Install with `pip install flash-attn` "
        "or install the 'flash' extra for this project."
    ) from e


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.num_heads = n_heads
        self.head_dim = dim // n_heads

        self.to_query = nn.Linear(dim, dim, bias=True)
        self.to_key = nn.Linear(dim, dim, bias=True)
        self.to_value = nn.Linear(dim, dim, bias=True)

        # per-head normalization
        self.query_norm = nn.RMSNorm(self.head_dim)
        self.key_norm = nn.RMSNorm(self.head_dim)

        self.out_layer = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        Returns: [B, T, D]

        Internally:
          1. Flatten to [N, D] with N = B*T
          2. Compute Q/K/V and pack to [N, 3, H, Hd]
          3. Build cu_seqlens for equal-length sequences
          4. Call flash_attn_varlen_qkvpacked_func
          5. Reshape back to [B, T, D]
        """
        B, T, D = x.shape
        assert D == self.dim, f"Expected last dim={self.dim}, got {D}"

        # Flatten batch+time into a single token dimension
        x_flat = x.reshape(B * T, D)

        # Linear projections
        q = self.to_query(x_flat)  # [N, D]
        k = self.to_key(x_flat)  # [N, D]
        v = self.to_value(x_flat)  # [N, D]

        # Reshape to [N, H, Hd] and apply per-head RMSNorm
        N = B * T
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_heads, self.head_dim)
        v = v.view(N, self.num_heads, self.head_dim)

        q = self.query_norm(q).type_as(q)
        k = self.key_norm(k).type_as(k)

        # Pack into [N, 3, H, Hd]
        qkv = torch.stack([q, k, v], dim=1)  # [N, 3, H, Hd]

        # cu_seqlens: equal-length sequences of length T
        # shape = [B+1], values = [0, T, 2T, ..., B*T]
        cu_seqlens = torch.arange(0, (B + 1) * T, step=T, device=x.device, dtype=torch.int32)
        max_seqlen = T

        # FlashAttention varlen qkvpacked interface.
        # Output shape: [N, H, Hd]
        out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen)

        # Merge heads back into D and restore [B, T, D]
        out = out.flatten(-2, -1)  # [N, D]
        out = self.out_layer(out)
        out = out.view(B, T, D)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


class FlashBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = MultiheadSelfAttention(dim, n_heads)

        self.feed_forward_norm = nn.RMSNorm(dim)
        self.feed_forward = FeedForward(dim, ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Model(nn.Module):
    """
    FlashAttention-backed version of TinyModel.

    Signature and attributes are identical to TinyModel:
      __init__(in_dim, dim, n_heads, ff_dim, n_layers)
      .inp, .blocks, .out
      forward(x: [B,T,in_dim]) -> [B,T,in_dim]
    """

    def __init__(self, in_dim: int, dim: int, n_heads: int, ff_dim: int, n_layers: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([FlashBlock(dim, n_heads, ff_dim) for _ in range(n_layers)])
        self.out = nn.Linear(dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)
