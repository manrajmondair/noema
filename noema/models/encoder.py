"""Rotary temporal transformer. Bidirectional for representation learning,
causal when the world model rolls out in time."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def rotary_tables(dim: int, seq: int, device, base: float = 10_000.0):
    inv = 1.0 / base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    ang = torch.outer(torch.arange(seq, device=device, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()  # each [seq, dim/2]


def apply_rotary(x, cos, sin):  # x: [B, H, T, D]
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, causal):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(o.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mult: int = 4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))

    def forward(self, x, cos, sin, causal):
        x = x + self.attn(self.norm1(x), cos, sin, causal)
        return x + self.mlp(self.norm2(x))


class TemporalEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head_dim = dim // heads

    def forward(self, x, causal: bool = False):
        cos, sin = rotary_tables(self.head_dim, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, cos, sin, causal)
        return self.norm(x)


class SpatioTemporalEncoder(nn.Module):
    """Factorized attention over time and over units (STNDT-style). Input is
    per-unit tokens [B,T,N,D]: temporal blocks attend over T with rotary; spatial
    blocks attend over the unit set N with no positional code (units are a set)."""

    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.temporal = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.spatial = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head_dim = dim // heads

    def forward(self, x):  # x: [B, T, N, D]
        B, T, N, D = x.shape
        cos_t, sin_t = rotary_tables(self.head_dim, T, x.device)
        one = torch.ones(N, self.head_dim // 2, device=x.device)   # identity rotary = no positional
        zero = torch.zeros(N, self.head_dim // 2, device=x.device)
        # per-unit activations are large; checkpoint each block to keep memory bounded
        run = ((lambda blk, *a: checkpoint(blk, *a, use_reentrant=False)) if self.training
               else (lambda blk, *a: blk(*a)))
        for tblk, sblk in zip(self.temporal, self.spatial):
            xt = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
            x = run(tblk, xt, cos_t, sin_t, False).reshape(B, N, T, D).permute(0, 2, 1, 3)
            xs = x.reshape(B * T, N, D)
            x = run(sblk, xs, one, zero, False).reshape(B, T, N, D)
        return self.norm(x)
