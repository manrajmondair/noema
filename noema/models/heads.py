import torch
import torch.nn.functional as F
from torch import nn


class BehaviorHead(nn.Module):
    """Decodes kinematics (e.g. cursor velocity) from the latent state."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim))

    def forward(self, z):
        return self.net(z)


class CrossReadout(nn.Module):
    """Co-smoothing readout for the spatial model: each held-out unit's embedding
    attends over the observed population's per-unit tokens to predict its log-rate.
    Uses the per-unit structure the spatial encoder learns, instead of a linear
    readout of a pooled latent."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qn, self.kn = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, 1)

    def forward(self, tokens, queries):
        # tokens [B,T,N,D] (observed units), queries [M,D] (held-out unit embeds) -> [B,T,M]
        B, T, N, D = tokens.shape
        M = queries.size(0)
        q = self.q(self.qn(queries)).view(1, self.heads, M, self.head_dim).expand(B * T, -1, -1, -1)
        kv = self.kv(self.kn(tokens)).view(B * T, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, kv[0], kv[1])          # [B*T, heads, M, d]
        return self.out(o.transpose(1, 2).reshape(B, T, M, D)).squeeze(-1)
