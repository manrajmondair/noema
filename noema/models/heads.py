import torch
import torch.nn.functional as F
from torch import nn


class FiLMReadout(nn.Module):
    """Nonlinear per-unit co-smoothing readout. A shared nonlinear feature of the latent z
    is modulated per held-out unit (FiLM, using that unit's readout embedding as the scale),
    then projected to a log-rate. Strictly more expressive than the linear z @ readout dot-
    product. Held-out only (small unit set), so the [B,T,M,D] modulation stays memory-bounded."""

    def __init__(self, dim: int):
        super().__init__()
        self.feat = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.out = nn.Linear(dim, 1)

    def forward(self, z, unit_embeds, unit_bias):
        # z [B,T,D], unit_embeds [M,D] (per-unit FiLM scale), unit_bias [M] -> log-rate [B,T,M]
        h = self.feat(z)                                  # [B,T,D] shared nonlinear feature
        mod = h[:, :, None, :] * unit_embeds[None, None]  # [B,T,M,D] per-unit modulation
        return self.out(mod).squeeze(-1) + unit_bias


class AttentionPool(nn.Module):
    """Pool per-unit tokens [B,T,N,D] into the shared latent [B,T,D] with a learned
    query attending over the unit set — a content-weighted summary that keeps which
    units drive the population state, unlike a mean that weights every unit equally.
    The co-smoothing readout decodes held-out rates from this latent, so a sharper
    pool feeds the co-bps metric directly."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Parameter(torch.randn(dim) * dim ** -0.5)
        self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, tokens):  # [B,T,N,D] -> [B,T,D]
        B, T, N, D = tokens.shape
        x = self.norm(tokens).reshape(B * T, N, D)
        q = self.query.view(1, self.heads, 1, self.head_dim).expand(B * T, -1, -1, -1)
        kv = self.kv(x).view(B * T, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, kv[0], kv[1])  # [B*T, heads, 1, head_dim]
        return self.proj(o.reshape(B * T, D)).reshape(B, T, D)


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
        # split the head axis from M *before* moving heads to the front, else the
        # flat [M, heads*head_dim] buffer misassigns each head-slice to the wrong unit
        q = self.q(self.qn(queries)).view(M, self.heads, self.head_dim).permute(1, 0, 2)
        q = q.unsqueeze(0).expand(B * T, -1, -1, -1)
        kv = self.kv(self.kn(tokens)).view(B * T, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, kv[0], kv[1])          # [B*T, heads, M, d]
        return self.out(o.transpose(1, 2).reshape(B, T, M, D)).squeeze(-1)
