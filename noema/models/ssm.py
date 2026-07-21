"""Diagonal complex state-space encoder (S5/LRU-style) — an evidence-backed alternative
to the temporal transformer for NLB co-bps (state-space models reach the SOTA band).

Each layer is a diagonal linear recurrence x_t = A x_{t-1} + B u_t, y_t = Re(C x_t) + D u_t
with A = exp(-exp(nu) + i*theta) (|A|<1, stable). Computed in the materialized-kernel form
x_t = sum_{k<=t} A^{t-k} (B u_k), a causal convolution — fully parallel over time (no scan),
verified equal to the sequential recurrence in tests.
"""
import math

import torch
from torch import nn


class _Dir(nn.Module):
    """One diagonal-SSM direction: x_t = A x_{t-1} + B u_t, returns Re(C x_t) (no skip).
    Computed as a causal convolution with kernel A^j (materialized, fully parallel)."""

    def __init__(self, dim: int, state: int):
        super().__init__()
        # S4D-Lin-style STRUCTURED init: oscillation frequencies (Im log A) spread evenly
        # across [0, pi] for full spectral coverage, magnitudes |A| log-spaced in [0.9,0.999]
        # for a range of memory timescales. A naive random diagonal (near-zero random phase,
        # uniform |A|) covers almost no frequencies — the documented reason a plain diagonal
        # SSM underperforms S4D/S5 (published S5 ~0.382 vs our random-init 0.333).
        n = torch.arange(state, dtype=torch.float32)
        self.theta = nn.Parameter(math.pi * n / max(state - 1, 1))               # Im log A: 0..pi even
        self.nu = nn.Parameter(torch.log(-torch.log(torch.linspace(0.9, 0.999, state))))  # |A| range
        self.B = nn.Parameter(torch.randn(state, dim) / dim ** 0.5)
        self.C = nn.Parameter(torch.randn(dim, state, 2) / state ** 0.5)

    def log_A(self):
        return -torch.exp(self.nu) + 1j * self.theta

    def forward(self, u):
        T = u.size(1)
        j = torch.arange(T, device=u.device)
        kernel = torch.exp(j[:, None] * self.log_A()[None, :])          # [T,state] = A^j
        Bu = u.to(torch.complex64) @ self.B.t().to(torch.complex64)      # [B,T,state]
        d = (j[:, None] - j[None, :]).clamp(min=0)
        K = torch.where((j[:, None] >= j[None, :])[..., None], kernel[d], torch.zeros_like(kernel[0, 0]))
        x = torch.einsum("tks,bks->bts", K, Bu)
        return torch.einsum("bts,ds->btd", x, torch.view_as_complex(self.C.contiguous())).real

    @torch.no_grad()
    def sequential(self, u):
        A = torch.exp(self.log_A())
        Bu = u.to(torch.complex64) @ self.B.t().to(torch.complex64)
        C = torch.view_as_complex(self.C.contiguous())
        x = torch.zeros(u.size(0), self.B.shape[0], dtype=torch.complex64, device=u.device)
        out = []
        for t in range(u.size(1)):
            x = A * x + Bu[:, t]
            out.append((x @ C.t()).real)
        return torch.stack(out, 1)


class DiagonalSSM(nn.Module):
    """Diagonal state-space layer. Bidirectional by default: a forward (past-context) and a
    backward (future-context) direction are summed, so the encoder sees the whole trial —
    the co-bps encoder is not autoregressive, so full context helps (like the transformer)."""

    def __init__(self, dim: int, state: int = 128, bidirectional: bool = True):
        super().__init__()
        self.fwd = _Dir(dim, state)
        self.bwd = _Dir(dim, state) if bidirectional else None
        self.D = nn.Parameter(torch.ones(dim))

    def forward(self, u):  # [B,T,dim] -> [B,T,dim]
        y = self.fwd(u)
        if self.bwd is not None:
            y = y + self.bwd(u.flip(1)).flip(1)
        return y + self.D * u

    @torch.no_grad()
    def sequential(self, u):  # reference for the correctness test (matches forward)
        y = self.fwd.sequential(u)
        if self.bwd is not None:
            y = y + self.bwd.sequential(u.flip(1)).flip(1)
        return y + self.D * u


class SSMBlock(nn.Module):
    def __init__(self, dim: int, state: int, mult: int = 4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ssm = DiagonalSSM(dim, state)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))

    def forward(self, x):
        x = x + self.proj(torch.nn.functional.gelu(self.ssm(self.norm1(x))))
        return x + self.mlp(self.norm2(x))


class SSMEncoder(nn.Module):
    """Drop-in for TemporalEncoder: same (x, causal) call, [B,T,dim] -> [B,T,dim].
    With hybrid=True, odd layers are bidirectional-attention blocks — combining the
    state-space temporal dynamics with attention over time (each buys a different bias)."""

    def __init__(self, dim: int, depth: int, heads: int, state: int = 128, hybrid: bool = False):
        super().__init__()
        from .encoder import Block
        self.hybrid = hybrid
        self.blocks = nn.ModuleList(
            (Block(dim, heads) if (hybrid and i % 2 == 1) else SSMBlock(dim, state)) for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head_dim = dim // heads

    def forward(self, x, causal: bool = False):
        cos = sin = None
        if self.hybrid:
            from .encoder import rotary_tables
            cos, sin = rotary_tables(self.head_dim, x.size(1), x.device)
        for block in self.blocks:
            x = block(x) if hasattr(block, "ssm") else block(x, cos, sin, causal)  # SSMBlock vs attention Block
        return self.norm(x)
