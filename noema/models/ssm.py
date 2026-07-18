"""Diagonal complex state-space encoder (S5/LRU-style) — an evidence-backed alternative
to the temporal transformer for NLB co-bps (state-space models reach the SOTA band).

Each layer is a diagonal linear recurrence x_t = A x_{t-1} + B u_t, y_t = Re(C x_t) + D u_t
with A = exp(-exp(nu) + i*theta) (|A|<1, stable). Computed in the materialized-kernel form
x_t = sum_{k<=t} A^{t-k} (B u_k), a causal convolution — fully parallel over time (no scan),
verified equal to the sequential recurrence in tests.
"""
import torch
from torch import nn


class DiagonalSSM(nn.Module):
    def __init__(self, dim: int, state: int = 128):
        super().__init__()
        # stable long-memory init: |A| ~ U[0.9, 0.999], small phase (slow modes)
        r = torch.rand(state)
        self.nu = torch.nn.Parameter(torch.log(-torch.log(0.9 + 0.099 * r)))  # |A| = exp(-exp(nu))
        self.theta = torch.nn.Parameter(torch.rand(state) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(state, dim) / dim ** 0.5)
        self.C = torch.nn.Parameter(torch.randn(dim, state, 2) / state ** 0.5)  # real/imag
        self.D = torch.nn.Parameter(torch.ones(dim))

    def _log_A(self):
        return -torch.exp(self.nu) + 1j * self.theta  # [state] complex

    def forward(self, u):  # u [B,T,dim] -> [B,T,dim]
        T = u.size(1)
        j = torch.arange(T, device=u.device)
        kernel = torch.exp(j[:, None] * self._log_A()[None, :])         # [T,state] = A^j
        Bu = u.to(torch.complex64) @ self.B.t().to(torch.complex64)      # [B,T,state]
        d = (j[:, None] - j[None, :]).clamp(min=0)                       # [T,T] = t-k
        K = torch.where((j[:, None] >= j[None, :])[..., None], kernel[d], torch.zeros_like(kernel[0, 0]))
        x = torch.einsum("tks,bks->bts", K, Bu)                         # [B,T,state] complex
        C = torch.view_as_complex(self.C.contiguous())                  # [dim,state]
        return torch.einsum("bts,ds->btd", x, C).real + self.D * u

    @torch.no_grad()
    def sequential(self, u):  # reference recurrence for the correctness test
        B, T, _ = u.shape
        A = torch.exp(self._log_A())
        Bu = u.to(torch.complex64) @ self.B.t().to(torch.complex64)
        C = torch.view_as_complex(self.C.contiguous())
        x = torch.zeros(B, self.B.shape[0], dtype=torch.complex64, device=u.device)
        out = []
        for t in range(T):
            x = A * x + Bu[:, t]
            out.append((x @ C.t()).real + self.D * u[:, t])
        return torch.stack(out, 1)


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
    Causal by construction (the kernel is lower-triangular), so `causal` is ignored."""

    def __init__(self, dim: int, depth: int, heads: int, state: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList(SSMBlock(dim, state) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, causal: bool = False):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
