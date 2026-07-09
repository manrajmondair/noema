"""Synthetic population activity for wiring tests and quick local iteration.

Firing rates and behavior are both linear readouts of a smooth low-dimensional
latent, so a correctly wired model should recover them from Poisson spike counts.
"""

import torch


def synthetic_batch(batch=16, steps=50, units=60, latent=4, behavior_dim=2, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *s: torch.rand(*s, generator=g, device=device)
    randn = lambda *s: torch.randn(*s, generator=g, device=device)

    t = torch.linspace(0, 6.2832, steps, device=device)
    freq = rand(latent) * 2 + 0.5
    phase = rand(batch, latent) * 6.2832
    z = torch.sin(t[None, :, None] * freq[None, None] + phase[:, None, :])  # [B,T,latent]

    rates = torch.exp(z @ (randn(latent, units) * 0.8) - 1.0)
    counts = torch.poisson(rates, generator=g)
    behavior = z @ randn(latent, behavior_dim)
    unit_ids = torch.arange(units, device=device)
    return counts, unit_ids, behavior
