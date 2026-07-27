"""Modality-agnostic front end.

Each recorded unit owns a learned embedding, so a session's population vector
collapses to a single token per time bin — independent of channel count, array
layout, region, or subject. The same per-unit tables drive the Poisson readout,
which keeps the model fully cross-session at both ends.
"""

import torch
from torch import nn


class PopulationTokenizer(nn.Module):
    def __init__(self, dim: int, max_units: int, graft: bool = False):
        super().__init__()
        self.embed = nn.Embedding(max_units, dim)   # input mixing weights
        self.readout = nn.Embedding(max_units, dim)  # output (log-rate) weights
        self.bias = nn.Embedding(max_units, 1)
        self.value = nn.Linear(1, dim)               # per-unit count -> token (spatial path)
        self.scale = dim ** -0.5
        nn.init.zeros_(self.bias.weight)
        # GRAFT neuron interface (opt-in): derive the per-neuron read-in gain and readout
        # direction from the neuron embedding via small MLPs, instead of using the raw
        # embedding rows (per-neuron read-in gain + per-neuron readout over a pooled
        # backbone). Read-in is 1/sqrt(N) normalized (GRAFT Eq. 5).
        self.gin = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) if graft else None
        self.gout = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) if graft else None

    def encode(self, counts, unit_ids):
        # counts [B,T,N] -> tokens [B,T,dim]; log1p tames the count dynamic range
        if self.gin is not None:
            g = self.gin(self.embed(unit_ids))                          # per-neuron read-in gain
            return torch.log1p(counts) @ g * (unit_ids.shape[0] ** -0.5)
        return torch.log1p(counts) @ self.embed(unit_ids)

    def decode(self, z, unit_ids):
        # latent [B,T,dim] -> per-unit Poisson log-rate [B,T,N]
        w = self.gout(self.readout(unit_ids)) if self.gout is not None else self.readout(unit_ids)
        return z @ w.t() * self.scale + self.bias(unit_ids).squeeze(-1)

    def encode_units(self, counts, unit_ids):
        # counts [B,T,N] -> per-unit tokens [B,T,N,dim]: unit identity + its count
        return self.embed(unit_ids) + self.value(torch.log1p(counts).unsqueeze(-1))

    def decode_units(self, tokens, unit_ids):
        # per-unit tokens [B,T,N,dim] -> per-unit log-rate [B,T,N]
        w = self.readout(unit_ids)
        return (tokens * w).sum(-1) * self.scale + self.bias(unit_ids).squeeze(-1)
