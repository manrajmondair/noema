"""Modality-agnostic front end.

Each recorded unit owns a learned embedding, so a session's population vector
collapses to a single token per time bin — independent of channel count, array
layout, region, or subject. The same per-unit tables drive the Poisson readout,
which keeps the model fully cross-session at both ends.
"""

import torch
from torch import nn


class PopulationTokenizer(nn.Module):
    def __init__(self, dim: int, max_units: int):
        super().__init__()
        self.embed = nn.Embedding(max_units, dim)   # input mixing weights
        self.readout = nn.Embedding(max_units, dim)  # output (log-rate) weights
        self.bias = nn.Embedding(max_units, 1)
        self.scale = dim ** -0.5
        nn.init.zeros_(self.bias.weight)

    def encode(self, counts, unit_ids):
        # counts [B,T,N] -> tokens [B,T,dim]; log1p tames the count dynamic range
        return torch.log1p(counts) @ self.embed(unit_ids)

    def decode(self, z, unit_ids):
        # latent [B,T,dim] -> per-unit Poisson log-rate [B,T,N]
        w = self.readout(unit_ids)
        return z @ w.t() * self.scale + self.bias(unit_ids).squeeze(-1)
