"""Sensory coupling: brain ⊗ world.

An external stimulus stream is mapped into the same neural latent the spike
encoder uses, and read out through the same per-unit decoder. Sharing the readout
forces stimulus and neural activity into a common frame, which is what turns the
model into an encoding model — a digital twin that predicts a population's
response to the outside world.

Stimulus-to-response is temporal filtering, so a causal dilated convolution stack
carries it: each layer widens the receptive field while never seeing the future.
"""

import torch.nn.functional as F
from torch import nn


class SensoryEncoder(nn.Module):
    def __init__(self, context_dim, dim, depth, kernel=5):
        super().__init__()
        self.proj = nn.Linear(context_dim, dim)
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, kernel, dilation=2**i) for i in range(depth))
        self.pads = [(kernel - 1) * 2**i for i in range(depth)]
        self.norm = nn.LayerNorm(dim)

    def forward(self, context):
        x = self.proj(context).transpose(1, 2)  # [B, dim, T]
        for conv, pad in zip(self.convs, self.pads):
            x = x + F.gelu(conv(F.pad(x, (pad, 0))))  # left pad only -> causal
        return self.norm(x.transpose(1, 2))
