"""Domain-adversarial session invariance.

A classifier tries to name the session a latent came from; a gradient-reversal
layer flips its gradient into the encoder, which is thereby pushed to represent
the shared dynamics without session-identifying detail. This is what lets a new
session route into the frozen latent instead of landing in its own private frame.
"""

import torch
from torch import nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


def grad_reverse(x, weight=1.0):
    return _GradReverse.apply(x, weight)


class SessionAdversary(nn.Module):
    def __init__(self, dim, sessions, weight=1.0):
        super().__init__()
        self.weight = weight
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, sessions))

    def forward(self, z, session):
        # pool over time, then predict session identity through the reversal
        pooled = grad_reverse(z.mean(dim=1), self.weight)
        return nn.functional.cross_entropy(self.net(pooled), session)
