"""Action-conditioned forward model.

Predicts the next latent from the causal history of latents and actions, so it
doubles as an autoregressive rollout engine for the closed-loop simulator.
"""

from torch import nn

from .encoder import TemporalEncoder


class WorldModel(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, action_dim: int = 0):
        super().__init__()
        self.action = nn.Linear(action_dim, dim) if action_dim else None
        self.core = TemporalEncoder(dim, depth, heads)
        self.head = nn.Linear(dim, dim)

    def forward(self, z, actions=None):
        if self.action is not None and actions is not None:
            z = z + self.action(actions)
        return self.head(self.core(z, causal=True))  # next-step latent at each position
