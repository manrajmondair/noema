from torch import nn


class BehaviorHead(nn.Module):
    """Decodes kinematics (e.g. cursor velocity) from the latent state."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim))

    def forward(self, z):
        return self.net(z)
