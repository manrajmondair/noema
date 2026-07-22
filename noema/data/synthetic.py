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


class LinearSpikeSystem:
    """Action-driven latent dynamics emitting Poisson spikes.

    z_{t+1} = decay * z_t + a_t B, with firing rates and behavior read out
    linearly. Used to check the world model against a known simulator: it can
    generate training data and roll ground truth forward from any latent state.
    """

    def __init__(self, units=60, latent=6, action_dim=2, behavior_dim=2, decay=0.9,
                 seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        randn = lambda *s: torch.randn(*s, generator=g, device=device)
        self.decay, self.latent, self.action_dim, self.device = decay, latent, action_dim, device
        self.B = randn(action_dim, latent) * 0.5
        self.W = randn(latent, units) * 0.7
        self.Wb = randn(latent, behavior_dim)
        self._g = g

    def rollout(self, actions, z0=None):
        z = torch.zeros(actions.size(0), self.latent, device=self.device) if z0 is None else z0
        states = []
        for t in range(actions.size(1)):
            z = self.decay * z + actions[:, t] @ self.B
            states.append(z)
        z = torch.stack(states, dim=1)
        # Bounded, realistic per-bin firing (mean ~1-2 counts); the cap stops the
        # exp tail from producing counts no real neuron would fire.
        rates = (0.4 * (z @ self.W)).clamp(max=3.0).exp()
        return z, rates, z @ self.Wb

    def sample(self, batch=64, steps=50):
        actions = torch.randn(batch, steps, self.action_dim, generator=self._g, device=self.device)
        _, rates, behavior = self.rollout(actions)
        counts = torch.poisson(rates, generator=self._g)
        unit_ids = torch.arange(rates.size(-1), device=self.device)
        return counts, unit_ids, actions, behavior


class MultiSessionSystem:
    """One latent dynamics observed by a different population each session.

    Dynamics, action drive, and behavior read-out are shared; only the unit
    projection changes per session, so a model can transfer across sessions only
    by routing new units into the shared latent space.
    """

    def __init__(self, sessions=4, units=40, latent=6, action_dim=2, behavior_dim=2,
                 decay=0.9, seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        randn = lambda *s: torch.randn(*s, generator=g, device=device)
        self.units, self.action_dim, self.latent, self.decay = units, action_dim, latent, decay
        self.device, self._g = device, g
        self.B = randn(action_dim, latent) * 0.5
        self.Wb = randn(latent, behavior_dim)
        self.W = [randn(latent, units) * 0.7 for _ in range(sessions)]  # per-session units

    def unit_ids(self, session):
        return torch.arange(session * self.units, (session + 1) * self.units, device=self.device)

    def sample(self, session, batch=64, steps=50):
        actions = torch.randn(batch, steps, self.action_dim, generator=self._g, device=self.device)
        z = torch.zeros(batch, self.latent, device=self.device)
        states = []
        for t in range(steps):
            z = self.decay * z + actions[:, t] @ self.B
            states.append(z)
        z = torch.stack(states, dim=1)
        rates = (0.4 * (z @ self.W[session])).clamp(max=3.0).exp()  # realistic firing
        counts = torch.poisson(rates, generator=self._g)
        return counts, self.unit_ids(session), actions, z @ self.Wb


class SensorySystem:
    """External stimulus drives the neural latent. Used to check that the model can
    predict a population's response to the outside world (an encoding model)."""

    def __init__(self, units=60, latent=6, stim_dim=8, behavior_dim=2, decay=0.9,
                 seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        randn = lambda *s: torch.randn(*s, generator=g, device=device)
        self.stim_dim, self.latent, self.decay, self.device = stim_dim, latent, decay, device
        self.S = randn(stim_dim, latent) * 0.6
        self.W = randn(latent, units) * 0.7
        self.Wb = randn(latent, behavior_dim)
        self._g = g

    def response(self, stim):
        z = torch.zeros(stim.size(0), self.latent, device=self.device)
        states = []
        for t in range(stim.size(1)):
            z = self.decay * z + stim[:, t] @ self.S
            states.append(z)
        z = torch.stack(states, dim=1)
        # Keep firing in a realistic per-bin range (~single-digit counts) so the
        # Poisson likelihood is well conditioned, as it is for real recordings.
        return torch.exp(0.3 * (z @ self.W) - 1.5), z @ self.Wb

    def sample(self, batch=64, steps=50):
        stim = torch.randn(batch, steps, self.stim_dim, generator=self._g, device=self.device)
        rates, behavior = self.response(stim)
        counts = torch.poisson(rates, generator=self._g)
        return counts, torch.arange(rates.size(-1), device=self.device), stim, behavior
