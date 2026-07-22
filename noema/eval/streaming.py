"""Online causal decoder.

Feed one bin of spikes at a time and read out the current behavior estimate.
A rolling window of recent activity is re-encoded each step, so decoding stays
causal — the substrate for closed-loop control and streaming benchmarks (FALCON).
"""

import torch


class StreamingDecoder:
    def __init__(self, model, unit_ids, window=50, device=None, latent_transform=None):
        self.model = model.eval()
        self.window = window
        self.device = device or next(model.parameters()).device
        self.unit_ids = torch.as_tensor(unit_ids, device=self.device)
        self.buffer = None
        # optional latent-space alignment applied to the current state before decoding
        # (e.g. matching an unseen session's latent distribution to the training one)
        self.latent_transform = latent_transform

    def reset(self, batch_size=1):
        self.buffer = torch.zeros(batch_size, self.window, self.unit_ids.numel(), device=self.device)

    @torch.no_grad()
    def step(self, observations):
        """observations: [batch, n_channels] spike counts for the current bin."""
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if self.buffer is None or self.buffer.size(0) != obs.size(0):
            self.reset(obs.size(0))
        self.buffer = torch.cat([self.buffer[:, 1:], obs[:, None]], dim=1)  # roll in the new bin
        z = self.model.encode(self.buffer, self.unit_ids)
        zl = z[:, -1]
        if self.latent_transform is not None:
            zl = self.latent_transform(zl)
        return self.model.behavior(zl)
