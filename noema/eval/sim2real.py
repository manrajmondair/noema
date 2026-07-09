"""Sim-to-real: train a decoder in imagination.

Roll the world model forward under random action plans, sample spikes from the
imagined rates, and fit a fresh independent decoder (ridge) on that synthetic
data alone. Scoring it on real held-out recordings measures whether the learned
simulator can stand in for scarce neural data.
"""

import torch

from ..data.dataset import SpikeWindows
from ..sim import imagine
from .baselines import ridge_velocity


@torch.no_grad()
def decoder_in_imagination(world_model, unit_ids, seed_ds, real_val_ds,
                           episodes=128, horizon=20, device=None):
    device = device or next(world_model.parameters()).device
    world_model.eval()
    seeds = seed_ds.heldin[:episodes].to(device)
    seed_len = seeds.size(1) // 2
    action_dim = seed_ds.actions.size(-1)

    actions = torch.randn(seeds.size(0), seed_len + horizon, action_dim, device=device)
    rates, behavior = imagine(world_model, seeds[:, :seed_len], unit_ids.to(device),
                              actions[:, seed_len:], seed_actions=actions[:, :seed_len])
    imagined = SpikeWindows(torch.poisson(rates).cpu(), behavior=behavior.cpu())
    return {"sim2real_r2": ridge_velocity(imagined, real_val_ds)}
