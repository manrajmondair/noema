"""Sim-to-real: train a decoder in imagination.

Roll the world model forward from real seed windows, sample spikes from the
imagined rates, and fit a fresh independent decoder (ridge) on that synthetic
data alone. Scoring it on real held-out recordings measures whether the learned
simulator can stand in for scarce neural data.
"""

import torch

from ..data.dataset import SpikeWindows
from ..sim import imagine


@torch.no_grad()
def imagine_windows(model, unit_ids, seeds, horizon, labels=None, samples=1,
                    actions=None, batch=256, device=None):
    """Imagined spike windows rolled out from real seed windows.

    Unconditioned when `actions` is None: spontaneous recordings such as FALCON have
    no observed control input, so the rollout is driven only by the seed's latent
    state and the model's own dynamics.

    Labels default to the behavior head read off the *predicted* latents, so no real
    kinematics are consumed. Passing `labels` substitutes the true kinematics of the
    same bins instead, which isolates spike-generation quality from label quality.
    """
    device = device or next(model.parameters()).device
    model.eval()
    ids = unit_ids.to(device)
    seed_len = seeds.size(1)
    rates, beh = [], []
    for i in range(0, seeds.size(0), batch):
        s = seeds[i:i + batch].to(device)
        seed_a = future_a = None
        if actions is None:
            future_a = torch.zeros(s.size(0), horizon, 0, device=device)
        else:
            a = actions[i:i + batch].to(device)
            seed_a, future_a = a[:, :seed_len], a[:, seed_len:seed_len + horizon]
        rate, b = imagine(model, s, ids, future_a, seed_actions=seed_a)
        rates.append(rate.cpu())
        if b is not None:
            beh.append(b.cpu())
    rates = torch.cat(rates)
    behavior = labels if labels is not None else (torch.cat(beh) if beh else None)
    # Sample on CPU: torch.poisson has no MPS kernel, and the ridge solves on CPU anyway.
    # Repeating before sampling draws independent spike trains from the same rate path.
    counts = torch.poisson(rates.repeat(samples, 1, 1))
    return SpikeWindows(counts, behavior=None if behavior is None else behavior.repeat(samples, 1, 1))


