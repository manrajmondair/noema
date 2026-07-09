"""Closed-loop neural simulator.

Given a seed window of real spikes and a plan of actions, roll the world model
forward in its own latent space and decode imagined firing rates and behavior.
This is the substrate for training and stress-testing decoders in imagination.
"""

import torch


@torch.no_grad()
def imagine(model, seed_counts, unit_ids, future_actions, seed_actions=None):
    model.eval()
    z = model.encode(seed_counts, unit_ids)
    a = seed_actions
    steps = []
    for t in range(future_actions.size(1)):
        nxt = model.world(z, a)[:, -1:]
        z = torch.cat([z, nxt], dim=1)
        if a is not None:
            a = torch.cat([a, future_actions[:, t : t + 1]], dim=1)
        steps.append(nxt)

    future = torch.cat(steps, dim=1)
    rates = model.tokenizer.decode(future, unit_ids).exp()
    behavior = model.behavior(future) if model.behavior is not None else None
    return rates, behavior
