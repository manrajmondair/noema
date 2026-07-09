"""Binned-spike dataset with a fixed held-in/held-out neuron split.

Held-out units are hidden from the encoder and supervised only through the
co-smoothing target, matching the Neural Latents Benchmark protocol. Works for
trialized data ([trials, time, units]) and for continuous data sliced into
fixed windows ([time, units]).
"""

import torch
from torch.utils.data import Dataset


def _trialize(x, window):
    if x is None:
        return None
    if x.dim() == 2:  # continuous -> non-overlapping windows
        n = x.size(0) // window
        return x[: n * window].reshape(n, window, x.size(1))
    return x if window is None else x[:, :window]


class SpikeWindows(Dataset):
    def __init__(self, heldin, heldout=None, behavior=None, actions=None, window=None,
                 unit_ids=None):
        self.heldin = _trialize(torch.as_tensor(heldin, dtype=torch.float32), window)
        self.heldout = _trialize(_as_f32(heldout), window)
        self.behavior = _trialize(_as_f32(behavior), window)
        self.actions = _trialize(_as_f32(actions), window)
        n_in = self.heldin.size(-1)
        n_out = self.heldout.size(-1) if self.heldout is not None else 0
        # Explicit ids place a session's units in a shared global embedding table.
        base = torch.arange(n_in) if unit_ids is None else torch.as_tensor(unit_ids)
        self.in_ids = base
        self.out_ids = torch.arange(n_out) + (base.max() + 1 if n_in else 0)

    def __len__(self):
        return self.heldin.size(0)

    def __getitem__(self, i):
        item = {"counts": self.heldin[i]}
        if self.heldout is not None:
            item["target_counts"] = self.heldout[i]
        if self.behavior is not None:
            item["behavior"] = self.behavior[i]
        if self.actions is not None:
            item["actions"] = self.actions[i]
        return item

    def collate(self, samples):
        batch = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        batch["unit_ids"] = self.in_ids
        if self.heldout is not None:
            batch["target_unit_ids"] = self.out_ids
        return batch


def _as_f32(x):
    return None if x is None else torch.as_tensor(x, dtype=torch.float32)
