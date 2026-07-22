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
    def __init__(self, heldin, heldout=None, behavior=None, actions=None, context=None,
                 window=None, unit_ids=None, session=None, behavior_stats=None):
        # (mean, std) used to standardize behavior, so metrics can recover raw kinematics.
        self.behavior_stats = behavior_stats
        self.heldin = _trialize(torch.as_tensor(heldin, dtype=torch.float32), window)
        self.heldout = _trialize(_as_f32(heldout), window)
        self.behavior = _trialize(_as_f32(behavior), window)
        self.actions = _trialize(_as_f32(actions), window)
        self.context = _trialize(_as_f32(context), window)
        self.session = session
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
        if self.context is not None:
            item["context"] = self.context[i]
        if self.session is not None:  # per-sample so concatenated sessions keep their ids
            item["session"] = torch.as_tensor(self.session, dtype=torch.long)
        return item

    def collate(self, samples):
        batch = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        batch["unit_ids"] = self.in_ids
        if self.heldout is not None:
            batch["target_unit_ids"] = self.out_ids
        return batch


def _as_f32(x):
    return None if x is None else torch.as_tensor(x, dtype=torch.float32)


def split_trials(ds, frac, seed=0):
    """Partition a dataset's trials into two SpikeWindows (train core, selection set),
    so checkpoints and hyperparameters are chosen on data disjoint from the reported split."""
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))
    cut = int(len(ds) * frac)

    def carve(idx):
        pick = lambda t: t[idx] if t is not None else None
        return SpikeWindows(ds.heldin[idx], pick(ds.heldout), pick(ds.behavior),
                            actions=pick(ds.actions), unit_ids=ds.in_ids,
                            behavior_stats=ds.behavior_stats)
    return carve(perm[:cut]), carve(perm[cut:])
