"""Multi-session pretraining.

Places each session's units in a disjoint slice of one global embedding table,
tags it with a session label (which drives the adversarial invariance term), and
returns a shuffled stream of collated batches drawn across all sessions — the
Stage-1 corpus for a cross-subject foundation model.
"""

import torch
from torch.utils.data import DataLoader


def combine_sessions(sessions, batch_size=64, seed=0):
    """Return (batches, max_units, n_sessions) for a list of SpikeWindows.

    Each session is remapped onto its own id range so the shared model can tell
    populations apart; heldin/heldout splits and behavior/action fields are kept.
    """
    offset, batches = 0, []
    for label, ds in enumerate(sessions):
        n_in, n_out = ds.in_ids.numel(), ds.out_ids.numel()
        ds.in_ids = torch.arange(n_in) + offset
        ds.out_ids = torch.arange(n_out) + offset + n_in
        ds.session = label
        offset += n_in + n_out
        batches += list(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                   collate_fn=ds.collate, drop_last=True))

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[i] for i in order], offset, len(sessions)
