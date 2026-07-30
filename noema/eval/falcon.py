"""FALCON benchmark integration (falcon-challenge).

`load_falcon` reuses the challenge's NWB reader for training data; `make_decoder`
wraps a trained model in the streaming BCIDecoder interface used by the evaluator.
Covers the continuous-kinematic tasks (h1/m1/m2); h2/b1 use different targets.
"""

import re

import numpy as np
import torch

from ..data.dataset import SpikeWindows
from .streaming import StreamingDecoder


def session_id(path):
    """The recording's session key, matched wherever it sits in the filename.

    A fixed offset into the name is what this used to do, and it landed mid-word: it
    kept `ses-` on the calib side by five characters of margin and would have stopped
    matching on any filename of a different length, silently.
    """
    found = re.search(r"ses-([0-9A-Za-z]+)", path.split("/")[-1])
    if not found:
        raise ValueError(f"no session id in {path}")
    return found.group(1)


def load_sessions(pattern, task="h1"):
    """(session id, neural, kinematics, eval_mask) for every NWB matching `pattern`."""
    import glob

    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    out = []
    for f in sorted(glob.glob(pattern)):
        neural, kin, _, mask = load_nwb(f, FalconTask[task])
        out.append((session_id(f), neural.astype("float32"), kin.astype("float32"), mask))
    if not out:
        raise FileNotFoundError(f"no sessions matched {pattern}")
    return out


def disjoint_calib(calib, minival):
    """Calibration sessions with the minival recording removed.

    In this dandiset each held-in-minival file is a byte-identical PREFIX of the
    held-in-calib file for the same session (verified on all 13). Training on all of
    calib and then scoring minival therefore scores a model on its own training data,
    which inflates the result in proportion to how much the model can memorize. Drop
    the overlapping prefix so the two splits are genuinely disjoint.

    Every way this can go wrong raises. It used to cut purely on the minival recording's
    LENGTH, and to treat an unmatched session as nothing to remove — so a changed
    filename, a partial download, or a minival that stopped being a prefix would each
    have left the overlap in place and returned a higher score with no error. That is
    the same silent-inflation shape as the defect it exists to fix.
    """
    overlap = {name: (len(neural), neural, kin) for name, neural, kin, _ in minival}
    if len(overlap) != len(minival):
        raise ValueError("duplicate session ids among the minival recordings")
    missing = {name for name, *_ in calib} - set(overlap)
    if missing:
        raise ValueError(f"no minival counterpart for {sorted(missing)}; "
                         "the overlap cannot be excised and the split is not disjoint")

    trimmed = []
    for name, neural, kin, mask in calib:
        cut, head, head_kin = overlap[name]
        if not (np.array_equal(neural[:cut], head) and np.array_equal(kin[:cut], head_kin)):
            raise ValueError(f"{name}: minival is not a prefix of calib, so cutting the "
                             "first {cut} bins would remove the wrong rows")
        trimmed.append((name, neural[cut:], kin[cut:], None if mask is None else mask[cut:]))
    return trimmed


def load_falcon(path, task="h1", window=50):
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    neural, kinematics, _, _ = load_nwb(path, FalconTask[task])
    return SpikeWindows(neural, behavior=kinematics, window=window)


def make_decoder(model, task_config, window=50, batch_size=1):
    """Return a BCIDecoder over the trained model for the FALCON evaluator."""
    from falcon_challenge.interface import BCIDecoder

    class NoemaDecoder(BCIDecoder):
        def __init__(self):
            super().__init__(task_config, batch_size)
            self.stream = StreamingDecoder(model, torch.arange(task_config.n_channels), window)

        def reset(self, dataset_tags=[""]):
            self.stream.reset(len(dataset_tags))

        def predict(self, neural_observations):
            return self.stream.step(neural_observations).cpu().numpy()

        def on_done(self, dones):
            pass

    return NoemaDecoder()
