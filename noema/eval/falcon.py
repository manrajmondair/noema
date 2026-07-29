"""FALCON benchmark integration (falcon-challenge).

`load_falcon` reuses the challenge's NWB reader for training data; `make_decoder`
wraps a trained model in the streaming BCIDecoder interface used by the evaluator.
Covers the continuous-kinematic tasks (h1/m1/m2); h2/b1 use different targets.
"""

import torch

from ..data.dataset import SpikeWindows
from .streaming import StreamingDecoder


def load_sessions(pattern, task="h1"):
    """(name, neural, kinematics, eval_mask) for every NWB matching `pattern`."""
    import glob

    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    out = []
    for f in sorted(glob.glob(pattern)):
        neural, kin, _, mask = load_nwb(f, FalconTask[task])
        out.append((f.split("/")[-1][-28:-4], neural.astype("float32"), kin.astype("float32"), mask))
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
    """
    session = lambda name: name.split("ses-")[-1]
    overlap = {session(name): len(neural) for name, neural, _, _ in minival}
    trimmed = []
    for name, neural, kin, mask in calib:
        cut = overlap.get(session(name), 0)
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
