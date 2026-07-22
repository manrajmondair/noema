"""FALCON benchmark integration (falcon-challenge).

`load_falcon` reuses the challenge's NWB reader for training data; `make_decoder`
wraps a trained model in the streaming BCIDecoder interface used by the evaluator.
Covers the continuous-kinematic tasks (h1/m1/m2); h2/b1 use different targets.
"""

import torch

from ..data.dataset import SpikeWindows
from .streaming import StreamingDecoder


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
