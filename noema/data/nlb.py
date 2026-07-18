"""Neural Latents Benchmark loader.

Fetch a dataset first (Dandi Archive), e.g. mc_maze = dandiset 000128:

    dandi download DANDI:000128/draft -o data/

then point `path` at the extracted NWB. Requires the `data` extra.
"""

from functools import lru_cache

import torch

from .dataset import SpikeWindows

# NLB dataset name -> Dandi id, for reference/tooling.
DANDISETS = {"mc_maze": "000128", "mc_rtt": "000129", "area2_bump": "000127",
             "dmfc_rsg": "000130", "mc_maze_large": "000138", "mc_maze_medium": "000139",
             "mc_maze_small": "000140"}


def _find_nwb(path):
    import glob
    import os
    if os.path.isfile(path):
        return path
    # Dandi nests NWBs under a subject dir and ships separate train/test files;
    # the train file carries held-in + held-out spikes and behavior (val splits
    # out of it), while the test file is label-free eval only.
    files = glob.glob(os.path.join(path, "**", "*.nwb"), recursive=True)
    train = [f for f in files if "train" in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(f"no NWB files under {path}")
    return (train or files)[0]


@lru_cache(maxsize=2)
def _load_resampled(path, bin_ms):
    """Load and resample the NWB once; splits reuse it. Loading + resampling
    dominates eval, and callers pull several splits from the same recording."""
    from nlb_tools.nwb_interface import NWBDataset

    dataset = NWBDataset(_find_nwb(path))
    dataset.resample(bin_ms)
    return dataset


def load_nlb(path, name="mc_maze", bin_ms=5, window=None, split="train"):
    from nlb_tools.make_tensors import make_train_input_tensors

    dataset = _load_resampled(path, bin_ms)
    import os
    _beh = os.environ.get("NOEMA_NO_BEHAVIOR") != "1"
    tensors = make_train_input_tensors(
        dataset, dataset_name=name, trial_split=split,
        save_file=False, include_behavior=_beh,
    )
    behavior, stats = tensors.get("train_behavior"), None
    if behavior is not None:
        # Standardize velocity per dimension: raw NLB kinematics are large-scale, so
        # an unnormalized MSE would dominate the multi-task loss. Keep the (mean, std)
        # so the leaderboard R² can score against the raw kinematics it expects.
        import numpy as np
        behavior = np.asarray(behavior, dtype="float32")
        flat = behavior.reshape(-1, behavior.shape[-1])
        mean, std = np.nanmean(flat, 0), np.nanstd(flat, 0) + 1e-6
        behavior = (behavior - mean) / std
        stats = (torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32))

    return SpikeWindows(
        tensors["train_spikes_heldin"],
        tensors["train_spikes_heldout"],
        behavior,
        window=window,
        behavior_stats=stats,
    )
