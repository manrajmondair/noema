"""Neural Latents Benchmark loader.

Fetch a dataset first (Dandi Archive), e.g. mc_maze = dandiset 000128:

    dandi download DANDI:000128/draft -o data/

then point `path` at the extracted NWB. Requires the `data` extra.
"""

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


def load_nlb(path, name="mc_maze", bin_ms=5, window=None, split="train"):
    from nlb_tools.make_tensors import make_train_input_tensors
    from nlb_tools.nwb_interface import NWBDataset

    dataset = NWBDataset(_find_nwb(path))
    dataset.resample(bin_ms)
    tensors = make_train_input_tensors(
        dataset, dataset_name=name, trial_split=split,
        save_file=False, include_behavior=True,
    )
    return SpikeWindows(
        tensors["train_spikes_heldin"],
        tensors["train_spikes_heldout"],
        tensors.get("train_behavior"),
        window,
    )
