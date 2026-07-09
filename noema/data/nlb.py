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


def load_nlb(path, name="mc_maze", bin_ms=5, window=None, split="train"):
    from nlb_tools.make_tensors import make_train_input_tensors
    from nlb_tools.nwb_interface import NWBDataset

    dataset = NWBDataset(path)
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
