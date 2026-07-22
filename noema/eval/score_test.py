"""Score a submission .h5 against the NLB'21 held-out TEST labels.

The test labels were sequestered on EvalAI; they are now public in the nlb_tools repo
(`data/eval_data_test.h5`, Git-LFS). This computes the leaderboard-comparable
co-bps locally, without an EvalAI upload.
"""
import argparse
import os
import urllib.request

import h5py
import numpy as np

from .metrics import bits_per_spike

TEST_LABELS_URL = "https://media.githubusercontent.com/media/neurallatents/nlb_tools/main/data/eval_data_test.h5"


def _labels(path):
    if not os.path.exists(path):
        print(f"downloading public test labels -> {path}", flush=True)
        urllib.request.urlretrieve(TEST_LABELS_URL, path)
    return h5py.File(path, "r")


def score(submission_h5, name="mc_maze", labels_h5="data/eval_data_test.h5"):
    """co-bps (and fp-bps/vel/PSTH if present) of a submission on the public test split."""
    import torch

    lab = _labels(labels_h5)[name]
    ho = np.asarray(lab["eval_spikes_heldout"][()], dtype=np.float64)  # [K,T,Nout]
    with h5py.File(submission_h5, "r") as h:
        pred = np.asarray(h[name]["eval_rates_heldout"][()], dtype=np.float64)
    if pred.shape != ho.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs test held-out {ho.shape}")
    out = {"co_bps": float(bits_per_spike(torch.tensor(pred), torch.tensor(ho)))}
    # fp-bps if the forward keys are present in both
    if "eval_spikes_heldout_forward" in lab:
        with h5py.File(submission_h5, "r") as h:
            if "eval_rates_heldout_forward" in h[name]:
                fho = np.asarray(lab["eval_spikes_heldout_forward"][()], dtype=np.float64)
                fpred = np.asarray(h[name]["eval_rates_heldout_forward"][()], dtype=np.float64)
                if fpred.shape == fho.shape:
                    out["fp_bps"] = float(bits_per_spike(torch.tensor(fpred), torch.tensor(fho)))
    return out


def main():
    p = argparse.ArgumentParser(prog="noema.eval.score_test")
    p.add_argument("--submission", required=True, help="path to a submission .h5")
    p.add_argument("--name", default="mc_maze")
    p.add_argument("--labels", default="data/eval_data_test.h5")
    args = p.parse_args()
    for k, v in score(args.submission, args.name, args.labels).items():
        print(f"TEST {k} = {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
