"""Build an EvalAI submission: model rates on the sequestered NLB test trials.

co-bps requires only `eval_rates_heldout`; the held-in and train rates enable the
optional velocity/PSTH metrics. This writes the submission .h5 — the final upload
to EvalAI (which holds the test labels) is a manual step with your account.
"""

import argparse

import torch

from ..data.nlb import _find_nwb
from ..utils import default_device
from .ensemble_run import build_from_state


# Batch over trials: the spatial members build per-unit token tensors
# [trials, T, units, dim] whose size explodes if the whole split is run at once.
_BATCH = 16


@torch.no_grad()
def _cosmooth(models, spikes, in_ids, out_ids, device):
    """Mean held-out rates across models, in trial batches."""
    spikes = torch.as_tensor(spikes, dtype=torch.float32)
    out = [torch.stack([m.cosmooth(spikes[i:i + _BATCH].to(device), in_ids, out_ids).exp()
                        for m in models]).mean(0).cpu()
           for i in range(0, spikes.size(0), _BATCH)]
    return torch.cat(out)


@torch.no_grad()
def _rates(models, spikes, in_ids, out_ids, device):
    spikes = torch.as_tensor(spikes, dtype=torch.float32)
    his, hos = [], []
    for i in range(0, spikes.size(0), _BATCH):
        s = spikes[i:i + _BATCH].to(device)
        his.append(torch.stack([m.tokenizer.decode(m.encode(s, in_ids), in_ids).exp() for m in models]).mean(0).cpu())
        hos.append(torch.stack([m.cosmooth(s, in_ids, out_ids).exp() for m in models]).mean(0).cpu())
    return torch.cat(his).numpy(), torch.cat(hos).numpy()


def make_submission(ckpts, path, name, out_h5, bin_ms=5, heads=8):
    import os

    from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors, save_to_h5
    from nlb_tools.nwb_interface import NWBDataset

    # load the directory (train + test NWB merged) so the test trial_split is present
    dataset = NWBDataset(os.path.dirname(_find_nwb(path)))
    dataset.resample(bin_ms)
    eval_hi = make_eval_input_tensors(dataset, name, trial_split="test", save_file=False)["eval_spikes_heldin"]
    train = make_train_input_tensors(dataset, name, trial_split="train", save_file=False)
    n_hi, n_ho = eval_hi.shape[-1], train["train_spikes_heldout"].shape[-1]

    device = default_device()
    models = [build_from_state(torch.load(c, map_location="cpu"), n_hi + n_ho, heads)[0].to(device) for c in ckpts]
    in_ids = torch.arange(n_hi, device=device)
    out_ids = torch.arange(n_ho, device=device) + n_hi

    # Greedy-select the ensemble and tune inference smoothing on the held-out val
    # split (our dev set; the test labels stay sequestered), then predict test/train
    # with the chosen members (weighted by pick count) — our best ensemble, honestly.
    from .baselines import gaussian_smooth
    from .ensemble import greedy_ensemble
    from .metrics import bits_per_spike
    val = make_train_input_tensors(dataset, name, trial_split="val", save_file=False)
    val_member = [_cosmooth([m], val["train_spikes_heldin"], in_ids, out_ids, device) for m in models]
    val_ho = torch.as_tensor(val["train_spikes_heldout"], dtype=torch.float32)
    chosen = greedy_ensemble(val_member, val_ho)
    models = [models[j] for j in chosen]
    sel_val = sum(val_member[j] for j in chosen) / len(chosen)
    sigmas = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
    sigma = max(sigmas, key=lambda s: bits_per_spike(gaussian_smooth(sel_val, s), val_ho))
    print(f"greedy selected {len(chosen)} picks from {len(val_member)} members, smoothing sigma={sigma}", flush=True)

    smooth = lambda a: gaussian_smooth(torch.as_tensor(a, dtype=torch.float32), sigma).numpy()
    er_hi, er_ho = (smooth(r) for r in _rates(models, eval_hi, in_ids, out_ids, device))
    tr_hi, tr_ho = (smooth(r) for r in _rates(models, train["train_spikes_heldin"], in_ids, out_ids, device))
    submission = {name: {
        "eval_rates_heldin": er_hi, "eval_rates_heldout": er_ho,
        "train_rates_heldin": tr_hi, "train_rates_heldout": tr_ho,
    }}
    save_to_h5(submission, out_h5, overwrite=True)
    print(f"wrote {out_h5}: {er_ho.shape[0]} test trials, {n_ho} held-out neurons", flush=True)


def main():
    p = argparse.ArgumentParser(prog="noema.eval.submission")
    p.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--name", default="mc_maze")
    p.add_argument("--path", required=True)
    p.add_argument("--out", default="submission.h5")
    p.add_argument("--bin-ms", type=int, default=5)
    args = p.parse_args()
    make_submission(args.ckpts.split(","), args.path, args.name, args.out, args.bin_ms)


if __name__ == "__main__":
    main()
