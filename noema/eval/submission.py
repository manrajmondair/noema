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


@torch.no_grad()
def _rates(models, spikes, in_ids, out_ids, device):
    spikes = torch.as_tensor(spikes, dtype=torch.float32, device=device)
    hi = torch.stack([m.tokenizer.decode(m.encode(spikes, in_ids), in_ids).exp() for m in models]).mean(0)
    ho = torch.stack([m.cosmooth(spikes, in_ids, out_ids).exp() for m in models]).mean(0)
    return hi.cpu().numpy(), ho.cpu().numpy()


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

    # Greedy-select the ensemble on the held-out val split, then predict test/train
    # with the chosen members (weighted by pick count) — our best ensemble, honestly.
    from .ensemble import greedy_ensemble
    val = make_train_input_tensors(dataset, name, trial_split="val", save_file=False)
    vh = torch.as_tensor(val["train_spikes_heldin"], dtype=torch.float32, device=device)
    val_member = [m.cosmooth(vh, in_ids, out_ids).exp().cpu() for m in models]
    chosen = greedy_ensemble(val_member, torch.as_tensor(val["train_spikes_heldout"], dtype=torch.float32))
    models = [models[j] for j in chosen]
    print(f"greedy selected {len(chosen)} picks from {len(val_member)} members", flush=True)

    er_hi, er_ho = _rates(models, eval_hi, in_ids, out_ids, device)
    tr_hi, tr_ho = _rates(models, train["train_spikes_heldin"], in_ids, out_ids, device)
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
