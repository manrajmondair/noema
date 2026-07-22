"""Build an EvalAI submission: model rates on the sequestered NLB test trials.

co-bps requires only `eval_rates_heldout`; the held-in and train rates enable the
optional velocity/PSTH metrics. This writes the submission .h5; the final upload
to EvalAI, which holds the test labels, is performed separately.
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
def _cosmooth(models, spikes, in_ids, out_ids, device, tta=0):
    """Mean held-out rates across models, in trial batches."""
    spikes = torch.as_tensor(spikes, dtype=torch.float32)
    out = [torch.stack([(m.cosmooth_tta(spikes[i:i + _BATCH].to(device), in_ids, out_ids, tta) if tta
                                     else m.cosmooth(spikes[i:i + _BATCH].to(device), in_ids, out_ids).exp())
                        for m in models]).mean(0).cpu()
           for i in range(0, spikes.size(0), _BATCH)]
    return torch.cat(out)


def _heldin(m, s, in_ids):
    """A model's held-in log-rates via its OWN readout: per-unit for the spatial
    encoder, pooled-latent for the temporal one (mirrors Noema.forward)."""
    tokens, z = m._represent(s, in_ids)
    return m.tokenizer.decode_units(tokens, in_ids) if m.spatial else m.tokenizer.decode(z, in_ids)


@torch.no_grad()
def _rates(models, spikes, in_ids, out_ids, device):
    spikes = torch.as_tensor(spikes, dtype=torch.float32)
    his, hos = [], []
    for i in range(0, spikes.size(0), _BATCH):
        s = spikes[i:i + _BATCH].to(device)
        his.append(torch.stack([_heldin(m, s, in_ids).exp() for m in models]).mean(0).cpu())
        hos.append(torch.stack([m.cosmooth(s, in_ids, out_ids).exp() for m in models]).mean(0).cpu())
    return torch.cat(his).numpy(), torch.cat(hos).numpy()


@torch.no_grad()
def _forward(models, spikes, in_ids, out_ids, device, fp_steps):
    """Forward-prediction rates: roll the world model open-loop fp_steps past each trial
    and decode the imagined held-in/held-out firing (the fp-bps target). The rollout is
    in the pooled latent, so both readouts use the pooled decode."""
    spikes = torch.as_tensor(spikes, dtype=torch.float32)
    hif, hof = [], []
    for i in range(0, spikes.size(0), _BATCH):
        s = spikes[i:i + _BATCH].to(device)
        hi_m, ho_m = [], []
        for m in models:
            z = m.encode(s, in_ids)
            for _ in range(fp_steps):
                z = torch.cat([z, m.world(z, None)[:, -1:]], dim=1)
            fwd = z[:, -fp_steps:]
            # open-loop rollouts of a one-step-trained model can diverge; clamp to a
            # physical ceiling so a drifting member cannot poison the ensemble mean.
            hi_m.append(m.tokenizer.decode(fwd, in_ids).exp().clamp_max(20.0))
            ho_m.append(m.tokenizer.decode(fwd, out_ids).exp().clamp_max(20.0))
        hif.append(torch.stack(hi_m).mean(0).cpu())
        hof.append(torch.stack(ho_m).mean(0).cpu())
    return torch.cat(hif).numpy(), torch.cat(hof).numpy()


def make_submission(ckpts, path, name, out_h5, bin_ms=5, heads=8, forward=False, mint=False, teacher=False):
    import os

    from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors, save_to_h5
    from nlb_tools.nwb_interface import NWBDataset

    # load the directory (train + test NWB merged) so the test trial_split is present
    dataset = NWBDataset(os.path.dirname(_find_nwb(path)))
    dataset.resample(bin_ms)
    eval_hi = make_eval_input_tensors(dataset, name, trial_split="test", save_file=False)["eval_spikes_heldin"]
    train = make_train_input_tensors(dataset, name, trial_split="train", save_file=False)
    n_hi, n_ho = eval_hi.shape[-1], train["train_spikes_heldout"].shape[-1]

    # Provenance: the sequestered TEST split must actually be present (requires the test
    # NWB in the dandiset dir). A train-only download yields an empty test mask and a
    # well-formed but meaningless submission — fail loudly rather than upload garbage.
    if eval_hi.shape[0] == 0:
        raise RuntimeError(f"no '{name}' test trials found — the sequestered test NWB is absent; "
                           "download the full dandiset before generating a submission.")
    print(f"test trials={eval_hi.shape[0]}, train trials={train['train_spikes_heldin'].shape[0]}, "
          f"held-in units={n_hi}, held-out units={n_ho}", flush=True)

    device = default_device()
    models = [build_from_state(torch.load(c, map_location="cpu"), n_hi + n_ho, heads)[0].to(device) for c in ckpts]
    if teacher:  # co-smooth through the EMA teacher encoder (smoothed weights -> steadier rates)
        for m in models:
            m.encoder = m.teacher
            if getattr(m, "teacher_pooler", None) is not None:
                m.pooler = m.teacher_pooler
    in_ids = torch.arange(n_hi, device=device)
    out_ids = torch.arange(n_ho, device=device) + n_hi

    # Greedy-select the ensemble and tune inference smoothing on the held-out val
    # split (the dev set; the test labels stay sequestered), then predict test/train
    # with the chosen members (weighted by pick count).
    from .baselines import gaussian_smooth
    from .ensemble import greedy_ensemble
    from .metrics import bits_per_spike
    val = make_train_input_tensors(dataset, name, trial_split="val", save_file=False)
    val_member = [_cosmooth([m], val["train_spikes_heldin"], in_ids, out_ids, device) for m in models]
    val_ho = torch.as_tensor(val["train_spikes_heldout"], dtype=torch.float32)

    # MINT: a decorrelated trajectory-library member. Predicts held-in+held-out rates for
    # val (greedy), train, and the sequestered test from a train-only library. Appended as
    # the last member (index len(models)); greedy may pick it, count-weighted, with the rest.
    mint_rates = None
    if mint:
        import numpy as np

        from .mint import mint_all_rates
        mint_rates, mint_val_ho = mint_all_rates(dataset, name)
        assert np.allclose(mint_val_ho, val["train_spikes_heldout"]), "MINT/transformer val trial misalignment"
        val_member.append(torch.as_tensor(mint_rates["val_ho"], dtype=torch.float32))

    chosen = greedy_ensemble(val_member, val_ho)
    sel_val = sum(val_member[j] for j in chosen) / len(chosen)
    sigmas = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
    sigma = max(sigmas, key=lambda s: bits_per_spike(gaussian_smooth(sel_val, s), val_ho))
    n_mint, tot = len(models), len(chosen)
    tf_picks = [models[j] for j in chosen if j != n_mint]   # transformer picks (with repeats)
    mc = chosen.count(n_mint) if mint else 0
    print(f"greedy selected {tot} picks from {len(val_member)} members (mint picks={mc}), "
          f"smoothing sigma={sigma}", flush=True)

    smooth = lambda a: gaussian_smooth(torch.as_tensor(a, dtype=torch.float32), sigma).numpy()

    def blend(tf_input, mint_hi, mint_ho):
        # count-weighted mean over chosen members: transformer picks via _rates, MINT (mc
        # copies) via its precomputed rates. Reduces to the plain transformer mean when mc=0.
        n = len(tf_picks)
        hi, ho = _rates(tf_picks, tf_input, in_ids, out_ids, device) if n else (0.0, 0.0)
        if mc:
            hi, ho = (hi * n + mc * mint_hi) / tot, (ho * n + mc * mint_ho) / tot
        return smooth(hi), smooth(ho)

    mr = mint_rates or {}
    er_hi, er_ho = blend(eval_hi, mr.get("test_hi"), mr.get("test_ho"))
    tr_hi, tr_ho = blend(train["train_spikes_heldin"], mr.get("train_hi"), mr.get("train_ho"))
    submission = {name: {
        "eval_rates_heldin": er_hi, "eval_rates_heldout": er_ho,
        "train_rates_heldin": tr_hi, "train_rates_heldout": tr_ho,
    }}

    # Forward-prediction rates (fp-bps) — opt-in. A one-step-trained ensemble drifts in
    # open loop, so its forward rollout is unreliable; only include it with a world model
    # trained for multi-step rollout, else omit fp and score co-bps + vel + PSTH only.
    if forward:
        from nlb_tools.make_tensors import PARAMS
        fp_steps = int(PARAMS[name]["fp_len"] // bin_ms)  # MINT has no rollout -> forward uses transformer picks only
        ef_hi, ef_ho = (smooth(r) for r in _forward(tf_picks, eval_hi, in_ids, out_ids, device, fp_steps))
        tf_hi, tf_ho = (smooth(r) for r in _forward(tf_picks, train["train_spikes_heldin"], in_ids, out_ids, device, fp_steps))
        submission[name].update({
            "eval_rates_heldin_forward": ef_hi, "eval_rates_heldout_forward": ef_ho,
            "train_rates_heldin_forward": tf_hi, "train_rates_heldout_forward": tf_ho,
        })

    save_to_h5(submission, out_h5, overwrite=True)
    print(f"wrote {out_h5}: {er_ho.shape[0]} test trials, {n_ho} held-out neurons, "
          f"forward={'yes' if forward else 'no'}", flush=True)


def main():
    p = argparse.ArgumentParser(prog="noema.eval.submission")
    p.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--name", default="mc_maze")
    p.add_argument("--path", required=True)
    p.add_argument("--out", default="submission.h5")
    p.add_argument("--bin-ms", type=int, default=5)
    p.add_argument("--forward", action="store_true", help="include fp-bps forward rates (needs a multi-step world model)")
    p.add_argument("--mint", action="store_true", help="add the MINT trajectory-library member (decorrelated)")
    p.add_argument("--teacher", action="store_true", help="co-smooth through the EMA teacher encoder")
    args = p.parse_args()
    make_submission(args.ckpts.split(","), args.path, args.name, args.out, args.bin_ms,
                    forward=args.forward, mint=args.mint, teacher=args.teacher)


if __name__ == "__main__":
    main()
