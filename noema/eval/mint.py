"""MINT-style rate estimator for NLB co-bps — a non-neural-net ensemble member.

MINT (Perich/Kaufman, eLife 2024) builds a trajectory library with no gradients, which
decorrelates it from the transformer members in a rate-space ensemble. Two readouts, both
validated on synthetic (this file's __main__ has no data dependency):
  * condition mix  — assign a whole trial to a soft mix of conditions (fast; suited to
    trials aligned to movement onset, with little residual timing jitter)
  * state matching — match each timestep to library (condition, time) states with a
    temporal-continuity prior; re-times each trial, so it tolerates trial-to-trial timing
    variability (time-warped trials).

Reuses noema.data.nlb._find_nwb and noema.eval.metrics. Pass the condition-defining
trial_info columns via cond_keys (e.g. ['trial_type','trial_version'] for MC_Maze).
"""
import numpy as np


def _smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(4 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * x.ndim
    pad[-2] = (r, r)
    xp = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * xp[..., i:i + x.shape[-2], :]
    return out


def build_library(train_spikes, cond_ids, sigma=2.0, shrink=0.0):
    """Condition-averaged smoothed rate templates: spikes [K,T,N], cond_ids [K] -> [C,T,N].
    shrink>0 pulls each condition template toward the grand mean (empirical-Bayes partial
    pooling), trading a little condition contrast for lower per-template estimation noise
    (~15 trials/condition); tune on the select split."""
    train_spikes = np.asarray(train_spikes, dtype=np.float64)
    conds = np.unique(cond_ids)
    templates = np.stack([train_spikes[cond_ids == c].mean(0) for c in conds])
    if shrink > 0:
        templates = (1 - shrink) * templates + shrink * train_spikes.mean(0, keepdims=True)
    return _smooth(templates, sigma).clip(min=1e-4), conds


def infer_condition_mix(test_hi, lib_hi, lib_full, temp=1.0):
    """Soft condition assignment (fast path). Returns [K,T,Nall]."""
    test_hi = np.asarray(test_hi, dtype=np.float64)
    ll = np.einsum("ktn,ctn->kc", test_hi, np.log(lib_hi)) - lib_hi.sum((1, 2))[None]
    ll -= ll.max(1, keepdims=True)
    w = np.exp(ll / temp)
    w /= w.sum(1, keepdims=True)
    return np.einsum("kc,ctn->ktn", w, lib_full)


def infer_state_match(test_hi, lib_hi, lib_full, window=2, temp=1.0, cont=0.15):
    """State matching (real-data path): per test time, weight every library (condition,time)
    state by windowed Poisson LL + a continuity prior toward matching time, mix full states.
    Re-times each trial, so it tolerates trial-to-trial timing variability. Returns [K,T,Nall]."""
    test_hi = np.asarray(test_hi, dtype=np.float64)
    K, T, Nin = test_hi.shape
    C, Ts, Nall = lib_full.shape
    logr = np.log(lib_hi)
    lib_states = lib_full.reshape(C * Ts, Nall)
    lib_sum = lib_hi.sum(-1)                                   # [C,Ts]
    s_idx = np.tile(np.arange(Ts), C)
    c_idx = np.repeat(np.arange(C), Ts)
    out = np.zeros((K, T, Nall))
    for t in range(T):
        w0, w1 = max(0, t - window), min(T, t + window + 1)
        obs = test_hi[:, w0:w1]
        ls = np.clip(np.arange(w0, w1) + (s_idx[:, None] - t), 0, Ts - 1)
        lr = logr[c_idx[:, None], ls]                         # [C*Ts, w, Nin]
        lm = lib_sum[c_idx[:, None], ls]                      # [C*Ts, w]
        ll = np.einsum("kwn,swn->ks", obs, lr) - lm.sum(1)[None]
        ll += -cont * (s_idx[None] - t) ** 2
        ll -= ll.max(1, keepdims=True)
        w = np.exp(ll / temp)
        w /= w.sum(1, keepdims=True)
        out[:, t] = w @ lib_states
    return out


def _aligned_condition_ids(dataset, name, split, keys):
    """Conditions for the EXACT trials make_train_input_tensors returns — some trials drop
    during alignment, so filtering trial_info by split over-counts. Replicate the selection:
    make_trial_data on the same mask, surviving trial_ids in groupby(sort=False) order.
    For MC_Maze use keys=['trial_type','trial_version'] (the canonical 108-condition group)."""
    import pandas as pd

    from nlb_tools.make_tensors import PARAMS, _prep_mask
    mp = PARAMS[name]["make_params"].copy()
    mp.pop("ignored_trials", None)
    mask = _prep_mask(dataset, split)
    td = dataset.make_trial_data(ignored_trials=~mask, **mp)
    tids = [tid for tid, _ in td.groupby("trial_id", sort=False)]
    ti = dataset.trial_info.set_index("trial_id") if "trial_id" in dataset.trial_info.columns else dataset.trial_info
    return np.asarray(pd.factorize(pd.MultiIndex.from_frame(ti.loc[tids][list(keys)]))[0], dtype=np.int64)


def _load_split(name, cond_keys, path, bin_ms, split):
    """Return (train_hi, train_ho, train_cond, eval_hi, eval_ho) for `split`."""
    import os

    from nlb_tools.make_tensors import make_train_input_tensors
    from nlb_tools.nwb_interface import NWBDataset

    from noema.data.nlb import _find_nwb

    d = NWBDataset(os.path.dirname(_find_nwb(path)))
    d.resample(bin_ms)
    tr = make_train_input_tensors(d, name, trial_split="train", save_file=False)
    ev = make_train_input_tensors(d, name, trial_split=split, save_file=False)
    keys = cond_keys or (["trial_type", "trial_version"] if name == "mc_maze" else None)
    if keys is None:
        raise ValueError("pass cond_keys — the trial_info columns defining conditions")
    cond_tr = _aligned_condition_ids(d, name, "train", keys)
    if len(cond_tr) != tr["train_spikes_heldin"].shape[0]:
        raise ValueError(f"condition/trial mismatch: {len(cond_tr)} vs {tr['train_spikes_heldin'].shape[0]}")
    return (tr["train_spikes_heldin"], tr["train_spikes_heldout"], cond_tr,
            ev["train_spikes_heldin"], ev["train_spikes_heldout"])


def _predict(train_full, train_cond, test_hi, n_in, sigma, temp, state_match, shrink=0.0):
    lib = build_library(train_full, train_cond, sigma, shrink)[0]
    infer = infer_state_match if state_match else infer_condition_mix
    return infer(test_hi, lib[..., :n_in], lib, temp=temp)[..., n_in:]


def mint_cosmooth_rates(path, name="mc_maze", bin_ms=5, sigma=8.0, temp=20.0,
                        cond_keys=None, split="val", state_match=False, shrink=0.0):
    """Fit the library on train, predict held-out rates for `split`. Returns
    (heldout_rates [K,T,Nout], heldout_counts [K,T,Nout]) for bits_per_spike.

    Defaults tuned on MC_Maze (select split): condition-mix, sigma=8, temp=20. On real NLB
    data condition-mix suits trials aligned to movement onset (little residual timing
    jitter); state-match is preferable when trials are time-warped. Re-tune (sigma, temp)
    per dataset."""
    tr_hi, tr_ho, cond_tr, ev_hi, ev_ho = _load_split(name, cond_keys, path, bin_ms, split)
    full = np.concatenate([tr_hi, tr_ho], -1)
    return _predict(full, cond_tr, ev_hi, tr_hi.shape[-1], sigma, temp, state_match, shrink), ev_ho


def mint_member_rates(path, name="mc_maze", bin_ms=5, sigma=8.0, temp=20.0, cond_keys=None,
                      select_frac=0.85, select_seed=0, state_match=False):
    """MINT held-out rates as an ensemble member, aligned to the transformer members.
    Library on the CORE train trials -> predict the SELECT trials (for greedy selection);
    library on FULL train -> predict VAL (for reporting). The core/select partition
    replicates dataset.split_trials(frac=select_frac, seed=select_seed) exactly, so MINT's
    rates line up trial-for-trial with `member_rates(models, select/val)`.
    Returns (sel_rates, sel_targets, val_rates, val_targets), all numpy [K,T,Nout]."""
    import torch

    tr_hi, tr_ho, cond_tr, va_hi, va_ho = _load_split(name, cond_keys, path, bin_ms, "val")
    n_in = tr_hi.shape[-1]
    full = np.concatenate([tr_hi, tr_ho], -1)
    perm = torch.randperm(len(tr_hi), generator=torch.Generator().manual_seed(select_seed)).numpy()
    core, sel = perm[:int(len(tr_hi) * select_frac)], perm[int(len(tr_hi) * select_frac):]
    sel_rates = _predict(full[core], cond_tr[core], tr_hi[sel], n_in, sigma, temp, state_match)
    val_rates = _predict(full, cond_tr, va_hi, n_in, sigma, temp, state_match)
    return sel_rates, tr_ho[sel], val_rates, va_ho


def mint_all_rates(dataset, name="mc_maze", sigma=8.0, temp=20.0, cond_keys=None):
    """MINT held-in + held-out rates for every submission split, as a first-class ensemble
    member. The library is built on train only (no val/test leakage; train's ~16 trials/
    condition is ample) and used to predict val (for greedy selection), train (in-sample,
    for the train_rates outputs) and the sequestered test. `dataset` is an already-resampled
    NWBDataset. Returns a dict of numpy arrays keyed val_ho / {test,train}_{hi,ho}."""
    from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors

    keys = cond_keys or (["trial_type", "trial_version"] if name == "mc_maze" else None)
    if keys is None:
        raise ValueError("pass cond_keys — the trial_info columns defining conditions")
    tr = make_train_input_tensors(dataset, name, trial_split="train", save_file=False)
    va = make_train_input_tensors(dataset, name, trial_split="val", save_file=False)
    te_hi = make_eval_input_tensors(dataset, name, trial_split="test", save_file=False)["eval_spikes_heldin"]
    tr_hi, tr_ho = tr["train_spikes_heldin"], tr["train_spikes_heldout"]
    n_in = tr_hi.shape[-1]
    cond_tr = _aligned_condition_ids(dataset, name, "train", keys)
    lib = build_library(np.concatenate([tr_hi, tr_ho], -1), cond_tr, sigma)[0]
    lib_hi = lib[..., :n_in]

    def pred(test_hi):
        return infer_condition_mix(test_hi, lib_hi, lib, temp)

    val_p, train_p, test_p = pred(va["train_spikes_heldin"]), pred(tr_hi), pred(te_hi)
    return {
        "val_ho": val_p[..., n_in:],
        "test_hi": test_p[..., :n_in], "test_ho": test_p[..., n_in:],
        "train_hi": train_p[..., :n_in], "train_ho": train_p[..., n_in:],
    }, va["train_spikes_heldout"]


if __name__ == "__main__":  # synthetic self-test, no data dependency
    def _bps(rates, counts):
        rates = np.clip(rates, 1e-8, None)
        ll = (counts * np.log(rates) - rates).sum()
        n = counts.mean((0, 1), keepdims=True).clip(1e-8)
        return (ll - (counts * np.log(n) - n).sum()) / (counts.sum() * np.log(2))

    rng = np.random.default_rng(0)
    C, T, N, nin = 8, 50, 60, 40
    tt = np.linspace(0, 2 * np.pi, T)
    prof = (rng.uniform(0.2, 1.5, (1, N)) * (1 + 0.8 * np.sin(
        rng.uniform(1, 3, (1, N)) * tt[None, :, None] + rng.uniform(0, 2 * np.pi, (C, N))[:, None]))).clip(0.02)

    def draw(K, seed, jit):
        r = np.random.default_rng(seed)
        cond = r.integers(0, C, K)
        sh = r.integers(-jit, jit + 1, K)
        p = np.stack([np.roll(prof[c], s, 0) for c, s in zip(cond, sh)])
        return r.poisson(p).astype(np.float64), cond

    for jit in (0, 6):
        tr, tc = draw(500, 1, jit)
        ev, _ = draw(150, 2, jit)
        lib, _ = build_library(tr, tc, 1.5)
        lh = lib[..., :nin]
        cm = _bps(infer_condition_mix(ev[..., :nin], lh, lib)[..., nin:], ev[..., nin:])
        sm = _bps(infer_state_match(ev[..., :nin], lh, lib)[..., nin:], ev[..., nin:])
        print(f"jitter=±{jit}: condition-mix={cm:.4f}  state-match={sm:.4f}")
