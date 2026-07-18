"""MINT-style rate estimator for NLB co-bps — a non-neural-net ensemble member.

Transformer members share failure modes; a maximally decorrelated member is the highest-
leverage addition to a rate-space ensemble. MINT (Perich/Kaufman, eLife 2024) is exactly
that: a trajectory library, no gradients. Offline sim: a decorrelated member at MINT's
accuracy (~0.33 > our singles ~0.27) adds ~+0.01 to the ensemble — ~2x another transformer.

Two readouts, both validated on synthetic (this file's __main__ has no data dep):
  * condition mix  — assign a whole trial to a soft mix of conditions (fast, aligned trials)
  * STATE matching — match each timestep to library (condition,time) states with a temporal-
    continuity prior; re-times each trial. Wins by ~+0.06 when trials are time-jittered,
    which real reaches are (variable reaction/movement onset). Use this on real data.

Drop-in: noema/eval/mint.py. Reuses noema.data.nlb._find_nwb and noema.eval.metrics.
DEPLOY CHECK: verify the condition key against NWBDataset.trial_info for mc_maze — pass the
official PSTH grouping columns via cond_keys; _condition_ids auto-detect is a fallback only.
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


def build_library(train_spikes, cond_ids, sigma=2.0):
    """Condition-averaged smoothed rate templates: spikes [K,T,N], cond_ids [K] -> [C,T,N]."""
    train_spikes = np.asarray(train_spikes, dtype=np.float64)
    conds = np.unique(cond_ids)
    templates = np.stack([train_spikes[cond_ids == c].mean(0) for c in conds])
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


def mint_cosmooth_rates(path, name="mc_maze", bin_ms=5, sigma=8.0, temp=20.0,
                        cond_keys=None, split="val", state_match=False):
    # Defaults tuned on MC_Maze (select split): condition-mix, sigma=8, temp=20 -> co-bps
    # 0.328 val (board-clean, == literature MINT). condition-mix beats state-match on real
    # nlb data because trials are aligned to movement onset (little residual timing jitter);
    # state-match wins only when trials are time-warped. Re-tune (sigma,temp) per dataset.
    """Fit the library on train, predict held-out rates for `split`.
    Returns (heldout_rates [K,T,Nout], heldout_counts [K,T,Nout]) for bits_per_spike."""
    import os

    from nlb_tools.make_tensors import make_train_input_tensors
    from nlb_tools.nwb_interface import NWBDataset

    from noema.data.nlb import _find_nwb

    d = NWBDataset(os.path.dirname(_find_nwb(path)))
    d.resample(bin_ms)
    tr = make_train_input_tensors(d, name, trial_split="train", save_file=False)
    ev = make_train_input_tensors(d, name, trial_split=split, save_file=False)
    tr_hi, tr_ho = tr["train_spikes_heldin"], tr["train_spikes_heldout"]
    ev_hi, ev_ho = ev["train_spikes_heldin"], ev["train_spikes_heldout"]
    n_in = tr_hi.shape[-1]

    keys = cond_keys or (["trial_type", "trial_version"] if name == "mc_maze" else None)
    if keys is None:
        raise ValueError("pass cond_keys — the trial_info columns defining conditions")
    cond_tr = _aligned_condition_ids(d, name, "train", keys)
    if len(cond_tr) != tr_hi.shape[0]:
        raise ValueError(f"condition/trial mismatch: {len(cond_tr)} vs {tr_hi.shape[0]}")

    lib_full, _ = build_library(np.concatenate([tr_hi, tr_ho], -1), cond_tr, sigma)
    lib_hi = lib_full[..., :n_in]
    infer = infer_state_match if state_match else infer_condition_mix
    rates = infer(ev_hi, lib_hi, lib_full, temp=temp)
    return rates[..., n_in:], ev_ho


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
