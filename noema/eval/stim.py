"""Is a stimulation response predictable from population state, or only from a table?

The claim a write-in world model has to earn is that knowing the population's state at
the moment of stimulation tells you something about the response that the stimulation
parameters alone do not. The comparator is therefore not chance — it is the
condition-average response: for each (target region, depth, current, brain state), the
mean evoked count per unit over training trials, replayed for every held-out trial of
that condition. That table has no dynamics in it at all.

Three arms, each strictly containing the last, so a win cannot be an artefact of which
basis a model happened to be given:

    lookup                      condition average, fitted on training trials
    + own rate                  each unit from its OWN pre-stimulus rate
    + own rate + population     the population on the residual the first two leave

The middle arm exists because a unit firing fast a moment ago tends to keep firing
fast. That is carry-over on one channel and needs no model of a population. Only the
third arm beating the second says anything about population structure.

Pre-stimulus features are centred within condition using training trials only, so no
arm can recover the condition from them and re-derive the lookup table.
"""

import numpy as np

FRACTIONS = (0.60, 0.20)  # train, select; the remainder is reported and touched once
ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
COMPONENTS = (2, 5, 10, 20, 40)

PRE_WINDOW = (-0.500, 0.0)   # state, ending at stimulus onset
POST_WINDOW = (0.002, 0.050)  # response, starting past the stimulation artefact
# Allen's standard unit gates. Without them the per-unit distribution is dominated by
# units that barely fire, whose scores are noise in both directions.
QUALITY = dict(amplitude_cutoff=0.1, isi_violations=0.5, presence_ratio=0.9)


def peri_stimulus(path, dandiset="000458"):
    """Counts around every valid electrical stimulation trial, read over the network.

    These sessions are 12-27 GB each and the archive is 361 GB; the spikes are a small
    fraction of that, so the file is opened over HTTP range requests and only the unit
    and trial tables are ever fetched. Returns pre-stimulus counts, evoked counts, and
    the condition label, all indexed by trial.
    """
    import h5py
    import remfile
    from dandi.dandiapi import DandiAPIClient

    asset = DandiAPIClient().get_dandiset(dandiset, "draft").get_asset_by_path(path)
    f = h5py.File(remfile.File(asset.get_content_url(follow_redirects=1,
                                                     strip_query=True)), "r")
    text = lambda col: np.array([v.decode() if isinstance(v, bytes) else v for v in col[:]])

    t = f["intervals"]["trials"]
    valid = t["is_valid"][:].astype(bool) if "is_valid" in t else True
    keep = (text(t["stimulus_type"]) == "electrical") & valid
    onset = t["start_time"][:][keep]
    condition = np.array([" | ".join(c) for c in zip(
        text(t["estim_target_region"])[keep], text(t["estim_target_depth"])[keep],
        text(t["estim_current"])[keep], text(t["behavioral_epoch"])[keep])])

    u = f["units"]
    good = np.ones(u["id"].shape[0], bool)
    for field, limit in QUALITY.items():
        if field in u:
            v = u[field][:]
            good &= ~np.isnan(v) & ((v > limit) if field == "presence_ratio" else (v < limit))

    stop = u["spike_times_index"][:]
    start = np.concatenate([[0], stop[:-1]])
    pre = np.zeros((len(onset), int(good.sum())), np.int32)
    post = np.zeros_like(pre)
    for out, unit in enumerate(np.flatnonzero(good)):
        s = np.sort(u["spike_times"][start[unit]:stop[unit]])
        for window, into in ((PRE_WINDOW, pre), (POST_WINDOW, post)):
            into[:, out] = (np.searchsorted(s, onset + window[1])
                            - np.searchsorted(s, onset + window[0]))
    return pre, post, condition


def split_trials(condition, rng):
    """Stratified by condition, so every cell appears in all three parts."""
    parts = ([], [], [])
    for c in np.unique(condition):
        idx = np.flatnonzero(condition == c)
        rng.shuffle(idx)
        a = int(len(idx) * FRACTIONS[0])
        b = a + int(len(idx) * FRACTIONS[1])
        for part, chunk in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            part += list(chunk)
    return tuple(np.sort(np.array(p)) for p in parts)


def _by_condition(values, condition, train):
    table = {c: values[train][condition[train] == c].mean(0)
             for c in np.unique(condition[train])}
    fallback = values[train].mean(0)
    return lambda idx: np.array([table.get(c, fallback) for c in condition[idx]])


def lookup(post, condition, train):
    """The comparator. Fitted on training trials only — that is the leak that matters."""
    return _by_condition(post, condition, train)


def _centred(pre, condition, train):
    centre = _by_condition(pre, condition, train)
    return lambda idx: pre[idx] - centre(idx)


def own_rate(pre, post, condition, train, base):
    """Each unit from its own pre-stimulus rate: a per-unit scalar, no population."""
    centred = _centred(pre, condition, train)
    x, y = centred(train), post[train] - base(train)
    slope = (x * y).sum(0) / np.maximum((x * x).sum(0), 1e-9)
    return lambda idx: base(idx) + centred(idx) * slope


def population(pre, post, condition, train, base, k, alpha):
    """Ridge from the leading principal components of pre-stimulus state onto whatever
    `base` has left over. Pass the own-rate arm as `base` to make the test nested."""
    centred = _centred(pre, condition, train)
    x = centred(train)
    scale = np.where(x.std(0) > 0, x.std(0), 1.0)
    _, _, basis = np.linalg.svd(x / scale, full_matrices=False)
    project = lambda idx: (centred(idx) / scale) @ basis[:k].T

    a = project(train)
    weights = np.linalg.solve(a.T @ a + alpha * np.eye(k), a.T @ (post[train] - base(train)))
    return lambda idx: base(idx) + project(idx) @ weights


def r2_per_unit(truth, prediction):
    """Variance about each unit's own mean on the evaluated trials — a denominator every
    arm shares, so the differences between them are comparable."""
    resid = ((truth - prediction) ** 2).sum(0)
    total = ((truth - truth.mean(0)) ** 2).sum(0)
    return np.where(total > 0, 1 - resid / np.maximum(total, 1e-12), np.nan)


def trial_interval(truth, worse, better, rng, draws=2000):
    """Bootstrap the held-out TRIALS.

    Resampling units instead treats the shared trial noise every unit was scored on as
    independent evidence, and returns an interval narrow enough to call a null result
    significant. On this data that difference decides the answer.
    """
    gain = lambda pick: np.nanmean(r2_per_unit(truth[pick], better[pick])
                                   - r2_per_unit(truth[pick], worse[pick]))
    draws = [gain(rng.integers(0, len(truth), len(truth))) for _ in range(draws)]
    return (gain(np.arange(len(truth))),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def compare(pre, post, condition, seed=0, shuffle_state=False):
    """Score the three arms and return the two contrasts that decide the question."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    rng = np.random.default_rng(seed)
    train, select, test = split_trials(condition, rng)
    assert not (set(train) & set(select) | set(train) & set(test)
                | set(select) & set(test)), "splits overlap"

    if shuffle_state:
        # Break the trial-by-trial pairing, keep the condition structure. A gain that
        # survives this is a defect in the pipeline, not a finding.
        for c in np.unique(condition):
            idx = np.flatnonzero(condition == c)
            pre[idx] = pre[rng.permutation(idx)]

    base = lookup(post, condition, train)
    own = own_rate(pre, post, condition, train, base)
    tune = lambda make: max(((k, a) for k in COMPONENTS for a in ALPHAS),
                            key=lambda ka: np.nanmean(r2_per_unit(
                                post[select], make(*ka)(select))))
    build = lambda ka: population(pre, post, condition, train, own, *ka)
    nested = build(tune(lambda k, a: population(pre, post, condition, train, own, k, a)))

    truth = post[test]
    return dict(
        units=pre.shape[1], test_trials=len(test),
        lookup=float(np.nanmean(r2_per_unit(truth, base(test)))),
        own=float(np.nanmean(r2_per_unit(truth, own(test)))),
        nested=float(np.nanmean(r2_per_unit(truth, nested(test)))),
        own_over_lookup=trial_interval(truth, base(test), own(test), rng),
        population_over_own=trial_interval(truth, own(test), nested(test), rng),
    )
