import numpy as np

from noema.eval.mint import build_library


def test_shrinkage_pulls_templates_toward_grand_mean():
    rng = np.random.default_rng(0)
    spikes = rng.poisson(0.3, size=(60, 20, 5)).astype(float)
    cond = np.repeat(np.arange(4), 15)

    base, conds = build_library(spikes, cond, sigma=0.0, shrink=0.0)
    assert base.shape == (4, 20, 5)

    # shrink=1 collapses every condition template onto the (smoothed) grand mean,
    # so the between-condition spread must vanish.
    full = build_library(spikes, cond, sigma=0.0, shrink=1.0)[0]
    assert full.std(axis=0).max() < 1e-9

    # a partial shrink strictly reduces between-condition spread but keeps some.
    part = build_library(spikes, cond, sigma=0.0, shrink=0.3)[0]
    assert part.std(axis=0).max() < base.std(axis=0).max()
    assert part.std(axis=0).max() > 1e-9
