"""A negative result is only worth reporting if the instrument could have found a
positive one. These plant each effect in turn and check the probe recovers it, then
check it reports nothing when there is nothing there.
"""

import numpy as np
import pytest

from noema.eval.stim import compare

UNITS, TRIALS, SEED = 40, 900, 0


def synthetic(rng, carry=0.0, coupling=0.0):
    """Stimulation trials whose response is a condition average plus, optionally, a
    per-unit carry-over from that unit's own pre-stimulus rate, and a population term
    driven by a latent no single channel carries on its own."""
    condition = np.array([f"c{i % 6}" for i in range(TRIALS)])
    level = {c: rng.normal(2.0, 0.8, UNITS) for c in np.unique(condition)}

    latent = rng.normal(size=TRIALS)
    load = rng.normal(size=UNITS)
    pre = rng.poisson(4.0, (TRIALS, UNITS)).astype(float) + latent[:, None] * load
    post = np.array([level[c] for c in condition])
    post = post + carry * (pre - pre.mean(0)) + coupling * latent[:, None] * load
    return pre, post + rng.normal(0, 0.5, post.shape), condition


def test_nothing_is_found_when_nothing_is_there():
    pre, post, condition = synthetic(np.random.default_rng(SEED))
    r = compare(pre, post, condition, seed=SEED)
    for name in ("own_over_lookup", "population_over_own"):
        low, high = r[name][1], r[name][2]
        assert low < 0.02 and high > -0.02, f"{name} claims {r[name]} on pure noise"


def test_a_planted_carry_over_is_recovered():
    pre, post, condition = synthetic(np.random.default_rng(SEED), carry=0.35)
    r = compare(pre, post, condition, seed=SEED)
    assert r["own_over_lookup"][1] > 0.05, r["own_over_lookup"]


def test_a_planted_population_term_is_recovered_above_the_carry_over():
    # The latent is spread across units, so a per-unit scalar cannot capture it. This is
    # the arm that decides the real question, and it has to be able to fire.
    pre, post, condition = synthetic(np.random.default_rng(SEED), carry=0.2, coupling=1.2)
    r = compare(pre, post, condition, seed=SEED)
    assert r["population_over_own"][1] > 0.01, r["population_over_own"]


def test_shuffling_the_state_removes_every_gain():
    pre, post, condition = synthetic(np.random.default_rng(SEED), carry=0.35, coupling=1.2)
    r = compare(pre, post, condition, seed=SEED, shuffle_state=True)
    for name in ("own_over_lookup", "population_over_own"):
        assert r[name][0] < 0.02, f"{name} survives shuffling: {r[name]}"


@pytest.mark.parametrize("seed", [0, 1])
def test_the_reported_split_is_never_fitted(seed):
    # Every arm is built from `train` and tuned on `select`; if the test trials leaked
    # into either, scores would not move when their labels are replaced with noise.
    rng = np.random.default_rng(seed)
    pre, post, condition = synthetic(rng, carry=0.35)
    honest = compare(pre, post, condition, seed=seed)
    corrupted = compare(pre, rng.permutation(post), condition, seed=seed)
    assert honest["own"] > corrupted["own"] + 0.05
