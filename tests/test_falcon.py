import numpy as np
import pytest

from noema import Noema

pytest.importorskip("falcon_challenge")


def test_disjoint_calib_removes_the_minival_overlap():
    # The held-in-minival recordings are a byte-identical prefix of the matching
    # held-in-calib recordings, so fitting on calib and scoring minival scores a model
    # on its own training data. Guard the excision, and the session matching it relies on.
    from noema.eval.falcon import disjoint_calib

    # load_sessions now yields a parsed session id, so the two splits meet on that key
    # rather than on filenames that differ everywhere except a substring.
    session = np.arange(100).reshape(50, 2).astype("float32")
    calib = [("A", session, session[:, :1], np.ones(50, bool))]
    minival = [("A", session[:20], session[:20, :1], None)]

    (_, neural, kin, mask), = disjoint_calib(calib, minival)
    assert len(neural) == 30 and len(kin) == 30 and len(mask) == 30
    assert np.array_equal(neural, session[20:])  # the overlap, and only the overlap, is gone


def test_an_unexcisable_session_raises_instead_of_passing_through():
    # This case used to be asserted the other way round: a calib session with no minival
    # counterpart survived untouched, and the test called that correct. But the only way
    # to reach it on this dandiset is a lookup that failed, and the consequence is
    # training on the overlap and reporting a higher score with no error — the exact
    # shape of the defect the excision exists to undo.
    from noema.eval.falcon import disjoint_calib

    session = np.arange(100).reshape(50, 2).astype("float32")
    minival = [("A", session[:20], session[:20, :1], None)]

    with pytest.raises(ValueError, match="no minival counterpart"):
        disjoint_calib([("B", session, session[:, :1], None)], minival)


def test_cutting_the_wrong_rows_raises():
    # The cut is a length taken from the minival recording. If minival ever stops being
    # a prefix, that length removes bins that were never in it and leaves the overlap
    # behind, silently. Verify the prefix rather than assume it.
    from noema.eval.falcon import disjoint_calib

    session = np.arange(100).reshape(50, 2).astype("float32")
    not_a_prefix = session[20:40]  # right length, wrong rows
    with pytest.raises(ValueError, match="not a prefix"):
        disjoint_calib([("A", session, session[:, :1], None)],
                       [("A", not_a_prefix, not_a_prefix[:, :1], None)])


def test_the_session_key_is_found_wherever_it_sits():
    # A fixed offset into the filename used to supply this, and it landed mid-word.
    from noema.eval.falcon import session_id

    assert session_id("x/sub-HumanPitt-held-in-calib_ses-19250101T111740.nwb") == "19250101T111740"
    assert session_id("y/sub-HumanPitt-held-in-minival_ses-19250101T111740.nwb") == "19250101T111740"
    assert session_id("z/a_ses-B.nwb") == "B"
    with pytest.raises(ValueError):
        session_id("no-session-here.nwb")


def test_decoder_satisfies_falcon_interface():
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.interface import BCIDecoder
    from noema.eval.falcon import make_decoder

    cfg = FalconConfig(task=FalconTask.m1)
    model = Noema(dim=48, enc_depth=1, wm_depth=1, heads=4,
                  max_units=cfg.n_channels, behavior_dim=cfg.out_dim)
    decoder = make_decoder(model, cfg, window=16)

    assert isinstance(decoder, BCIDecoder)
    decoder.reset([""])
    for _ in range(20):  # stream single-timestep spike bins as the evaluator does
        out = decoder.predict(np.random.poisson(1.0, (1, cfg.n_channels)).astype("float32"))
        assert out.shape == (1, cfg.out_dim) and np.isfinite(out).all()


def test_temporal_metric_recovers_a_tracked_trajectory():
    # A null result is only worth reporting from an instrument that can return a
    # positive one, so plant a model that genuinely follows each channel's time course.
    from noema.eval.falcon_worldmodel import _temporal_fidelity

    rng = np.random.default_rng(0)
    true = rng.normal(size=(40, 10, 20))
    tracked = _temporal_fidelity([(true + 0.4 * rng.normal(size=true.shape), true)])
    assert tracked["model"] > 0.8
    assert tracked["model"] - tracked["shuffled"] > 0.5  # the ORDER is what it recovers

    noise = _temporal_fidelity([(rng.normal(size=true.shape), true)])
    assert abs(noise["model"]) < 0.05


def test_a_flat_forecast_cannot_score_on_the_temporal_metric():
    # This is the whole reason the metric exists. The population metric is beaten by a
    # constant; a constant has no time variance, so here it is undefined rather than
    # merely poor, and the two metrics are therefore asking different questions.
    from noema.eval.falcon_worldmodel import _temporal_fidelity

    rng = np.random.default_rng(0)
    true = rng.normal(size=(40, 10, 20))
    flat = np.repeat(true.mean(1, keepdims=True), 10, axis=1)
    assert np.isnan(_temporal_fidelity([(flat, true)])["model"])
