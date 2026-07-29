import numpy as np
import pytest

from noema import Noema

pytest.importorskip("falcon_challenge")


def test_disjoint_calib_removes_the_minival_overlap():
    # The held-in-minival recordings are a byte-identical prefix of the matching
    # held-in-calib recordings, so fitting on calib and scoring minival scores a model
    # on its own training data. Guard the excision, and the session matching it relies on.
    from noema.eval.falcon import disjoint_calib

    session = np.arange(100).reshape(50, 2).astype("float32")
    calib = [("sub-held-in-calib_ses-A", session, session[:, :1], np.ones(50, bool))]
    minival = [("sub-held-in-minival_ses-A", session[:20], session[:20, :1], None)]

    (_, neural, kin, mask), = disjoint_calib(calib, minival)
    assert len(neural) == 30 and len(kin) == 30 and len(mask) == 30
    assert np.array_equal(neural, session[20:])  # the overlap, and only the overlap, is gone
    # a session with no minival counterpart must survive untouched
    other = [("sub-held-in-calib_ses-B", session, session[:, :1], None)]
    assert len(disjoint_calib(other, minival)[0][1]) == 50


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
