import numpy as np
import pytest

from noema import Noema

pytest.importorskip("falcon_challenge")


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
