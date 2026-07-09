import torch

from noema import Noema
from noema.eval.streaming import StreamingDecoder


def _run(dec, stream):
    dec.reset(batch_size=stream[0].size(0))
    return [dec.step(o) for o in stream]


def test_streaming_decode_shaped_finite_and_deterministic():
    torch.manual_seed(0)
    model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=30, behavior_dim=2)
    dec = StreamingDecoder(model, unit_ids=torch.arange(30), window=20)
    stream = [torch.poisson(torch.rand(3, 30)) for _ in range(25)]

    first = _run(dec, stream)
    for out in first:
        assert out.shape == (3, 2) and torch.isfinite(out).all()

    replay = _run(dec, stream)  # same stream -> same decode (deterministic, causal)
    assert torch.allclose(first[-1], replay[-1])
