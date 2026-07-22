import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import synthetic_batch
from noema.train import TrainConfig, train


def test_cross_readout_cosmooths_and_shapes():
    torch.manual_seed(0)
    counts, _, behavior = synthetic_batch(batch=64, steps=25, units=40, seed=0)
    ds = SpikeWindows(counts[..., :28], counts[..., 28:], behavior)
    loader = DataLoader(ds, batch_size=16, collate_fn=ds.collate, drop_last=True)
    model = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=40,
                  behavior_dim=2, spatial=True, cross=True)

    b = next(iter(loader))
    pred = model.cosmooth(b["counts"], b["unit_ids"], b["target_unit_ids"])
    assert pred.shape == (16, 25, 12)  # [B, T, n_held_out] — attends per held-out unit

    logs = []
    train(model, loader, TrainConfig(steps=60, warmup=5, lr=3e-3, ckpt=""),
          device=torch.device("cpu"), on_log=lambda s, d: logs.append(d))
    assert logs[-1]["loss_cosmooth"] < logs[0]["loss_cosmooth"]  # the cross readout learns


def test_cross_readout_is_per_unit_equivariant():
    # each output row must correspond to its own query unit (guards the multi-head
    # query reshape from scrambling unit/head axes)
    from noema.models.heads import CrossReadout
    torch.manual_seed(0)
    cr = CrossReadout(dim=16, heads=4).eval()
    tokens, queries = torch.randn(2, 3, 5, 16), torch.randn(3, 16)
    with torch.no_grad():
        out = cr(tokens, queries)
        perm = [2, 0, 1]
        permuted = cr(tokens, queries[perm])
    for i in range(3):
        assert torch.allclose(permuted[..., i], out[..., perm[i]], atol=1e-5)
