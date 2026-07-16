import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import synthetic_batch
from noema.eval.ensemble import ensemble_co_bps
from noema.eval.nlb import evaluate
from noema.train import TrainConfig, train

CPU = torch.device("cpu")


def test_ensemble_not_worse_than_worst_member():
    torch.manual_seed(0)
    counts, _, behavior = synthetic_batch(batch=128, steps=25, units=40, seed=0)
    ds = SpikeWindows(counts[..., :28], counts[..., 28:], behavior)
    loader = DataLoader(ds, batch_size=32, collate_fn=ds.collate, drop_last=True)

    models = []
    for _ in range(2):
        m = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=40)
        train(m, loader, TrainConfig(steps=80, warmup=5, lr=3e-3, ckpt=""), device=CPU)
        models.append(m)

    singles = [evaluate(m, ds)["co_bps"] for m in models]
    ensemble = ensemble_co_bps(models, ds, device=CPU)
    assert ensemble >= min(singles) - 0.05  # rate averaging never trails the worst member


def test_build_from_state_recovers_head_count():
    # a non-default head count must survive the checkpoint round-trip, else attention
    # is silently repartitioned into the wrong number of heads (mechanism corrupted)
    from noema.eval.ensemble_run import build_from_state

    torch.manual_seed(0)
    m = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=40, spatial=True, cross=True).eval()
    counts = torch.rand(2, 6, 20)
    in_ids, out_ids = torch.arange(20), torch.arange(8) + 20
    ref = m.cosmooth(counts, in_ids, out_ids)

    rebuilt, _ = build_from_state(m.state_dict(), max_units=40, heads=8)  # 8 is the wrong fallback
    rebuilt.eval()
    assert int(rebuilt.num_heads.item()) == 4
    assert torch.allclose(rebuilt.cosmooth(counts, in_ids, out_ids), ref, atol=1e-5)


def test_greedy_ensemble_beats_naive_mean():
    from noema.eval.ensemble import greedy_ensemble, member_rates
    from noema.eval.metrics import bits_per_spike

    torch.manual_seed(1)
    counts, _, behavior = synthetic_batch(batch=128, steps=25, units=40, seed=1)
    ds = SpikeWindows(counts[..., :28], counts[..., 28:], behavior)
    loader = DataLoader(ds, batch_size=32, collate_fn=ds.collate, drop_last=True)
    models = []
    for _ in range(4):
        m = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=40)
        train(m, loader, TrainConfig(steps=50, warmup=5, lr=3e-3, ckpt=""), device=CPU)
        models.append(m)

    rates, targets = member_rates(models, ds, device=CPU)
    chosen = greedy_ensemble(rates, targets)
    greedy = bits_per_spike(sum(rates[j] for j in chosen) / len(chosen), targets)
    assert greedy >= ensemble_co_bps(models, ds, device=CPU) - 0.005  # selection ≥ naive mean
