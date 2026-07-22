import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import LinearSpikeSystem
from noema.eval.baselines import ridge_velocity
from noema.eval.sim2real import decoder_in_imagination
from noema.train import TrainConfig, train

CPU = torch.device("cpu")


def test_decoder_trained_in_imagination_transfers_to_real():
    torch.manual_seed(0)
    system = LinearSpikeSystem(units=30, latent=6, action_dim=2, seed=1)
    counts, unit_ids, actions, behavior = system.sample(batch=256, steps=40)
    ds = SpikeWindows(counts, behavior=behavior, actions=actions)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=ds.collate, drop_last=True)
    model = Noema(dim=64, enc_depth=2, wm_depth=2, heads=4, max_units=30,
                  action_dim=2, behavior_dim=2)
    train(model, loader, TrainConfig(steps=500, warmup=40, lr=3e-3, w_forecast=2.0, ckpt=""), device=CPU)

    c, _, _, b = system.sample(batch=96, steps=40)
    real_val = SpikeWindows(c[32:], behavior=b[32:])
    seed_ds = SpikeWindows(counts[:96], behavior=behavior[:96], actions=actions[:96])

    imagined = decoder_in_imagination(model, unit_ids, seed_ds, real_val, episodes=96, horizon=15)["sim2real_r2"]
    real_ceiling = ridge_velocity(SpikeWindows(counts[:96], behavior=behavior[:96]), real_val)

    assert imagined > 0.2          # imagined data alone decodes real activity
    assert imagined < real_ceiling  # but not beyond training on real data — no free lunch
