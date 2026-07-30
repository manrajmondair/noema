"""Imagined rollouts must carry decodable content, on the branch that is actually used.

This previously exercised `decoder_in_imagination`, a convenience wrapper with no call
site anywhere. The live path is `imagine_windows`, which the FALCON factorial drives
unconditioned — spontaneous recordings have no observed control input.
"""

import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import LinearSpikeSystem
from noema.eval.baselines import ridge_velocity
from noema.eval.sim2real import imagine_windows
from noema.train import TrainConfig, train

CPU = torch.device("cpu")


def test_imagined_windows_decode_above_chance_on_real_data():
    torch.manual_seed(0)
    system = LinearSpikeSystem(units=30, latent=6, action_dim=2, seed=1)
    counts, unit_ids, actions, behavior = system.sample(batch=256, steps=40)
    ds = SpikeWindows(counts, behavior=behavior, actions=actions)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=ds.collate, drop_last=True)
    model = Noema(dim=64, enc_depth=2, wm_depth=2, heads=4, max_units=30,
                  action_dim=2, behavior_dim=2)
    train(model, loader, TrainConfig(steps=500, warmup=40, lr=3e-3, w_forecast=2.0, ckpt=""),
          device=CPU)

    c, _, _, b = system.sample(batch=96, steps=40)
    real_val = SpikeWindows(c[32:], behavior=b[32:])
    seeds = torch.as_tensor(counts[:96, :25], dtype=torch.float32)

    imagined = imagine_windows(model, unit_ids, seeds, horizon=15, device=CPU)
    assert imagined.heldin.shape[0] == seeds.shape[0]
    assert torch.isfinite(imagined.heldin).all() and (imagined.heldin >= 0).all()

    # A decoder fitted only on imagined spikes, scored on real held-out recordings.
    trained = ridge_velocity(SpikeWindows(imagined.heldin, behavior=imagined.behavior), real_val)
    chance = ridge_velocity(
        SpikeWindows(imagined.heldin, behavior=imagined.behavior[torch.randperm(len(imagined.behavior))]),
        real_val)
    assert trained > chance
