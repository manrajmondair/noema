import torch

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.pretrain import combine_sessions
from noema.data.synthetic import MultiSessionSystem
from noema.train import TrainConfig, train


def _session(system, s, batch=32, steps=20):
    counts, _, actions, behavior = system.sample(s, batch=batch, steps=steps)
    return SpikeWindows(counts, behavior=behavior, actions=actions)


def test_combine_sessions_shared_disjoint_space():
    system = MultiSessionSystem(sessions=3, units=20, latent=5, seed=0)
    batches, max_units, n = combine_sessions([_session(system, s) for s in range(3)], batch_size=16)

    assert (n, max_units) == (3, 60)  # 3 sessions x 20 units, no overlap
    assert {int(b["session"][0]) for b in batches} == {0, 1, 2}
    ranges = {int(b["session"][0]): (b["unit_ids"].min().item(), b["unit_ids"].max().item())
              for b in batches}
    assert ranges[0] == (0, 19) and ranges[1] == (20, 39) and ranges[2] == (40, 59)


def test_pretraining_runs_with_adversary():
    system = MultiSessionSystem(sessions=3, units=20, latent=5, seed=1)
    batches, max_units, n = combine_sessions([_session(system, s) for s in range(3)], batch_size=16)
    model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=max_units,
                  action_dim=2, behavior_dim=2, sessions=n)

    logs = []
    train(model, batches, TrainConfig(steps=30, warmup=5, lr=3e-3, log_every=1, ckpt=""),
          device=torch.device("cpu"), on_log=lambda s, d: logs.append(d))
    assert any("loss_session" in d for d in logs)  # invariance term active across sessions


def test_checkpoint_warm_starts_backbone(tmp_path):
    system = MultiSessionSystem(sessions=2, units=20, latent=5, seed=2)
    batches, max_units, n = combine_sessions([_session(system, s) for s in range(2)], batch_size=16)
    ckpt = str(tmp_path / "pretrained.pt")
    pre = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=max_units, sessions=n)
    train(pre, batches, TrainConfig(steps=20, warmup=5, lr=3e-3, ckpt=ckpt), device=torch.device("cpu"))

    # A fine-tune model adds a behavior head the pretrained backbone lacks.
    fine = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=max_units, behavior_dim=2)
    result = fine.load_state_dict(torch.load(ckpt), strict=False)

    assert torch.equal(fine.encoder.norm.weight, pre.encoder.norm.weight)   # backbone transferred
    assert torch.equal(fine.tokenizer.embed.weight, pre.tokenizer.embed.weight)
    assert all("behavior" in k for k in result.missing_keys)                # only the new head is fresh
