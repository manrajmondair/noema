import copy

import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import MultiSessionSystem
from noema.eval.nlb import evaluate
from noema.train import TrainConfig, few_shot_adapt, train

CPU = torch.device("cpu")


def _loader(system, session, batch, steps=30, label=None):
    c, _, a, b = system.sample(session, batch=batch, steps=steps)
    ds = SpikeWindows(c, behavior=b, actions=a, unit_ids=system.unit_ids(session), session=label)
    return ds, DataLoader(ds, batch_size=min(batch, 8), shuffle=True,
                          collate_fn=ds.collate, drop_last=True)


def test_adapt_touches_only_new_units():
    torch.manual_seed(0)
    system = MultiSessionSystem(sessions=3, units=20, latent=5, seed=3)
    model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=60,
                  action_dim=2, behavior_dim=2)
    enc0 = [p.clone() for p in model.encoder.parameters()]
    beh0 = [p.clone() for p in model.behavior.parameters()]
    emb0 = model.tokenizer.embed.weight.detach().clone()

    _, loader = _loader(system, 2, batch=16)  # session 2 owns ids 40..59
    few_shot_adapt(model, loader, steps=20, device=CPU)

    for p, q in zip(model.encoder.parameters(), enc0):
        assert torch.equal(p, q)  # dynamics backbone frozen
    for p, q in zip(model.behavior.parameters(), beh0):
        assert torch.equal(p, q)

    moved = (model.tokenizer.embed.weight.detach() - emb0).abs().sum(-1)
    used = set(system.unit_ids(2).tolist())
    assert moved[sorted(used)].min() > 0                                   # new units learned
    assert moved[[i for i in range(60) if i not in used]].max() == 0       # others untouched


def test_session_invariant_transfer():
    # Pretraining over enough sessions with an adversarial session-invariance term
    # yields a latent a held-out population can route into: few-shot calibration then
    # decodes a session never seen in training.
    torch.manual_seed(0)
    pretrain = 6
    system = MultiSessionSystem(sessions=pretrain + 1, units=30, latent=6, seed=2)
    batches = []
    for s in range(pretrain):
        _, loader = _loader(system, s, batch=128, steps=40, label=s)
        batches += list(loader)
    model = Noema(dim=96, enc_depth=3, wm_depth=2, heads=4, max_units=30 * (pretrain + 1),
                  action_dim=2, behavior_dim=2, sessions=pretrain, adv_weight=0.3)
    train(model, batches, TrainConfig(steps=500, warmup=40, lr=3e-3, ckpt=""), device=CPU)

    test_ds, _ = _loader(system, pretrain, batch=64, steps=40)   # unseen session
    _, calib = _loader(system, pretrain, batch=32, steps=40)
    before = evaluate(copy.deepcopy(model), test_ds, device=CPU)["vel_r2"]
    after = evaluate(few_shot_adapt(copy.deepcopy(model), calib, steps=150, device=CPU),
                     test_ds, device=CPU)["vel_r2"]
    assert after > 0.2          # decodes a session whose units were never trained on
    assert after > before + 0.3  # calibration drives it from the cold backbone
