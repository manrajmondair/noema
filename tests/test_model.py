import torch

from noema import Noema
from noema.data.synthetic import synthetic_batch


def test_shapes():
    counts, unit_ids, behavior = synthetic_batch(batch=4, steps=20, units=30, behavior_dim=2)
    model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=32, behavior_dim=2)
    out = model(counts, unit_ids, behavior=behavior)
    assert out["rate"].shape == counts.shape
    assert out["z"].shape == (4, 20, 64)
    assert out["loss_rate"].item() > 0


def test_rollout_shape():
    from noema.sim.rollout import imagine

    counts, unit_ids, _ = synthetic_batch(batch=2, steps=16, units=25, behavior_dim=0)
    model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=32)
    future = torch.zeros(2, 8, 0)  # no actions in the unconditioned model
    rates, behavior = imagine(model, counts, unit_ids, future)
    assert rates.shape == (2, 8, 25)  # imagined future firing rates, seed excluded
    assert behavior is None


def test_multistep_loss_is_opt_in_and_rolls_out():
    counts, unit_ids, _ = synthetic_batch(batch=8, steps=20, units=30)
    actions = torch.randn(8, 20, 2)
    # default: no multi-step term
    off = Noema(dim=48, enc_depth=2, wm_depth=2, heads=4, max_units=32, action_dim=2)
    assert "loss_multistep" not in off(counts, unit_ids, actions=actions)
    # opt-in: present and finite, both action-conditioned and unconditioned
    on = Noema(dim=48, enc_depth=2, wm_depth=2, heads=4, max_units=32, action_dim=2, multistep=6)
    for a in (actions, None):
        ms = on(counts, unit_ids, actions=a)["loss_multistep"]
        assert torch.isfinite(ms) and ms.item() > 0


def test_overfits_single_batch():
    torch.manual_seed(0)
    counts, unit_ids, behavior = synthetic_batch(batch=16, steps=40, units=50, behavior_dim=2)
    model = Noema(dim=128, enc_depth=3, wm_depth=2, heads=4, max_units=64, behavior_dim=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for step in range(300):
        out = model(counts, unit_ids, behavior=behavior)
        loss = out["loss_rate"] + out["loss_jepa"] + 5 * out["loss_behavior"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_teacher()
        if step == 0:
            first = (out["loss_rate"].item(), out["loss_behavior"].item())

    assert out["loss_rate"].item() < 0.6 * first[0]      # firing rates fit
    assert out["loss_behavior"].item() < 0.2 * first[1]  # behavior decodes
