"""Assemble the assets the demo page ships, with a manifest describing every one.

The page contains real recordings and two models trained on other sessions of the same
days. A reader who wants to check it should be able to see where each array came from,
so every asset carries a manifest line naming its source file, split, shape and dtype.

Weights are float16: the page previously spent 22 bytes per parameter writing decimal
float32 into JSON, which is what made a 2.5 MB file out of one small model. Two models
and thirteen recordings cost less than that did.
"""

import base64
import gzip
import hashlib

import numpy as np
import torch

# The teacher is the EMA target for the training loss and the behavior head has no panel
# on this page; together they are 36% of the parameters and the browser never reads them.
SHIPPED = ("tokenizer", "encoder", "world")


def _pack(array, dtype):
    """Quantise, gzip and base64 an array, returning the payload and its manifest fields."""
    a = np.ascontiguousarray(array, dtype=dtype)
    blob = gzip.compress(a.tobytes(), 9)
    return {
        "b64": base64.b64encode(blob).decode(),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "bytes": len(blob),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest()[:16],
    }


def pack_weights(model):
    """The parameters the browser needs, in a fixed order, as one float16 block.

    Returns the packed block and the key order, so the JS side slices by name rather
    than trusting that two languages iterate a dict the same way.
    """
    named = [(n, p) for n, p in model.named_parameters() if n.split(".")[0] in SHIPPED]
    flat = np.concatenate([p.detach().float().numpy().astype(np.float16).ravel() for _, p in named])
    layout = [{"name": n, "shape": list(p.shape), "n": p.numel()} for n, p in named]
    return _pack(flat, np.float16), layout


def round_to_f16(model):
    """Round a model's weights through float16 in place.

    Any reference the parity gate compares against must come from the rounded weights,
    because those are the weights the browser runs. Quantisation alone moves a ten-step
    rollout by about 1.1e-3 against a 2e-3 gate — it would pass either way, but it would
    consume half the budget and leave nothing to detect a real defect with.
    """
    return model.half().float().eval()


def pack_tapes(tapes):
    """The shipped recordings, one manifest entry per day."""
    out = []
    for day, split, session, counts in tapes:
        entry = _pack(counts, np.uint8)
        entry.update(day=day, split=split, session=session,
                     source=f"DANDI:000954 sub-HumanPitt {split}-calib ses-{session}")
        out.append(entry)
    return out


@torch.no_grad()
def parity_fixtures(models, tape, window, horizon, cuts=8):
    """Reference rollouts the JS forward pass must reproduce.

    Cuts are spread across the tape rather than clustered, so a bug that only shows on
    quiet or busy stretches still has somewhere to surface.
    """
    from demo.measure import cut_grid

    grid = cut_grid(len(tape), window, horizon, cuts)
    t = torch.as_tensor(np.asarray(tape), dtype=torch.float32)
    ids = torch.arange(t.shape[1])
    out = []
    for cut in grid:
        seed = t[cut - window:cut].unsqueeze(0)
        entry = {"cut": int(cut), "seed": _pack(seed[0].numpy(), np.uint8)}
        for tag, model in models.items():
            from noema.sim.rollout import imagine
            rates, _ = imagine(model, seed, ids, torch.zeros(1, horizon, 0))
            entry[tag] = _pack(rates[0].numpy(), np.float32)
        out.append(entry)
    return out


def build_all(window=50, horizon=10, cuts=100, out="demo/assets.json"):
    """Every asset the page ships, written as one JSON bundle.

    Both models are rounded to float16 before anything is measured or referenced, so the
    curves on the page and the parity fixtures both describe the weights that actually run.
    """
    import json
    import pathlib

    from falcon_challenge.config import FalconConfig, FalconTask

    from demo.measure import forecast_skill
    from demo.tapes import assert_disjoint, extract, training_paths
    from noema import Noema
    from noema.eval.falcon import load_sessions

    cfg = FalconConfig(task=FalconTask["h1"])
    tapes = extract()
    # Check against what the models were ACTUALLY trained on, recorded at training time,
    # not against what we believe the training set to be. The first version of this guard
    # compared tapes to a declared file list while the trainer was loading every session
    # in the directory, so six of the thirteen tapes were recordings the model had read
    # and the check still passed. A guard fed its own assumption proves nothing.
    trained = pathlib.Path("checkpoints/demo-train-files.txt")
    if not trained.exists():
        raise RuntimeError("checkpoints/demo-train-files.txt is missing: retrain with "
                           "--train-files so the shipped tapes can be checked against the "
                           "files the model actually read")
    used = [p for p in trained.read_text().split() if p]
    assert set(used) == set(training_paths()), "the models were trained on a different file set"
    training = [n for path in used for _, n, _, _ in load_sessions(path)]
    assert_disjoint([a for *_, a in tapes], training)  # never ship a recording the model read

    models = {}
    for tag, path in (("multistep", "checkpoints/demo-multistep.pt"),
                      ("onestep", "checkpoints/demo-onestep.pt")):
        m = Noema(dim=96, enc_depth=3, wm_depth=2, heads=8,
                  max_units=cfg.n_channels, behavior_dim=cfg.out_dim)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[tag] = round_to_f16(m)

    weights, calendar = {}, []
    for tag, model in models.items():
        packed, layout = pack_weights(model)
        weights[tag] = {**packed, "layout": layout, "name": tag}
    for day, split, session, counts in tapes:
        row = {"day": day, "split": split, "session": session}
        for tag, model in models.items():
            row[tag] = {k: [round(float(x), 5) for x in v]
                        for k, v in forecast_skill(model, counts, window, horizon, cuts).items()}
        calendar.append(row)
        print(f"  day {day:>2} {split:<9} centred h1={row['multistep']['centred'][0]:.3f} "
              f"h{horizon}={row['multistep']['centred'][-1]:.3f}", flush=True)

    fixtures = parity_fixtures(models, tapes[0][3], window, horizon, cuts=8)
    bundle = {
        "config": {"dim": 96, "enc_depth": 3, "wm_depth": 2, "heads": 8,
                   "channels": cfg.n_channels, "window": window, "horizon": horizon,
                   "bin_ms": 20, "cuts_per_day": cuts},
        "weights": weights, "tapes": pack_tapes(tapes),
        "calendar": calendar, "fixtures": fixtures,
    }
    bundle["manifest"] = manifest({"weights": list(weights.values()),
                                   "tapes": bundle["tapes"], "fixtures": []})
    pathlib.Path(out).write_text(json.dumps(bundle))
    size = pathlib.Path(out).stat().st_size
    print(f"wrote {out}: {size/1e6:.2f} MB", flush=True)
    return bundle


def manifest(assets):
    """A flat description of everything shipped, so the page can print its own provenance."""
    lines = []
    for kind, items in assets.items():
        for item in items if isinstance(items, list) else [items]:
            lines.append({k: v for k, v in item.items() if k != "b64"} | {"kind": kind})
    total = sum(entry["bytes"] for entry in lines)
    return {"assets": lines, "compressed_bytes": total,
            "base64_bytes": int(total * 4 / 3)}
