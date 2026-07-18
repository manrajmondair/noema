# MINT ensemble member — MC_Maze co-bps

`noema/eval/mint.py` is a non-neural-net rate estimator (Perich/Kaufman-style trajectory
library) intended as a **decorrelated member** for the rate-space ensemble. Because it
shares no failure modes with the transformer members, it is the highest-leverage addition
on our one proven lever (ensembling).

## Result (real MC_Maze, our val split)

**co-bps = 0.328** — board-clean:
- condition-mix readout, `sigma=8`, `temp=20`;
- (sigma, temp) tuned on a **train-carved select split** and reported on the untouched
  val split (the select-tuned config independently matches the val peak, so no
  selection-on-reported-split bias);
- metric bit-exact vs `nlb_tools.evaluation.bits_per_spike` (1.7e-16);
- library built from train only (no val leakage); conditions = `['trial_type',
  'trial_version']` (the canonical 108-condition MC_Maze grouping), aligned to the exact
  trials `make_train_input_tensors` returns.

This matches the published MINT (~0.33) and exceeds our transformer single (0.273);
it is comparable to the full transformer ensemble (~0.33) via a completely different method.

## Readout choice

- **condition-mix** (soft whole-trial condition assignment) wins on real NLB data because
  trials are aligned to movement onset (little residual timing jitter).
- **state-match** (per-timestep library-state matching) wins only when trials are
  time-warped; it underperforms here (~0.23). Kept as an option for un-aligned datasets.

## Reproduce

```
python -m noema.eval.mint --path data/mc_maze --name mc_maze   # co-bps on val
```

Env note (nlb_tools vs modern pandas): needs `numpy<2`, `pandas==2.0.3`, and three
`nwb_interface.py` patches — writable `to_numpy()` copy in `resample`, a try/except around
`index.freq`, and `searchsorted` slicing in `make_trial_df`. (On the cluster's pandas-1.3.4
conda env these are unnecessary.)

## Next: fold into the ensemble (needs the transformer checkpoints)

Add MINT's held-out rates as an extra member in `ensemble_run`/`submission` greedy
selection: MINT val rates align with the transformer members' val rates (same
`make_train_input_tensors` order); for the select split, build the MINT library on the
core trials and predict the select trials (matching how the transformers are trained on
core and scored on select). Expected: two decorrelated ~0.33 predictors blend above 0.33.
