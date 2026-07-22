# MC_Maze submission

The submission `.h5` is a rate-space ensemble. Members are selected greedily (Caruana, with
replacement) and a single Gaussian smoothing width is tuned the same way, both on a train-carved
select split. The selected members' count-weighted mean rates — held-in via each member's own
readout, held-out via co-smoothing — are written for the test and train splits.

## Members

| family | count | config |
|---|---|---|
| temporal transformer | 4 | dim 256, enc 6, seeds 0–3 |
| temporal, width variants | 2 | dim 192, dim 320 |
| temporal, random-neuron co-smoothing | 2 | dim 256, neuron-mask 0.25 |
| spatial + cross-attention readout | 2 | dim 256, enc 3 |
| bidirectional state-space | 11 | dim 256, enc 6, state 128 (+ depth/state/seed variants) |
| learnable-timescale state-space | 4 | dim 256, enc 6, `--ssm-dt`, seeds 0–3 |
| MINT trajectory library | 1 | condition-mix, sigma 8, temp 20 |

All neural members use `heads=8`, 5 ms bins, and val-co-bps checkpoint selection on the select
split. The state-space members are the strongest singles; MINT is a non-neural member that shares
no failure modes with them. Greedy selection down-weights the weaker families.

## Regenerate

With the checkpoints in `checkpoints/` and the dandiset in `data/mc_maze`:

```
python -m noema.eval.submission \
  --ckpts "$(ls checkpoints/*.pt | paste -sd,)" \
  --name mc_maze --path data/mc_maze --out submission.h5 --mint
```

The generator checks that the test split is present and prints the trial and unit counts.

## Test result

The MC_Maze test labels are public (`nlb_tools/data/eval_data_test.h5`), so co-bps is scored
locally with the official metric:

```
python -m noema.eval.score_test --submission submission.h5
```

The best ensemble scores **0.367** co-bps on the test split via
`nlb_tools.evaluation.bits_per_spike`. Reference band: NDT 0.323, MINT 0.330, AutoLFADS 0.336,
STNDT single 0.369, S5 0.382, STNDT/GRAFT ensembles 0.386.

## Ceiling

A perfect-condition-knowledge oracle — each trial scored with its true-condition held-out PSTH —
reaches 0.392 on this test set, and no published method exceeds it. 0.367 is 94% of that, and the
remaining gap is information-bound rather than architectural:

- Every member computes a condition posterior from the held-in population, weighting the held-out
  PSTH, and differs only in the temporal filter. Per-trial deviations from the ensemble mean are
  uncorrelated, so the ensemble has already averaged away everything averageable; optimal linear
  reweighting of the members equals the greedy mean exactly.
- A single model 0.010 co-bps stronger (the learnable-timescale state-space encoder) moves the
  ensemble by 0.0002.
- A dedicated held-in-to-condition classifier produces a posterior no sharper than the ensemble
  consensus.

What remains is the conditional entropy of the reach given the held-in spikes: trials whose
observed neurons do not determine the condition. Inference-side and architectural changes that were
measured but did not improve the ensemble include edge-replicate smoothing, EMA-teacher
co-smoothing, multi-mask test-time augmentation, rate calibration, MINT shrinkage, member
reweighting, a one-step world-model smoother, structured state-space initialization, per-unit
spatial attention, cross-attention readout, nonlinear readouts, contrastive objectives, a
per-neuron gain interface, and width/depth scaling.
