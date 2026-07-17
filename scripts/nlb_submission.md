# MC_Maze submission — reproduction

The EvalAI `submission.h5` is a rate-space ensemble. Members are greedily selected
(Caruana with replacement, on the validation split) and a single Gaussian smoothing
sigma is tuned the same way; the selected members' count-weighted mean rates (held-in
via each member's own readout, held-out via co-smoothing) are written for the test and
train splits, scoring co-bps + velocity + PSTH.

The forward-rollout keys (fp-bps) are omitted by default (`--forward` to opt in): a
one-step-trained ensemble diverges in open loop, so its rollout is unreliable and would
poison the mean. Only include fp with a world model trained for multi-step rollout.

## Ensemble members (10)

| checkpoint prefix | dim | enc | encoder | note |
|---|---|---|---|---|
| noema-honest-s0..s3 | 256 | 6 | temporal | seeds 0–3 |
| noema-honest-d192 | 192 | 6 | temporal | width variant |
| noema-honest-d320 | 320 | 6 | temporal | width variant |
| noema-honest-nm4, nm5 | 256 | 6 | temporal | random-neuron co-smoothing (MtM) |
| noema-cross-s0, s1 | 256 | 3 | spatial + cross-attention readout | seeds 0/1 |

All members use `heads=8`, 5 ms bins, and are trained with val-co-bps checkpoint
selection on a train-carved select split (see `noema/train/run.py`).

The temporal members are the strongest (val co-bps ~0.27–0.30); the spatial + cross
members plateau lower (~0.24–0.26) and neither matches nor beats the temporal ensemble —
greedy selection down-weights them accordingly. Ensembling (~0.33 val) is the only lever
that beats the best single member; no single architecture wins on this benchmark.

## Regenerate

On the training cluster (checkpoints in `checkpoints/`, dandiset in `data/mc_maze`):

```
sbatch --export=ALL,PREFIX=noema-honest:noema-cross-s,DATASET=mc_maze,OUT=submission.h5 \
       scripts/farmshare_submission.sbatch
```

This globs the member checkpoints, runs `noema.eval.submission`, and writes the `.h5`.
The generator asserts the sequestered test split is present (fails loudly otherwise) and
prints the trial/unit counts; confirm they match the dataset before uploading.

## Submit

Upload the `.h5` to the EvalAI MC_Maze challenge. The returned test-split co-bps is the
only leaderboard-legitimate figure; locally computed validation numbers are development
metrics and are not directly comparable to published test-split results.
