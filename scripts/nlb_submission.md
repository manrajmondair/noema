# MC_Maze submission — reproduction

The EvalAI `submission.h5` is a rate-space ensemble. Members are greedily selected
(Caruana, on the validation split) and a single Gaussian smoothing sigma is tuned the
same way; the selected members' mean rates (held-in via each member's own readout,
held-out via co-smoothing, plus a forward rollout for fp-bps) are written for the test
and train splits.

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
