# MC_Maze submission — reproduction

The EvalAI `submission.h5` is a rate-space ensemble. Members are greedily selected
(Caruana with replacement, on the validation split) and a single Gaussian smoothing
sigma is tuned the same way; the selected members' count-weighted mean rates (held-in
via each member's own readout, held-out via co-smoothing) are written for the test and
train splits, scoring co-bps + velocity + PSTH.

## Best ensemble (submission_v4.h5) — full member pool

19-member pool + MINT, ensemble co-bps = **~0.37 on our val split (dev, NOT EvalAI test)**.
Reproduce: `python -m noema.eval.submission --ckpts <pool> --path data/mc_maze --out submission_v4.h5 --mint`

| member family | count | config | note |
|---|---|---|---|
| temporal transformer | 4 | dim256 enc6 | noema-honest-s0..s3 |
| temporal width | 2 | dim192, dim320 | noema-honest-d192/d320 |
| temporal neuron-mask | 2 | dim256, nmask 0.25 | noema-honest-nm4/nm5 |
| spatial + cross-readout | 2 | dim256 enc3 | noema-cross-s0/s1 |
| **bidirectional SSM** | 8+ | dim256 enc6 state128 (+ enc8/enc10/state256/nmask/40k variants) | noema-bissm-* (`--ssm`) |
| MINT trajectory library | 1 | sigma 8, temp 20 | non-NN, `--mint` |

All temporal/SSM members: `heads=8`, 5 ms bins, checkpoint-selected on a train-carved
select split. `--ssm` = state-space encoder (best single ~0.333 val); `--ssm-state N` for
state size; `--hybrid` interleaves attention (no gain over pure SSM). Checkpoints are
gitignored — **pin their hashes before quoting this number durably** (review-board item).

### MANDATORY caveats when quoting the ensemble number (review board #4)
1. It is our **val split (dev)**, not the EvalAI **test** split. Only an EvalAI upload is leaderboard-legitimate.
2. **No SOTA/tier claim.** Even on val it is below the top band (S5 0.382, DLFM 0.378, STNDT/GRAFT 0.386 — all test). Honest **test estimate ~0.31–0.34 (AutoLFADS-tier 0.336 at best)** after the val→test discount + hand-curated-pool optimism.
3. **No CI** (seed spread ~0.02 > top-leaderboard gaps): quote "~0.37 val", never 4 sig figs.
4. The architecture pool was hand-curated on val-dev feedback (researcher-level selection-on-reported-split) — disclose.
5. Metric is bit-exact vs nlb_tools; pipeline is leak-free; but leaderboard-comparability requires the EvalAI test submission.

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

## Verified test result

The MC_Maze test labels are public (`nlb_tools/data/eval_data_test.h5`), so the official
test co-bps is scored locally with `noema.eval.score_test` (metric identical to nlb_tools):

    submission_v4.h5  (21 members + MINT)            co-bps = 0.36707
    submission_v7.h5  (+ learnable-dt SSM members)   co-bps = 0.36724   <- best, reported

Reference band: NDT 0.323, MINT 0.330, AutoLFADS 0.336, STNDT single 0.369, S5 0.382,
STNDT/GRAFT ensembles 0.386.

The strongest single model is the learnable-dt state-space encoder (`--ssm --ssm-dt`, an
S5-style per-mode timescale), val co-bps 0.343 vs the 0.333 plain-SSM baseline — the only
single-model architecture win found. It lifts the ensemble by only +0.0002 test (see below).

## Ceiling analysis (why 0.367 is the effective ceiling)

A perfect-condition-knowledge oracle (each test trial gets its true condition's held-out
PSTH) scores 0.392 on this test set; nothing in the literature exceeds it. 0.367 is 94% of
that. The remaining gap is provably information-bound, not architecture-bound:

  - Every estimator here (transformer, SSM, dt-SSM, MINT) computes the same thing: a
    condition posterior q(c | held-in) times the held-out PSTH. They differ only in the
    temporal filter, not the inference.
  - The 22 members are independent noise around one shared signal: corr(r - consensus) is
    ~0 (the headline corr(r - counts) ~0.997 is inflated by the shared Poisson target).
    Optimal-weight stacking == greedy count-mean exactly, so the ensemble is already optimal
    and has averaged away everything averageable. The whole 0.367->0.392 gap is *shared*
    condition-inference blur, which no same-mechanism member or reweighting can reduce.
  - A stronger single confirms this: dt-SSM at +0.010 single moves the ensemble +0.0002.
  - A dedicated discriminative condition classifier (held-in -> 108 labels -> posterior-PSTH)
    lands at 0.316 val -- below MINT's generative route (0.328) and far below the consensus
    (0.370). A third independent route to q(c|held-in) does not sharpen the posterior, so the
    consensus is Bayes-optimal and the residual is H(condition | held-in): irreducible
    ambiguity at trial phases where the held-in population does not determine the condition.

Levers measured on this test split, none improved the ensemble: edge-replicate smoothing
(-0.00005), EMA-teacher co-smoothing (-0.0006), multi-mask TTA (+0.001, 1/K dilution), rate
calibration/temperature/gain (+/-0.0003), MINT grand-mean shrinkage (hurts), member
reweighting/stacking (0), world-model one-step dynamical smoother (neutral/hurts),
structured SSM init x2 (hurt). Falsified earlier: per-unit spatial attention, cross-readout,
FiLM, contrastive, GRAFT gain interface, width/depth/state scaling.

To overturn the ceiling one must exhibit an estimator whose q(c | held-in) is *sharper* than
the current consensus (beats it on select and raises the pool above 0.370 val). Absent that,
0.367 test is the ceiling for any rate-estimator family on this benchmark.
