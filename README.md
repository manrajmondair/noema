# Noema

A world model for neural population activity.

Most neural decoders treat the brain as a signal to classify: activity in, intent out. Noema takes the opposite stance — it models the population as a dynamical system and learns to predict its own future. Concretely, it fits

```
z_{t+1} ~ P(· | z_t, action_t, context_t)
```

over a learned latent state `z`. A forward model of this kind gives decoding, drift-robust calibration, a closed-loop simulator, and a per-subject digital twin as downstream consequences of a single objective.

## Design

- **Tokenizer** — every recorded unit owns a learned embedding, so a session's population vector becomes one token per time bin regardless of channel count or layout. Permutation-invariant and portable across sessions and subjects.
- **Encoder** — a rotary temporal transformer maps the token stream to latent states `z_t`.
- **World model** — an action-conditioned, causal predictor trained to forecast the next latent (joint-embedding style, against an EMA target encoder), and to roll out autoregressively.
- **Heads** — a Poisson readout for firing rates (co-smoothing) and a behavior decoder for kinematics.

## Layout

```
noema/
  models/   tokenizer, encoder, world model, sensory coupling, session adversary
  data/     dataset, synthetic systems, multi-session pretraining
  train/    trainer, few-shot adaptation, CLI
  sim/      imagined rollouts
  eval/     metrics, baselines, NLB, FALCON, streaming, calibration, sim2real
demo/       interactive in-browser world model (parity-checked against PyTorch)
tests/      unit + integration checks
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Training runs are GPU-first but the code is device-agnostic (CUDA → MPS → CPU). Benchmark and dataset extras install with `pip install -e ".[data,train]"`.

## Neural Latents Benchmark

```bash
scripts/nlb.sh mc_maze 000128   # download + train; reports co-bps and velocity R²
```

The trainer uses bf16 autocast on CUDA and reports `co-bps` on held-out neurons and velocity R² on a validation split. Swap the dataset/dandiset for `mc_rtt 000129`, `area2_bump 000127`, or `dmfc_rsg 000130`. The `co-bps` implementation matches the official benchmark definition.

### Cross-subject pretraining

```bash
scripts/pretrain.sh                        # self-supervised across several datasets
python -m noema.train.run --dataset nlb --name mc_maze --path data/mc_maze \
  --init checkpoints/noema.pt --steps 5000 # fine-tune, warm-started backbone
```

Stage 1 places each dataset's neurons in a disjoint slice of one embedding table with a session label (driving the adversarial invariance term); stage 2 fine-tunes a single dataset with its behavior labels from the pretrained backbone.

## Evaluation & capabilities

A forward model buys more than offline accuracy:

- **Streaming / online decode** — `eval/streaming.py` decodes one spike bin at a time over a rolling window; `eval/falcon.py` wraps it in the FALCON `BCIDecoder` interface (verified against the challenge package).
- **Calibration curve** — `eval/calibration.py` plots decode accuracy against calibration budget, few-shot transfer versus training from scratch: the practical payoff of pretraining.
- **Sim-to-real** — `eval/sim2real.py` fits a fresh decoder purely on world-model-imagined data and scores it on real held-out recordings.
- **Baselines** — `eval/baselines.py` ridge velocity decoder, the reference to beat.

## Demo

`demo/noema.html` runs the trained world model live in the browser — drag to command a movement and watch the imagined population firing and decoded motion respond. The JS forward pass is verified against PyTorch (`demo/parity.mjs`).

## Status

Architecture, training, the benchmarks, and the capabilities above are implemented and tested on synthetic neural dynamics; the data and evaluation paths are verified against the real NLB (`nlb_tools`) and FALCON (`falcon-challenge`) packages. Competitive numbers on real recordings are the next step and need GPU training.
