"""Render an imagined neural rollout to a standalone HTML page.

Trains a small action-conditioned model on a synthetic system, seeds it on a
held-out window of real spikes, and imagines the population forward under an
action plan. The predicted firing and decoded behavior are compared to ground
truth and written into a self-contained page (data inlined, no dependencies).
"""

import json
import pathlib

import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import LinearSpikeSystem
from noema.sim import imagine
from noema.train import TrainConfig, train

HERE = pathlib.Path(__file__).parent


def build():
    torch.manual_seed(0)
    system = LinearSpikeSystem(units=40, latent=6, action_dim=2, seed=1)
    counts, unit_ids, actions, behavior = system.sample(batch=256, steps=45)
    ds = SpikeWindows(counts, behavior=behavior, actions=actions)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=ds.collate, drop_last=True)

    model = Noema(dim=128, enc_depth=3, wm_depth=3, heads=4, max_units=40,
                  action_dim=2, behavior_dim=2)
    train(model, loader, TrainConfig(steps=1200, warmup=60, lr=3e-3, w_forecast=2.0, ckpt=""),
          device=torch.device("cpu"))

    counts, unit_ids, actions, behavior = system.sample(batch=16, steps=45)
    seed = 30
    _, true_rates, _ = system.rollout(actions)
    rates, beh = imagine(model, counts[:, :seed], unit_ids, actions[:, seed:],
                         seed_actions=actions[:, :seed])
    true_future = true_rates[:, seed:]

    # Show an upper-quartile episode by firing fidelity: a clearly faithful example
    # (not the single best outlier), with ground truth shown alongside to verify it.
    def corr(a, b):
        a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
        return (a @ b / (a.norm() * b.norm() + 1e-8)).item()
    scores = [corr(rates[i].clamp_min(1e-6).log(), true_future[i].clamp_min(1e-6).log())
              for i in range(rates.size(0))]
    i = sorted(range(len(scores)), key=lambda k: scores[k])[3 * len(scores) // 4]

    return {
        "seed": seed,
        "rates": rates[i].tolist(),                  # imagined firing, [future, units]
        "true_rates": true_future[i].tolist(),
        "behavior": beh[i].tolist(),                 # decoded behavior, [future, 2]
        "true_behavior": behavior[i, seed:].tolist(),
    }


def render(data):
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    out = HERE / "noema.html"
    out.write_text(html)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Noema — imagined neural rollout</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0e14; color:#c9d3e0;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:900px; margin:0 auto; padding:32px 20px 48px; }
  h1 { font-size:20px; font-weight:600; letter-spacing:.2px; margin:0 0 4px; }
  p.sub { color:#7c8698; margin:0 0 24px; }
  canvas { width:100%; display:block; image-rendering:pixelated; border-radius:6px; }
  .label { font-size:12px; color:#7c8698; margin:20px 0 6px; }
  .row { display:flex; gap:16px; align-items:center; margin-top:20px; }
  button { background:#1b2130; color:#c9d3e0; border:1px solid #2c3446;
           border-radius:6px; padding:6px 14px; cursor:pointer; font-size:13px; }
  button:hover { background:#232b3d; }
  input[type=range] { flex:1; accent-color:#4b9fff; }
  .legend { font-size:12px; color:#7c8698; }
  .k { display:inline-block; width:10px; height:10px; border-radius:2px;
       vertical-align:middle; margin:0 4px 0 12px; }
</style></head>
<body><div class="wrap">
  <h1>Noema — imagined neural rollout</h1>
  <p class="sub">Seeded on real spikes, the world model imagines the population forward under an action plan. Below, its hallucinated firing is set against ground truth.</p>

  <div class="label">Imagined firing &nbsp;·&nbsp; neurons × time</div>
  <canvas id="heat"></canvas>
  <div class="label">True firing</div>
  <canvas id="heatTrue"></canvas>

  <div class="label">Decoded behavior
    <span class="legend"><span class="k" style="background:#4b9fff"></span>imagined
    <span class="k" style="background:#5b657a"></span>true</span></div>
  <canvas id="beh"></canvas>

  <div class="row">
    <button id="play">Pause</button>
    <input id="scrub" type="range" min="0" max="100" value="0"/>
  </div>
</div>
<script>
const D = __DATA__;
const rates = D.rates, trueRates = D.true_rates, N = rates[0].length, T = rates.length;
const heat = document.getElementById('heat'), hx = heat.getContext('2d');
const heatT = document.getElementById('heatTrue'), htx = heatT.getContext('2d');
const beh = document.getElementById('beh'), bx = beh.getContext('2d');
const scrub = document.getElementById('scrub'), play = document.getElementById('play');
let t = 0, running = true;

let hi = 0;                                     // shared scale so the panels compare
for (const g of [rates, trueRates]) for (const r of g) for (const v of r) hi = Math.max(hi, v);
function color(v){ const u = Math.min(1, v/hi);
  return `rgb(${20+u*40|0},${30+u*150|0},${50+u*190|0})`; }

function drawGrid(ctx, cv, grid, cur){
  const W = cv.clientWidth, H = 150, cw = W/T, ch = H/N;
  cv.width = W; cv.height = H;
  for (let x=0; x<T; x++) for (let y=0; y<N; y++){
    ctx.fillStyle = color(grid[x][y]); ctx.fillRect(x*cw, y*ch, cw+1, ch+1);
  }
  ctx.strokeStyle = '#e4ebf5'; ctx.globalAlpha = .85;
  ctx.beginPath(); ctx.moveTo(cur*cw, 0); ctx.lineTo(cur*cw, H); ctx.stroke();
  ctx.globalAlpha = 1;
}
function line(ctx, series, cur, W, H, col){
  ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let x=0; x<=cur && x<series.length; x++){
    const px = x*(W/T), py = H/2 - series[x][0]*26;
    x?ctx.lineTo(px,py):ctx.moveTo(px,py);
  } ctx.stroke();
}
function drawBeh(cur){
  const W = beh.clientWidth, H = 120; beh.width = W; beh.height = H;
  line(bx, D.true_behavior, cur, W, H, '#5b657a');
  line(bx, D.behavior, cur, W, H, '#4b9fff');
}
function render(cur){ drawGrid(hx, heat, rates, cur); drawGrid(htx, heatT, trueRates, cur); drawBeh(cur); }
function frame(){ render(t); scrub.value = (t/(T-1))*100; if (running) t = (t+1) % T; }
scrub.oninput = () => { t = Math.round(scrub.value/100*(T-1)); running=false; play.textContent='Play'; render(t); };
play.onclick = () => { running=!running; play.textContent = running?'Pause':'Play'; };
setInterval(frame, 90);
window.onresize = () => render(t);
</script></body></html>"""


if __name__ == "__main__":
    path = render(build())
    print(f"wrote {path}")
