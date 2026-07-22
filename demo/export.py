"""Train a small world model and export its weights for the browser demo.

Only the action-conditioned world model, readout, and behavior head are exported;
the seed latents are precomputed so the browser never runs the neural encoder.
A reference rollout is included so demo/parity.mjs can prove the JS forward pass
matches PyTorch.
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


def _lin(m):
    return {"W": m.weight.tolist(), "b": None if m.bias is None else m.bias.tolist()}


def _ln(m):
    return {"w": m.weight.tolist(), "b": m.bias.tolist()}


def _block(b):
    return {
        "n1": _ln(b.norm1), "n2": _ln(b.norm2),
        "qkv": b.attn.qkv.weight.tolist(), "proj": b.attn.proj.weight.tolist(),
        "mlp0": {"W": b.mlp[0].weight.tolist(), "b": b.mlp[0].bias.tolist()},
        "mlp2": {"W": b.mlp[2].weight.tolist(), "b": b.mlp[2].bias.tolist()},
    }


def held_actions(batch, steps, dim, g, hold=10):
    """Piecewise-constant action headings held for several bins each. Matches how a
    user drives the demo — sustained steering — so held commands stay in-distribution
    and the decoder recovers direction faithfully (per-step white noise does not)."""
    actions, current = [], torch.randn(batch, dim, generator=g)
    for t in range(steps):
        if t % hold == 0:
            current = torch.randn(batch, dim, generator=g)
        actions.append(current)
    return torch.stack(actions, dim=1)


def rollout_dataset(system, batch, steps, g):
    actions = held_actions(batch, steps, system.action_dim, g)
    _, rates, _ = system.rollout(actions)
    return torch.poisson(rates, generator=g), actions


def export():
    torch.manual_seed(0)
    dim, heads, wm_depth, units = 64, 4, 2, 32
    system = LinearSpikeSystem(units=units, latent=6, action_dim=2, seed=1)
    g = torch.Generator().manual_seed(1)
    counts, actions = rollout_dataset(system, batch=384, steps=50, g=g)
    unit_ids = torch.arange(units)
    # Decode the intended movement itself, so steering the action moves the cursor
    # that way — the population encodes the command and the decoder recovers it.
    ds = SpikeWindows(counts, behavior=actions, actions=actions)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=ds.collate, drop_last=True)

    model = Noema(dim=dim, enc_depth=2, wm_depth=wm_depth, heads=heads, max_units=units,
                  action_dim=2, behavior_dim=2)
    train(model, loader, TrainConfig(steps=2000, warmup=100, lr=3e-3, w_forecast=2.0, ckpt=""),
          device=torch.device("cpu"))
    model.eval()

    with torch.no_grad():
        counts, actions = rollout_dataset(system, batch=1, steps=50, g=g)
        seed = 15
        z_seed = model.encode(counts[:, :seed], unit_ids)
        future = actions[:, seed:]
        rates, beh = imagine(model, counts[:, :seed], unit_ids, future, seed_actions=actions[:, :seed])

        data = {
            "dim": dim, "heads": heads, "head_dim": dim // heads, "action_dim": 2,
            "behavior_dim": 2, "scale": dim ** -0.5, "base": 10_000.0,
            "world": {
                "action": _lin(model.world.action),
                "blocks": [_block(b) for b in model.world.core.blocks],
                "norm": _ln(model.world.core.norm),
                "head": _lin(model.world.head),
            },
            "readout": model.tokenizer.readout(unit_ids).tolist(),
            "bias": model.tokenizer.bias(unit_ids).squeeze(-1).tolist(),
            "behavior": {
                "n0": {"W": model.behavior.net[0].weight.tolist(), "b": model.behavior.net[0].bias.tolist()},
                "n2": {"W": model.behavior.net[2].weight.tolist(), "b": model.behavior.net[2].bias.tolist()},
            },
            "seed": {"z": z_seed[0].tolist(), "actions": actions[0, :seed].tolist()},
            "reference": {"actions": future[0].tolist(), "rates": rates[0].tolist(), "behavior": beh[0].tolist()},
        }
    (HERE / "model.json").write_text(json.dumps(data))
    return data


def build_html(data):
    # Inline the parity-tested forward pass (strip module exports) plus the data,
    # so the page is fully self-contained and openable without a server.
    core = (HERE / "model.mjs").read_text()
    core = core.replace("export function", "function")
    core = "\n".join(line for line in core.splitlines() if not line.startswith("export {"))
    html = TEMPLATE.replace("/*MODEL_CORE*/", core).replace("__DATA__", json.dumps(data))
    (HERE / "noema.html").write_text(html)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Noema — steer the neural world model</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0e14; color:#c9d3e0;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:900px; margin:0 auto; padding:32px 20px 48px; }
  h1 { font-size:20px; font-weight:600; margin:0 0 4px; }
  p.sub { color:#7c8698; margin:0 0 24px; max-width:640px; }
  .grid { display:grid; grid-template-columns:1fr; gap:24px; align-items:start; }
  @media (min-width:720px) { .grid { grid-template-columns:220px 1fr; } }
  #pad { margin:0 auto; }
  .label { font-size:12px; color:#7c8698; margin:0 0 6px; }
  canvas { display:block; width:100%; border-radius:6px; }
  #pad { width:200px; height:200px; border-radius:50%; background:#141a26;
         border:1px solid #2c3446; position:relative; touch-action:none; cursor:grab; }
  #knob { width:34px; height:34px; border-radius:50%; background:#4b9fff;
          position:absolute; left:83px; top:83px; box-shadow:0 0 18px #4b9fff88; }
  button { background:#1b2130; color:#c9d3e0; border:1px solid #2c3446; border-radius:6px;
           padding:6px 14px; cursor:pointer; margin-top:12px; font-size:13px; }
  .foot { color:#4d566a; font-size:11px; margin-top:24px; }
</style></head>
<body><div class="wrap">
  <h1>Noema — steer the neural world model</h1>
  <p class="sub">Drag the pad to set an intended movement. The world model imagines the population's firing in response and decodes it back into motion — running live in your browser, weights matched to the trained model.</p>
  <div class="grid">
    <div>
      <div class="label">intended movement</div>
      <div id="pad"><div id="knob"></div></div>
      <button id="reset">Reset</button>
    </div>
    <div>
      <div class="label">imagined population firing &nbsp;·&nbsp; neurons × time →</div>
      <canvas id="raster"></canvas>
      <div class="label" style="margin-top:18px">decoded movement</div>
      <canvas id="path"></canvas>
    </div>
  </div>
  <div class="foot">A compact model trained on synthetic action-driven dynamics; the browser runs its exact forward pass (verified against PyTorch to 1e-5).</div>
</div>
<script>
/*MODEL_CORE*/
const M = __DATA__;
const N = M.readout.length;
let z = M.seed.z.map(r => r.slice());
let a = M.seed.actions.map(r => r.slice());
let action = [0, 0], pos = [0, 0];
const CAP = 45, history = [], path = [];

function stepModel() {
  const next = worldStep(z, a, M);
  z.push(next); a.push(action.slice());
  if (z.length > CAP) { z.shift(); a.shift(); }
  const rates = decode(next, M), vel = behavior(next, M);
  history.push(rates); if (history.length > 90) history.shift();
  // Light friction keeps the decoded cursor on-screen and eases it back to
  // center when steering stops, instead of drifting away unbounded.
  pos = [pos[0] * 0.95 + vel[0] * 0.15, pos[1] * 0.95 + vel[1] * 0.15];
  path.push(pos.slice()); if (path.length > 260) path.shift();
}

const raster = document.getElementById('raster'), rx = raster.getContext('2d');
const pathC = document.getElementById('path'), px = pathC.getContext('2d');
function drawRaster() {
  const W = raster.clientWidth, H = 220, T = history.length;
  raster.width = W; raster.height = H;
  const cw = W / 90, ch = H / N;
  for (let x = 0; x < T; x++) for (let y = 0; y < N; y++) {
    // Fixed scale + gamma: robust to rare high-firing bins, so typical activity
    // stays visible instead of washing out against an outlier-driven maximum.
    const u = Math.min(1, (history[x][y] / 6) ** 0.6);
    rx.fillStyle = `rgb(${20 + u * 40 | 0},${30 + u * 150 | 0},${50 + u * 190 | 0})`;
    rx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
  }
}
function drawPath() {
  const W = pathC.clientWidth, H = 180; pathC.width = W; pathC.height = H;
  px.strokeStyle = '#4b9fff'; px.lineWidth = 1.5; px.beginPath();
  const cx = W / 2, cy = H / 2, s = 6;
  path.forEach((p, i) => { const X = cx + p[0] * s, Y = cy - p[1] * s; i ? px.lineTo(X, Y) : px.moveTo(X, Y); });
  px.stroke();
  if (path.length) { const p = path[path.length - 1];
    px.fillStyle = '#e4ebf5'; px.beginPath(); px.arc(cx + p[0] * s, cy - p[1] * s, 3.5, 0, 7); px.fill(); }
}

const pad = document.getElementById('pad'), knob = document.getElementById('knob');
let dragging = false;
function setAction(ev) {
  const r = pad.getBoundingClientRect();
  let dx = (ev.clientX - r.left - 100) / 100, dy = (ev.clientY - r.top - 100) / 100;
  const m = Math.hypot(dx, dy); if (m > 1) { dx /= m; dy /= m; }
  knob.style.left = (83 + dx * 83) + 'px'; knob.style.top = (83 + dy * 83) + 'px';
  action = [dx * 1.3, -dy * 1.3];   // screen-down is negative velocity; stay in-distribution
}
pad.addEventListener('pointerdown', e => { dragging = true; pad.setPointerCapture(e.pointerId); setAction(e); });
pad.addEventListener('pointermove', e => dragging && setAction(e));
pad.addEventListener('pointerup', () => { dragging = false; });
document.getElementById('reset').onclick = () => {
  z = M.seed.z.map(r => r.slice()); a = M.seed.actions.map(r => r.slice());
  pos = [0, 0]; path.length = 0; history.length = 0;
  action = [0, 0]; knob.style.left = '83px'; knob.style.top = '83px';
};

setInterval(() => { stepModel(); drawRaster(); drawPath(); }, 110);
window.onresize = () => { drawRaster(); drawPath(); };
</script></body></html>"""


if __name__ == "__main__":
    data = export()
    build_html(data)
    print("wrote demo/model.json and demo/noema.html")
