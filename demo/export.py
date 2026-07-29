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


def _faces():
    """The three subset faces, base64-inlined so the page needs no network at all.

    Built by scripts/build_fonts.sh; see demo/fonts/OFL.md for why one face cannot
    cover all three roles. font-display is block rather than swap: a data URI has no
    round trip, so there is nothing to swap in, and swap can still flash for a frame.
    """
    import base64

    rules = []
    for name, file, weight in (("Noema Display", "noema-display.woff2", "400 700"),
                               ("Noema Text", "noema-text.woff2", "400"),
                               ("Noema Data", "noema-data.woff2", "400")):
        blob = base64.b64encode((HERE / "fonts" / file).read_bytes()).decode()
        rules.append(f'@font-face{{font-family:"{name}";'
                     f'src:url(data:font/woff2;base64,{blob}) format("woff2");'
                     f'font-weight:{weight};font-style:normal;font-display:block}}')
    return "\n".join(rules)


def build_html(data):
    # Inline the parity-tested forward pass (strip module exports) plus the data,
    # so the page is fully self-contained and openable without a server.
    core = (HERE / "model.mjs").read_text()
    core = core.replace("export function", "function")
    core = "\n".join(line for line in core.splitlines() if not line.startswith("export {"))
    html = (TEMPLATE
            .replace("/*FONT_FACES*/", _faces())
            .replace("/*MODEL_CORE*/", core)
            .replace("__DATA__", json.dumps(data)))
    (HERE / "noema.html").write_text(html)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="theme-color" content="#FAF7F0"/>
<title>Noema — steering a neural world model</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='6' fill='%2317161A'/%3E%3C/svg%3E"/>
<style>
/*FONT_FACES*/
:root{
  color-scheme:light;
  --paper:#FAF7F0; --leaf:#F3EFE5;
  --rule:#E2DCCD; --rule-hi:#CFC7B4;
  --ink-1:#17161A; --ink-2:#3F4149; --ink-3:#6E6A62; --ink-4:#918C82;
  --intent:#B03A1E; --decoded:#1D243E;
  /* Stop 0 is bit-identical to --paper on purpose: unfilled cells show the page
     through, so any drift would draw a hard rectangle edge around the field. */
  --ramp-0:#FAF7F0;  --ramp-1:#EDEFEF;  --ramp-2:#E1E7EE;  --ramp-3:#D4DCE6;
  --ramp-4:#C8D1DE;  --ramp-5:#BBC5D4;  --ramp-6:#AFB9CA;  --ramp-7:#A1ACBF;
  --ramp-8:#939FB5;  --ramp-9:#8490A9;  --ramp-10:#76829D; --ramp-11:#66728F;
  --ramp-12:#576381; --ramp-13:#485372; --ramp-14:#394363; --ramp-15:#2B3350;
  --ramp-16:#1D243E;
  --serif:"Noema Text",Charter,"Palatino Linotype",Palatino,Georgia,serif;
  --display:"Noema Display",var(--serif);
  --mono:"Noema Data",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
/* One weight per family ships, so every synthesis request is a defect rather than
   a fallback. Stating it here makes the failure visible in review. */
html{font-synthesis:none}
h1,h2,h3,th,b,strong,button{font-weight:400}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink-2);
     font:400 17px/1.62 var(--serif);font-feature-settings:"onum" 1}
.page{max-width:960px;margin:0 auto;padding:28px 24px 96px}
.rule{border:0;border-top:1px solid var(--rule);margin:0}
.head{display:flex;justify-content:space-between;align-items:baseline;
      padding-bottom:10px;border-bottom:1px solid var(--ink-1);
      font:400 12px/1 var(--mono);letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3)}
.head b{color:var(--ink-1)}
h1{font:400 44px/1.08 var(--display);letter-spacing:-.012em;color:var(--ink-1);margin:40px 0 14px}
.deck{max-width:34em;margin:0 0 12px;color:var(--ink-2)}
.deck em{font-style:normal;color:var(--ink-1)}
/* Numbers are compared here, so state the figure set rather than inheriting it.
   Written as one declaration: splitting the two keywords loses the first. */
.data,.readout,.gauge b,figcaption b{font-family:var(--mono);
  font-variant-numeric:tabular-nums slashed-zero;font-feature-settings:normal}
figure{margin:36px 0 0}
.plate{display:grid;grid-template-columns:212px 1fr;gap:0 34px;align-items:start}
.part{min-width:0}
.part-label{display:block;font:400 12px/1.4 var(--mono);letter-spacing:.09em;
            text-transform:uppercase;color:var(--ink-3);margin:0 0 10px}
.part-label i{font-style:normal;color:var(--ink-1)}
/* (a) intent */
#pad{width:212px;height:212px;background:var(--leaf);border:1px solid var(--rule-hi);
     position:relative;touch-action:none;cursor:crosshair}
#pad:focus-visible{outline:2px solid var(--intent);outline-offset:2px}
#pad .cross{position:absolute;background:var(--rule-hi)}
#pad .cross.h{left:0;right:0;top:50%;height:1px}
#pad .cross.v{top:0;bottom:0;left:50%;width:1px}
#knob{width:14px;height:14px;background:var(--intent);position:absolute;
      left:99px;top:99px}
#vec{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
.readout{margin:12px 0 0;font-size:13px;color:var(--ink-4);letter-spacing:.02em}
.readout.live{color:var(--ink-1)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
button{font:400 12px/1 var(--mono);letter-spacing:.09em;text-transform:uppercase;
       background:var(--paper);color:var(--ink-1);border:1px solid var(--ink-1);
       border-radius:0;padding:9px 14px;margin-top:14px;cursor:pointer}
button:hover{background:var(--ink-1);color:var(--paper)}
button:focus-visible{outline:2px solid var(--intent);outline-offset:2px}
/* (b) the plate. Only two rules, so the field reads as a printed figure. */
.raster-wrap{border-left:1px solid var(--rule-hi);border-bottom:1px solid var(--rule-hi);
             padding:20px 0 20px 10px;position:relative}
#raster{display:block;width:100%;height:288px;background:var(--paper)}
.nowlbl{position:absolute;top:0;right:0;font:400 12px/1 var(--mono);
        letter-spacing:.09em;color:var(--ink-1)}
.axis{display:flex;justify-content:space-between;margin:8px 0 0 10px;
      font:400 12px/1.4 var(--mono);letter-spacing:.09em;color:var(--ink-3)}
.legend{display:flex;align-items:center;gap:8px;margin:16px 0 0 10px;
        font:400 12px/1 var(--mono);letter-spacing:.09em;color:var(--ink-3)}
.legend i{display:block;width:132px;height:8px;outline:1px solid var(--rule-hi);
          background:linear-gradient(90deg,#FAF7F0,#EDEFEF,#E1E7EE,#D4DCE6,#C8D1DE,#BBC5D4,
          #AFB9CA,#A1ACBF,#939FB5,#8490A9,#76829D,#66728F,#576381,#485372,#394363,#2B3350,#1D243E)}
/* (c) decoded */
#path{display:block;width:100%;height:212px;background:var(--paper);
      border-left:1px solid var(--rule-hi);border-bottom:1px solid var(--rule-hi)}
.gauge{margin:10px 0 0;font:400 12px/1.5 var(--mono);letter-spacing:.09em;color:var(--ink-3)}
.gauge b{font-weight:400;color:var(--ink-1)}
figcaption{margin:26px 0 0;max-width:44em;font-size:15px;color:var(--ink-2)}
figcaption b{font-weight:400;color:var(--ink-1)}
.notes{margin:34px 0 0;padding:0;list-style:none;max-width:44em;
       font-size:14px;color:var(--ink-3)}
.notes li{margin:0 0 7px;padding-left:1.5em;text-indent:-1.5em}
.notes sup{font-feature-settings:"sups" 1;margin-right:.45em;color:var(--ink-2)}
.colophon{margin:52px 0 0;padding-top:12px;border-top:1px solid var(--rule);
          display:flex;gap:26px;flex-wrap:wrap;
          font:400 12px/1.6 var(--mono);letter-spacing:.09em;color:var(--ink-4)}
.colophon span b{font-weight:400;color:var(--ink-3)}
@media (max-width:820px){
  .plate{grid-template-columns:1fr;gap:30px}
  h1{font-size:34px}
  #pad{width:100%;max-width:260px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head>
<body>
<div class="page">
  <header class="head"><span><b>Noema</b></span><span>Interactive figure</span></header>

  <h1>Steering a neural world model</h1>
  <p class="deck">Set an intended movement. The model predicts what a neural population
  would do in response, then decodes that prediction back into motion — the forward model
  running in the open, one step at a time.</p>

  <figure>
    <div class="plate">
      <div class="part">
        <span class="part-label"><i>(a)</i> Intent</span>
        <div id="pad" role="application" tabindex="0"
             aria-label="Movement control. Use the arrow keys or W A S D to steer, Escape to stop.">
          <div class="cross h"></div><div class="cross v"></div>
          <canvas id="vec" width="212" height="212"></canvas>
          <div id="knob"></div>
        </div>
        <p class="sr">Press the arrow keys or W A S D to steer. Press Escape to stop.</p>
        <p class="readout data" id="readout">θ&nbsp;—&nbsp;·&nbsp;‖v‖&nbsp;0.00</p>
        <button id="reset" type="button">Reset</button>
      </div>

      <div class="part">
        <span class="part-label"><i>(b)</i> Predicted population — synthetic</span>
        <div class="raster-wrap">
          <span class="nowlbl">now</span>
          <canvas id="raster" role="img"
                  aria-label="Predicted firing rates for 32 units over the last 1.8 seconds."></canvas>
        </div>
        <div class="axis"><span>0 → 1.8 s · 90 bins × 20 ms</span><span>unit 1…32</span></div>
        <div class="legend"><span>low</span><i></i><span>high</span><span>predicted rate</span></div>

        <span class="part-label" style="margin-top:34px"><i>(c)</i> Decoded movement</span>
        <canvas id="path" role="img"
                aria-label="Path reconstructed from the predicted population firing."></canvas>
        <p class="gauge">alignment, intent against decoded <b id="align">—</b></p>
      </div>
    </div>

    <figcaption><b>Figure 1.</b> A compact action-conditioned world model, trained on
    synthetic dynamics, rolled forward in the browser. Panel (b) shows predicted rates,
    not sampled spikes. This figure demonstrates the mechanism; it is not an experimental
    recording, and it establishes nothing about the benchmark results.<sup>1</sup></figcaption>
  </figure>

  <ol class="notes">
    <li><sup>1</sup>The interactive model is trained on a synthetic linear spiking system,
    so its numbers describe that system and no brain.</li>
    <li><sup>2</sup>The browser runs the same forward pass as the Python model: measured
    max |Δ| 3 × 10⁻⁵ against a 2 × 10⁻³ gate.</li>
    <li><sup>3</sup>Playback runs at roughly one fifth of model time.</li>
  </ol>

  <footer class="colophon">
    <span><b>Model</b> action-conditioned latent world model</span>
    <span><b>Display</b> 32 units · 90 bins</span>
    <span><b>Parity</b> max |Δ| 3 × 10⁻⁵</span>
  </footer>
</div>
<script>
/*MODEL_CORE*/
const M = __DATA__;
const N = M.readout.length;
let z = M.seed.z.map(r => r.slice());
let a = M.seed.actions.map(r => r.slice());
let action = [0, 0], pos = [0, 0];
const CAP = 45, HISTORY = 90, history = [], path = [];
let aligns = [];

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// Build the ramp from the stylesheet's own stops, so the field and the legend
// cannot drift apart when either is edited.
let LUT = [];
function buildLUT() {
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const stops = Array.from({ length: 17 }, (_, i) => hex(css('--ramp-' + i)));
  LUT = Array.from({ length: 256 }, (_, i) => {
    const t = i / 255 * 16, k = Math.min(15, Math.floor(t)), f = t - k;
    const c = stops[k].map((v, j) => Math.round(v + (stops[k + 1][j] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  });
}

function stepModel() {
  const next = worldStep(z, a, M);
  z.push(next); a.push(action.slice());
  if (z.length > CAP) { z.shift(); a.shift(); }
  const rates = decode(next, M), vel = behavior(next, M);
  history.push(rates); if (history.length > HISTORY) history.shift();
  // Light friction keeps the decoded cursor on-screen and eases it back to
  // center when steering stops, instead of drifting away unbounded.
  pos = [pos[0] * 0.95 + vel[0] * 0.15, pos[1] * 0.95 + vel[1] * 0.15];
  path.push(pos.slice()); if (path.length > 260) path.shift();
  const na = Math.hypot(action[0], action[1]), nv = Math.hypot(vel[0], vel[1]);
  if (na > 0.05 && nv > 1e-6) {
    aligns.push((action[0] * vel[0] + action[1] * vel[1]) / (na * nv));
    if (aligns.length > 40) aligns.shift();
  }
}

const raster = document.getElementById('raster'), rx = raster.getContext('2d');
const pathC = document.getElementById('path'), px = pathC.getContext('2d');
const vecC = document.getElementById('vec'), vx = vecC.getContext('2d');

function fit(c) {
  const d = window.devicePixelRatio || 1;
  c.width = Math.round(c.clientWidth * d); c.height = Math.round(c.clientHeight * d);
  c.getContext('2d').setTransform(d, 0, 0, d, 0, 0);
  return [c.clientWidth, c.clientHeight];
}

function drawRaster() {
  const [W, H] = fit(raster);
  rx.clearRect(0, 0, W, H);
  const cw = W / HISTORY, pitch = H / N;
  for (let x = 0; x < history.length; x++) {
    // Right-align: the newest column sits at the "now" edge and older activity
    // walks left, so the axis label and the data agree while the buffer fills.
    const col = HISTORY - history.length + x;
    for (let y = 0; y < N; y++) {
      // A log with a soft floor, not a gamma. Rates are sparse and long-tailed;
      // a gamma fitted to a dark background crushes the quiet baseline into the page.
      const u = Math.log1p(Math.min(history[x][y], 24) / 2.0) / 2.5649;
      rx.fillStyle = LUT[Math.max(0, Math.min(255, Math.round(u * 255)))];
      rx.fillRect(col * cw, y * pitch, Math.ceil(cw), Math.max(1, pitch - 1));
    }
  }
}

function drawPath() {
  const [W, H] = fit(pathC);
  px.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, s = 6;
  px.strokeStyle = css('--rule-hi'); px.lineWidth = 1;
  px.beginPath(); px.moveTo(0, cy); px.lineTo(W, cy);
  px.moveTo(cx, 0); px.lineTo(cx, H); px.stroke();
  // A ghost of the current intent, so the reader can see what the decode is chasing.
  if (Math.hypot(action[0], action[1]) > 0.05) {
    px.strokeStyle = css('--intent'); px.lineWidth = 1; px.setLineDash([3, 3]);
    px.beginPath(); px.moveTo(cx, cy);
    px.lineTo(cx + action[0] * 26, cy - action[1] * 26); px.stroke();
    px.setLineDash([]);
  }
  px.strokeStyle = css('--decoded'); px.lineWidth = 1.5;
  px.beginPath();
  path.forEach((p, i) => {
    const X = cx + p[0] * s, Y = cy - p[1] * s;
    i ? px.lineTo(X, Y) : px.moveTo(X, Y);
  });
  px.stroke();
  if (path.length) {
    const p = path[path.length - 1];
    px.fillStyle = css('--decoded');
    px.fillRect(cx + p[0] * s - 3, cy - p[1] * s - 3, 6, 6);
  }
}

function drawVector() {
  const [W, H] = fit(vecC);
  vx.clearRect(0, 0, W, H);
  if (Math.hypot(action[0], action[1]) < 0.05) return;
  vx.strokeStyle = css('--intent'); vx.lineWidth = 1;
  vx.beginPath(); vx.moveTo(W / 2, H / 2);
  vx.lineTo(W / 2 + action[0] / 1.3 * 99, H / 2 - action[1] / 1.3 * 99);
  vx.stroke();
}

const readout = document.getElementById('readout'), alignEl = document.getElementById('align');
function drawText() {
  const n = Math.hypot(action[0], action[1]);
  const live = n > 0.05;
  readout.classList.toggle('live', live);
  const deg = live ? String(Math.round((Math.atan2(action[1], action[0]) * 180 / Math.PI + 360) % 360)).padStart(3, '0') + '°' : '—';
  readout.textContent = `θ ${deg} · ‖v‖ ${(n / 1.3).toFixed(2)}`;
  alignEl.textContent = aligns.length >= 8
    ? (aligns.reduce((s, v) => s + v, 0) / aligns.length).toFixed(2) : '—';
}

const pad = document.getElementById('pad'), knob = document.getElementById('knob');
let dragging = false;
function place(dx, dy) {
  const m = Math.hypot(dx, dy); if (m > 1) { dx /= m; dy /= m; }
  knob.style.left = (99 + dx * 99) + 'px'; knob.style.top = (99 + dy * 99) + 'px';
  action = [dx * 1.3, -dy * 1.3];   // screen-down is negative velocity; stay in-distribution
}
function setAction(ev) {
  const r = pad.getBoundingClientRect();
  place((ev.clientX - r.left - r.width / 2) / (r.width / 2),
        (ev.clientY - r.top - r.height / 2) / (r.height / 2));
}
pad.addEventListener('pointerdown', e => { dragging = true; pad.setPointerCapture(e.pointerId); setAction(e); });
pad.addEventListener('pointermove', e => dragging && setAction(e));
pad.addEventListener('pointerup', () => { dragging = false; });
pad.addEventListener('keydown', e => {
  const k = { ArrowLeft: [-1, 0], a: [-1, 0], ArrowRight: [1, 0], d: [1, 0],
              ArrowUp: [0, -1], w: [0, -1], ArrowDown: [0, 1], s: [0, 1] }[e.key];
  if (k) { place(k[0], k[1]); e.preventDefault(); }
  else if (e.key === 'Escape') { place(0, 0); e.preventDefault(); }
});
document.getElementById('reset').onclick = () => {
  z = M.seed.z.map(r => r.slice()); a = M.seed.actions.map(r => r.slice());
  pos = [0, 0]; path.length = 0; history.length = 0; aligns = [];
  place(0, 0);
};

function frame() { stepModel(); drawRaster(); drawPath(); drawVector(); drawText(); }
// Pause when the tab is hidden. The standalone page would otherwise keep stepping
// the model forever in a background tab.
let timer = null;
const run = () => { if (!timer) timer = setInterval(frame, 110); };
const stop = () => { clearInterval(timer); timer = null; };
document.addEventListener('visibilitychange', () => document.hidden ? stop() : run());
window.addEventListener('resize', () => { drawRaster(); drawPath(); drawVector(); });

buildLUT();
// Canvas text and metrics do not participate in font loading, so wait for the
// faces before the first paint rather than letting the first frame rasterise wrong.
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { frame(); run(); });
</script></body></html>"""


if __name__ == "__main__":
    data = export()
    build_html(data)
    print("wrote demo/model.json and demo/noema.html")
