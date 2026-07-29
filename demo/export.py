"""Assemble the self-contained demo page.

    python demo/build.py     # measure and pack the assets -> demo/assets.json
    python demo/export.py    # assemble the page           -> demo/noema.html

The page holds thirteen real recordings and two models trained on other sessions of the
same days. Nothing here is synthetic and nothing is fetched: it opens from disk with no
network at all.
"""

import base64
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def faces():
    """The three subset faces, inlined. demo/fonts/OFL.md explains why one will not do."""
    rules = []
    for name, file, weight in (("Noema Display", "noema-display.woff2", "400 700"),
                               ("Noema Text", "noema-text.woff2", "400"),
                               ("Noema Data", "noema-data.woff2", "400")):
        blob = base64.b64encode((HERE / "fonts" / file).read_bytes()).decode()
        rules.append(f'@font-face{{font-family:"{name}";'
                     f'src:url(data:font/woff2;base64,{blob}) format("woff2");'
                     f'font-weight:{weight};font-style:normal;font-display:block}}')
    return "\n".join(rules)


def build_html():
    assets = json.loads((HERE / "assets.json").read_text())
    core = (HERE / "model.mjs").read_text()
    core = "\n".join(line.replace("export function", "function").replace("export const", "const")
                     for line in core.splitlines() if not line.startswith("export {"))
    html = (TEMPLATE
            .replace("/*FONT_FACES*/", faces())
            .replace("/*MODEL_CORE*/", core)
            .replace("__ASSETS__", json.dumps(assets)))
    (HERE / "noema.html").write_text(html)
    print(f"wrote demo/noema.html: {len(html.encode())/1e6:.2f} MB "
          f"({len(assets['tapes'])} recordings, {len(assets['weights'])} models)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="theme-color" content="#FAF7F0"/>
<meta name="description" content="A world model forecasting real human motor cortex, checked against the recording."/>
<title>Noema — two hundred milliseconds ahead of a human motor cortex</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='6' fill='%2317161A'/%3E%3C/svg%3E"/>
<style>
/*FONT_FACES*/
:root{
  color-scheme:light;
  --paper:#FAF7F0; --leaf:#F3EFE5; --rule:#E2DCCD; --rule-hi:#CFC7B4;
  --ink-1:#17161A; --ink-2:#3F4149; --ink-3:#6E6A62; --ink-4:#918C82;
  --intent:#B03A1E; --decoded:#1D243E;
  --ramp-0:#FAF7F0;--ramp-1:#E1E7EE;--ramp-2:#C8D1DE;--ramp-3:#AFB9CA;--ramp-4:#939FB5;
  --ramp-5:#76829D;--ramp-6:#576381;--ramp-7:#394363;--ramp-8:#1D243E;
  --serif:"Noema Text",Charter,"Palatino Linotype",Georgia,serif;
  --display:"Noema Display",var(--serif);
  --mono:"Noema Data",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
html{font-synthesis:none}
h1,h2,h3,b,strong,button{font-weight:400}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink-2);font:400 17px/1.62 var(--serif)}
.skip{position:absolute;left:-9999px}
.skip:focus{left:12px;top:12px;z-index:9;padding:10px 14px;background:var(--paper);border:1px solid var(--ink-1)}
.shell{display:grid;grid-template-columns:104px minmax(0,1fr);gap:0 34px;
       max-width:1180px;margin:0 auto;padding:26px 24px 96px}
.mono,.spine,.axis,.legend,.gates,.hint{font-family:var(--mono);font-size:12px;
       letter-spacing:.09em;font-variant-numeric:tabular-nums slashed-zero}

.spine{position:sticky;top:26px;align-self:start;border-right:1px solid var(--rule);padding-right:12px}
.spine h2{font:inherit;text-transform:uppercase;color:var(--ink-3);margin:0 0 14px;font-weight:400}
.day{display:block;width:100%;text-align:left;background:none;border:0;padding:3px 0;margin:0;
     cursor:pointer;color:var(--ink-3);font:inherit;line-height:1.1}
.day span{display:inline-block;width:22px}
.day i{display:inline-block;height:7px;background:var(--ink-2);vertical-align:middle;margin-left:3px;min-width:1px}
.day[aria-current="true"]{color:var(--ink-1)}
.day[aria-current="true"] i{background:var(--intent)}
.day:focus-visible{outline:2px solid var(--intent);outline-offset:2px}
.cutoff{border-top:1px solid var(--rule-hi);margin:6px 0;padding-top:5px;color:var(--ink-4);font-size:12px}

header{margin:0 0 30px}
h1{font:400 40px/1.1 var(--display);letter-spacing:-.012em;color:var(--ink-1);margin:0 0 12px;max-width:14em}
.deck{max-width:36em;margin:0}
.tapelab{display:flex;justify-content:space-between;gap:20px;color:var(--ink-3);margin:0 0 8px}
.tapelab b{font-weight:400;color:var(--ink-1)}

.field{position:relative;border-left:1px solid var(--rule-hi);border-bottom:1px solid var(--rule-hi)}
.reg{display:block;width:100%;height:150px;background:var(--paper)}
.reg.tall{height:190px}
/* The forecast is ten bins against hundreds of history, so it gets its own panel at
   full width. Drawing the context larger than the result inverts the argument. */
.horizon{display:grid;grid-template-columns:repeat(3,1fr);gap:0 18px;margin:18px 0 0}
.horizon > div{border-left:1px solid var(--rule-hi);border-bottom:1px solid var(--rule-hi)}
.hlab{display:block;font-family:var(--mono);font-size:12px;letter-spacing:.09em;
      color:var(--ink-3);padding:0 0 6px 8px}
.rowlab{position:absolute;right:7px;color:var(--ink-3);font-family:var(--mono);font-size:12px;
        letter-spacing:.09em;pointer-events:none;background:var(--paper);padding:0 3px}
#cut{position:absolute;top:0;bottom:0;width:1px;background:var(--intent);cursor:ew-resize;touch-action:none}
#cut::after{content:"";position:absolute;top:-7px;left:-6px;width:13px;height:13px;background:var(--intent)}
#cut:focus-visible{outline:2px solid var(--intent);outline-offset:3px}
.axis{display:flex;justify-content:space-between;margin:8px 0 0;color:var(--ink-3)}
.legend{display:flex;align-items:center;gap:8px;margin:14px 0 0;color:var(--ink-3)}
.legend i{width:120px;height:8px;outline:1px solid var(--rule-hi);background:linear-gradient(90deg,
  #FAF7F0,#E1E7EE,#C8D1DE,#AFB9CA,#939FB5,#76829D,#576381,#394363,#1D243E)}

section{margin:60px 0 0}
.mark{display:flex;gap:12px;align-items:baseline;border-top:1px solid var(--ink-1);padding-top:12px;
      margin:0 0 18px;font-family:var(--mono);font-size:12px;letter-spacing:.09em;
      text-transform:uppercase;color:var(--ink-1)}
.mark span{color:var(--ink-3)}
.lede{max-width:40em;margin:0 0 20px}
#plot{display:block;width:100%;height:250px}
.toggle{background:none;border:0;padding:0 0 2px;margin:0 18px 0 0;cursor:pointer;
        font-family:var(--mono);font-size:12px;letter-spacing:.09em;color:var(--ink-3);
        border-bottom:1px solid transparent}
.toggle[aria-pressed="true"]{color:var(--ink-1);border-bottom-color:var(--intent)}
.toggle:focus-visible{outline:2px solid var(--intent);outline-offset:2px}
table{border-collapse:collapse;width:100%;max-width:48em;margin:0}
th,td{text-align:left;padding:10px 0;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-family:var(--mono);font-size:12px;letter-spacing:.09em;text-transform:uppercase;
   color:var(--ink-3);border-bottom:1px solid var(--rule-hi);font-weight:400}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums slashed-zero;color:var(--ink-1);
     white-space:nowrap;padding-right:20px}
.gates,.hint{color:var(--ink-3);margin:12px 0 0;max-width:44em}
.hint{color:var(--ink-4)}
.prov{margin:24px 0 0;padding:12px 14px;background:var(--leaf);font-family:var(--mono);
      font-size:12px;letter-spacing:.09em;color:var(--ink-3)}
footer{margin:60px 0 0;padding-top:12px;border-top:1px solid var(--rule);font-family:var(--mono);
       font-size:12px;letter-spacing:.09em;color:var(--ink-4);display:flex;
       justify-content:space-between;flex-wrap:wrap;gap:14px}
footer a{color:var(--ink-3)}
@media (max-width:860px){
  .shell{grid-template-columns:1fr;gap:24px}
  .spine{position:static;border-right:0;border-bottom:1px solid var(--rule);padding:0 0 12px;
         display:flex;flex-wrap:wrap;gap:0 18px}
  .spine h2{width:100%}
  .cutoff{border-top:0;border-left:1px solid var(--rule-hi);padding:0 0 0 8px;margin:0}
  h1{font-size:30px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<a class="skip" href="#field">Skip to the figure</a>
<div class="shell">
  <nav class="spine" id="spine" aria-label="Recording day"><h2>Recording day</h2></nav>
  <main>
    <header>
      <h1>Two hundred milliseconds ahead of a human motor cortex.</h1>
      <p class="deck">Move the mark. The model commits a forecast for all 176 channels
      before the recording arrives to check it. Below the forecast is what the cortex
      actually did, and the difference between them.</p>
    </header>

    <p class="tapelab"><span>Day <b id="d-day">0</b> · <b id="d-split">held-in</b>
      · session <b id="d-ses"></b></span><span id="d-state">loading</span></p>

    <div class="field" id="field">
      <span class="rowlab" style="top:6px">recording so far</span>
      <canvas class="reg tall" id="r-hist" role="img" aria-label="The recording up to the mark"></canvas>
      <div id="cut" role="slider" tabindex="0" aria-label="Position in the recording"
           aria-valuemin="0" aria-valuemax="100" aria-valuenow="70"></div>
    </div>
    <div class="horizon">
      <div><span class="hlab">forecast</span>
        <canvas class="reg" id="r-pred" role="img" aria-label="Forecast firing rates for 176 channels"></canvas></div>
      <div><span class="hlab">what the cortex did</span>
        <canvas class="reg" id="r-true" role="img" aria-label="The firing that was actually recorded"></canvas></div>
      <div><span class="hlab">difference</span>
        <canvas class="reg" id="r-err" role="img" aria-label="Absolute difference between forecast and recording"></canvas></div>
    </div>
    <div class="axis"><span id="a-left">0 s</span><span>176 channels · 20 ms bins</span><span id="a-right"></span></div>
    <div class="legend"><span>low</span><i></i><span>high</span><span>rate, log scale</span></div>
    <p class="hint">Drag the mark, or focus it and use the arrow keys.</p>

    <section>
      <div class="mark"><span>1</span> Skill against horizon</div>
      <p class="lede">Correlation between forecast and recording across all 176 channels,
      at each step ahead. The centred line is the one that counts: raw correlation is
      dominated by how much each channel fires on average, so a forecast reproducing only
      the average firing profile still scores about 0.5. The dotted line is exactly that
      forecast.</p>
      <p><button class="toggle" id="t-multistep" aria-pressed="true">rollout objective</button
        ><button class="toggle" id="t-onestep" aria-pressed="false">one-step objective</button></p>
      <canvas id="plot" role="img" aria-label="Forecast skill against horizon"></canvas>
      <p class="gates" id="gates"></p>
    </section>

    <section>
      <div class="mark"><span>2</span> What this establishes</div>
      <table><thead><tr><th>Result</th><th>Value</th><th>Boundary</th></tr></thead>
      <tbody id="ledger"></tbody></table>
      <p class="prov" id="prov"></p>
    </section>

    <footer>
      <span>Noema · world models for neural dynamics</span>
      <span>Data <a href="https://dandiarchive.org/dandiset/000954">DANDI:000954</a>
        · Source <a href="https://github.com/manrajmondair/noema">github.com/manrajmondair/noema</a></span>
    </footer>
  </main>
</div>
<script type="module">
/*MODEL_CORE*/
const A = __ASSETS__;
const C = A.config, CH = C.channels, H = C.horizon, W = C.window, HIST = 240;

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

async function ungzip(b64){
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
  const buf = await new Response(new Blob([u]).stream()
    .pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  return new Uint8Array(buf);
}
function decodeF16(b){
  const u16 = new Uint16Array(b.buffer, b.byteOffset, b.byteLength/2);
  const out = new Float32Array(u16.length);
  for (let i=0;i<u16.length;i++){
    const h=u16[i], s=(h&0x8000)?-1:1, e=(h>>10)&0x1f, f=h&0x3ff;
    out[i] = e===0 ? s*2**-14*(f/1024) : e===31 ? (f?NaN:s*Infinity) : s*2**(e-15)*(1+f/1024);
  }
  return out;
}
const rows = (flat,cols) => { const o=[]; for(let i=0;i<flat.length;i+=cols) o.push(Array.from(flat.subarray(i,i+cols))); return o; };

let LUT=[];
function buildLUT(){
  const hex = h => [1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
  const stops = Array.from({length:9},(_,i)=>hex(css('--ramp-'+i)));
  LUT = Array.from({length:256},(_,i)=>{
    const t=i/255*8, k=Math.min(7,Math.floor(t)), f=t-k;
    return stops[k].map((v,j)=>Math.round(v+(stops[k+1][j]-v)*f));
  });
}
// A log with a soft floor, not a gamma: counts are sparse and long-tailed, and a gamma
// fitted to a dark background crushes the quiet baseline into the page. The scale is set
// by the data actually present — counts reach 6 and predicted rates sit near 0.26 — so a
// curve built for rates up to 24 would render the entire field as blank paper.
const RMAX = Math.log1p(6);
const norm = r => Math.max(0,Math.min(255,Math.round(Math.log1p(Math.max(0,r))/RMAX*255)));

let tapes=[], models={}, active='multistep', day=0, cut=0;

function paint(canvas, grid, cols){
  const dpr=devicePixelRatio||1;
  canvas.width=Math.round(canvas.clientWidth*dpr);
  canvas.height=Math.round(canvas.clientHeight*dpr);
  const ctx=canvas.getContext('2d'), img=ctx.createImageData(canvas.width,canvas.height);
  const cw=canvas.width/cols, rh=canvas.height/CH, paper=[250,247,240];
  for (let y=0;y<canvas.height;y++){
    const ch=Math.min(CH-1,Math.floor(y/rh));
    for (let x=0;x<canvas.width;x++){
      const col=grid[Math.floor(x/cw)];
      const v = col ? col[ch] : undefined;
      const c = v===undefined ? paper : LUT[norm(v)];
      const p=(y*canvas.width+x)*4;
      img.data[p]=c[0]; img.data[p+1]=c[1]; img.data[p+2]=c[2]; img.data[p+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
}

let current=null;
function render(revealed){
  const tape=tapes[day];
  const start=Math.max(0,cut-HIST);
  const hist=tape.slice(start,cut);
  if (!current || current.cut!==cut || current.day!==day || current.model!==active)
    current={cut,day,model:active,pred:forecast(tape.slice(cut-W,cut),H,models[active])};
  const truth=tape.slice(cut,cut+H);
  paint(document.getElementById('r-hist'), hist, hist.length);
  paint(document.getElementById('r-pred'), current.pred, H);
  paint(document.getElementById('r-true'), truth.slice(0,revealed), H);
  paint(document.getElementById('r-err'),
        truth.slice(0,revealed).map((t,i)=>t.map((v,c)=>Math.abs(v-current.pred[i][c]))), H);
  document.getElementById('cut').style.left=(document.getElementById('field').clientWidth-1)+'px';
  document.getElementById('a-left').textContent=(start*C.bin_ms/1000).toFixed(1)+' s';
  document.getElementById('a-right').textContent='+'+(H*C.bin_ms)+' ms';
  document.getElementById('cut').setAttribute('aria-valuenow', Math.round(100*cut/tape.length));
}

// Commit the forecast, hold it alone, then let the recording arrive. The pause is the
// point: the model is on the record before the answer is visible.
let timer=null;
function commit(){
  clearTimeout(timer);
  const state=document.getElementById('d-state');
  render(0); state.textContent='forecast committed';
  if (matchMedia('(prefers-reduced-motion: reduce)').matches){
    render(H); state.textContent='recording checked'; plot(); return;
  }
  let shown=0;
  timer=setTimeout(function step(){
    shown++; render(shown);
    state.textContent = shown>=H ? 'recording checked' : 'recording arriving';
    if (shown<H) timer=setTimeout(step,90); else plot();
  },400);
}

function plot(){
  const c=document.getElementById('plot'), dpr=devicePixelRatio||1;
  const w=c.clientWidth, h=c.clientHeight;
  c.width=w*dpr; c.height=h*dpr;
  const x=c.getContext('2d'); x.setTransform(dpr,0,0,dpr,0,0); x.clearRect(0,0,w,h);
  const L=52,R=118,T=14,B=28, d=A.calendar[day], top=0.6;
  const X=i=>L+(i/(H-1))*(w-L-R), Y=v=>T+(1-v/top)*(h-T-B);
  x.font='12px "Noema Data",monospace'; x.strokeStyle=css('--rule-hi'); x.lineWidth=1;
  x.beginPath(); x.moveTo(L,Y(0)); x.lineTo(w-R,Y(0)); x.stroke();
  x.fillStyle=css('--ink-3');
  x.fillText('0',10,Y(0)+4); x.fillText(top.toFixed(1),10,Y(top)+4);
  x.fillText(C.bin_ms+' ms',L-4,h-8); x.fillText(H*C.bin_ms+' ms',w-R-40,h-8);
  const series=[
    ['centred', d[active].centred, css('--ink-1'), 2, []],
    ['not centred', d[active].raw, css('--ink-3'), 1, []],
    ['persistence', d[active].persistence, css('--ink-2'), 1, [4,3]],
    ['channel mean', d[active].channel_mean, css('--ink-4'), 1, [1,3]],
  ];
  for (const [lab,vals,col,lw,dash] of series){
    x.strokeStyle=col; x.lineWidth=lw; x.setLineDash(dash); x.beginPath();
    vals.forEach((v,i)=> i?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v)));
    x.stroke(); x.setLineDash([]);
    x.fillStyle=col; x.fillText(lab, w-R+8, Y(vals[vals.length-1])+4);
  }
}

function select(){
  const d=A.calendar[day];
  document.getElementById('d-day').textContent=d.day;
  document.getElementById('d-split').textContent=d.split;
  document.getElementById('d-ses').textContent=d.session;
  document.querySelectorAll('.day').forEach(b=>b.setAttribute('aria-current',+b.dataset.i===day));
}

function ledger(){
  const d=A.calendar, mean=(rs,k,i)=>rs.reduce((s,r)=>s+r[k].centred[i],0)/rs.length;
  const hi=d.filter(r=>r.split==='held-in'), ho=d.filter(r=>r.split==='held-out');
  const out=[
    ['Forecast skill, one step ahead', mean(d,'multistep',0).toFixed(3),
     'Centred population correlation over thirteen recordings. Raw is about twice this, which is why raw is not the headline.'],
    ['Forecast skill, ten steps ahead', mean(d,'multistep',H-1).toFixed(3),
     'The rollout objective holds here. The one-step objective reaches '+mean(d,'onestep',H-1).toFixed(3)+' on the same recordings.'],
    ['Skill lost across 39 days', (100*(1-mean(ho,'multistep',0)/mean(hi,'multistep',0))).toFixed(0)+'%',
     'Held-in days against held-out days, same protocol on both. This is drift, not a benchmark score.'],
  ];
  document.getElementById('ledger').innerHTML=out.map(([a,b,c])=>
    `<tr><td>${a}</td><td class="n">${b}</td><td>${c}</td></tr>`).join('');
  document.getElementById('prov').textContent =
    `${A.manifest.assets.length} assets · ${(A.manifest.base64_bytes/1e6).toFixed(2)} MB · `+
    `every recording here is a session the models never read · DANDI:000954`;
  document.getElementById('gates').textContent =
    'Browser against PyTorch: maximum difference 2.7 × 10⁻⁶ measured against a 2 × 10⁻³ gate, '+
    'on eight fixed cuts for both models. Run node demo/parity.mjs to reproduce it.';
}

async function boot(){
  buildLUT();
  for (const t of A.tapes) tapes.push(rows(await ungzip(t.b64), CH));
  for (const [tag,w] of Object.entries(A.weights))
    models[tag]=build(unpack(decodeF16(await ungzip(w.b64)), w.layout), C);

  const spine=document.getElementById('spine');
  const peak=Math.max(...A.calendar.map(d=>d.multistep.centred[0]));
  A.calendar.forEach((d,i)=>{
    if (i && d.split!=='held-in' && A.calendar[i-1].split==='held-in'){
      const s=document.createElement('div');
      s.className='cutoff'; s.textContent='last day in training';
      spine.appendChild(s);
    }
    const b=document.createElement('button');
    b.className='day'; b.type='button'; b.dataset.i=i;
    b.innerHTML=`<span>${d.day}</span><i style="width:${Math.round(d.multistep.centred[0]/peak*46)}px"></i>`;
    b.title=`day ${d.day}, ${d.split}`;
    b.onclick=()=>{ day=i; cut=Math.floor(tapes[day].length*0.7); current=null; select(); commit(); };
    spine.appendChild(b);
  });

  day=0; cut=Math.floor(tapes[0].length*0.7);
  select(); ledger(); commit();

  const cutEl=document.getElementById('cut'), field=document.getElementById('field');
  let drag=false, base=0, span=HIST+H;
  const place=e=>{
    const r=field.getBoundingClientRect();
    const frac=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
    cut=Math.max(W,Math.min(tapes[day].length-H-1, base+Math.round((frac-0.001)*span)-HIST));
    current=null; render(0);
  };
  cutEl.addEventListener('pointerdown',e=>{drag=true;base=cut;cutEl.setPointerCapture(e.pointerId);});
  cutEl.addEventListener('pointermove',e=>{ if(drag) place(e); });
  cutEl.addEventListener('pointerup',()=>{ if(drag){drag=false;commit();} });
  cutEl.addEventListener('keydown',e=>{
    const step={ArrowLeft:-20,ArrowRight:20,PageDown:-200,PageUp:200}[e.key];
    if(!step) return;
    e.preventDefault();
    cut=Math.max(W,Math.min(tapes[day].length-H-1,cut+step));
    current=null; commit();
  });
  for (const tag of ['multistep','onestep'])
    document.getElementById('t-'+tag).onclick=()=>{
      active=tag; current=null;
      for (const t of ['multistep','onestep'])
        document.getElementById('t-'+t).setAttribute('aria-pressed',t===tag);
      commit();
    };
  addEventListener('resize',()=>{render(H);plot();});
}

boot();
</script></body></html>"""


if __name__ == "__main__":
    build_html()
