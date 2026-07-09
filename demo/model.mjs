// Noema world model — forward pass in plain JS, shared by the browser demo and
// the Node parity test. Mirrors noema/models {encoder, world_model, tokenizer}
// exactly (batch size 1). The neural encoder is not needed here: the seed latents
// are baked in, so only the action-conditioned world model rolls forward.

const erf = (x) => {                              // Abramowitz-Stegun 7.1.26, ~1e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x < 0 ? -y : y;
};
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const linear = (x, W, b) => W.map((row, o) => dot(x, row) + (b ? b[o] : 0));

function layernorm(x, w, b, eps = 1e-5) {
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  const v = x.reduce((s, e) => s + (e - m) * (e - m), 0) / x.length;
  const inv = 1 / Math.sqrt(v + eps);
  return x.map((e, i) => (e - m) * inv * w[i] + b[i]);
}

function rotary(vec, cos, sin) {          // vec: [head_dim]; cos/sin: [head_dim/2]
  const h = vec.length / 2, out = new Array(vec.length);
  for (let d = 0; d < h; d++) {
    const x1 = vec[d], x2 = vec[d + h];
    out[d] = x1 * cos[d] - x2 * sin[d];
    out[d + h] = x2 * cos[d] + x1 * sin[d];
  }
  return out;
}

function rotaryTables(headDim, seq, base) {
  const half = headDim / 2, cos = [], sin = [];
  for (let t = 0; t < seq; t++) {
    const c = [], s = [];
    for (let i = 0; i < half; i++) {
      const ang = t / base ** ((2 * i) / headDim);
      c.push(Math.cos(ang)); s.push(Math.sin(ang));
    }
    cos.push(c); sin.push(s);
  }
  return { cos, sin };
}

function attention(X, blk, cfg, cos, sin) {   // X: [T][dim], causal self-attention
  const { heads, headDim, dim } = cfg, T = X.length;
  const q = [], k = [], v = [];
  for (let t = 0; t < T; t++) {
    const p = linear(X[t], blk.qkv);
    const qh = [], kh = [], vh = [];
    for (let h = 0; h < heads; h++) {
      const o = h * headDim;
      qh.push(rotary(p.slice(o, o + headDim), cos[t], sin[t]));
      kh.push(rotary(p.slice(dim + o, dim + o + headDim), cos[t], sin[t]));
      vh.push(p.slice(2 * dim + o, 2 * dim + o + headDim));
    }
    q.push(qh); k.push(kh); v.push(vh);
  }
  const out = [];
  for (let t = 0; t < T; t++) {
    const row = [];
    for (let h = 0; h < heads; h++) {
      const scores = [];
      let mx = -Infinity;
      for (let j = 0; j <= t; j++) {                      // causal: j <= t
        const s = dot(q[t][h], k[j][h]) / Math.sqrt(headDim);
        scores.push(s); if (s > mx) mx = s;
      }
      let z = 0; const w = scores.map((s) => { const e = Math.exp(s - mx); z += e; return e; });
      const ctx = new Array(headDim).fill(0);
      for (let j = 0; j <= t; j++) for (let d = 0; d < headDim; d++) ctx[d] += (w[j] / z) * v[j][h][d];
      row.push(...ctx);
    }
    out.push(linear(row, blk.proj));
  }
  return out;
}

function block(X, blk, cfg, cos, sin) {
  const normed = X.map((x) => layernorm(x, blk.n1.w, blk.n1.b));
  const att = attention(normed, blk, cfg, cos, sin);
  const h = X.map((x, t) => x.map((e, i) => e + att[t][i]));
  return h.map((x) => {
    const m = layernorm(x, blk.n2.w, blk.n2.b);
    const mlp = linear(linear(m, blk.mlp0.W, blk.mlp0.b).map(gelu), blk.mlp2.W, blk.mlp2.b);
    return x.map((e, i) => e + mlp[i]);
  });
}

// World model over a latent+action sequence -> predicted next latent at each step.
function worldStep(zSeq, aSeq, m) {
  const cfg = { heads: m.heads, headDim: m.head_dim, dim: m.dim };
  const X = zSeq.map((z, t) => {
    const a = linear(aSeq[t], m.world.action.W, m.world.action.b);
    return z.map((e, i) => e + a[i]);
  });
  const { cos, sin } = rotaryTables(m.head_dim, X.length, m.base);
  let h = X;
  for (const blk of m.world.blocks) h = block(h, blk, cfg, cos, sin);
  h = h.map((x) => layernorm(x, m.world.norm.w, m.world.norm.b));
  const last = linear(h[h.length - 1], m.world.head.W, m.world.head.b);  // next latent
  return last;
}

function decode(z, m) {   // latent -> per-unit firing rate (exp of log-rate)
  return m.readout.map((row, u) => Math.exp(dot(z, row) * m.scale + m.bias[u]));
}

function behavior(z, m) {
  return linear(linear(z, m.behavior.n0.W, m.behavior.n0.b).map(gelu), m.behavior.n2.W, m.behavior.n2.b);
}

// Roll the world model forward under a plan of actions, from the seed latents.
export function rollout(m, futureActions) {
  const z = m.seed.z.map((r) => r.slice());
  const a = m.seed.actions.map((r) => r.slice());
  const rates = [], beh = [];
  for (const action of futureActions) {
    const next = worldStep(z, a, m);
    z.push(next); a.push(action.slice());
    rates.push(decode(next, m));
    beh.push(behavior(next, m));
  }
  return { rates, behavior: beh };
}

export { decode, behavior, worldStep };
