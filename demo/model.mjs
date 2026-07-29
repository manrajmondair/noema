// Noema forward pass in plain JS, shared by the browser page and the Node parity test.
// Mirrors noema/models {tokenizer, encoder, world_model} exactly, at batch size one.
//
// The page encodes real recorded spikes, so unlike the previous version this carries the
// tokenizer and the encoder as well as the world model. The encoder is BIDIRECTIONAL and
// the world model is CAUSAL, which is the same asymmetry as the Python: a window that
// ends at the cut may look across itself, and a rollout may not look ahead.

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

function attention(X, blk, cfg, cos, sin, causal) {   // X: [T][dim]
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
    const row = [], last = causal ? t : T - 1;
    for (let h = 0; h < heads; h++) {
      const scores = [];
      let mx = -Infinity;
      for (let j = 0; j <= last; j++) {
        const s = dot(q[t][h], k[j][h]) / Math.sqrt(headDim);
        scores.push(s); if (s > mx) mx = s;
      }
      let z = 0; const w = scores.map((s) => { const e = Math.exp(s - mx); z += e; return e; });
      const ctx = new Array(headDim).fill(0);
      for (let j = 0; j <= last; j++) for (let d = 0; d < headDim; d++) ctx[d] += (w[j] / z) * v[j][h][d];
      row.push(...ctx);
    }
    out.push(linear(row, blk.proj));
  }
  return out;
}

function block(X, blk, cfg, cos, sin, causal) {
  const normed = X.map((x) => layernorm(x, blk.n1.w, blk.n1.b));
  const att = attention(normed, blk, cfg, cos, sin, causal);
  const h = X.map((x, t) => x.map((e, i) => e + att[t][i]));
  return h.map((x) => {
    const m = layernorm(x, blk.n2.w, blk.n2.b);
    const mlp = linear(linear(m, blk.mlp0.W, blk.mlp0.b).map(gelu), blk.mlp2.W, blk.mlp2.b);
    return x.map((e, i) => e + mlp[i]);
  });
}

function stack(X, blocks, norm, cfg, causal) {
  const { cos, sin } = rotaryTables(cfg.headDim, X.length, cfg.base);
  let h = X;
  for (const blk of blocks) h = block(h, blk, cfg, cos, sin, causal);
  return h.map((x) => layernorm(x, norm.w, norm.b));
}

const cfgOf = (m) => ({ heads: m.heads, headDim: m.head_dim, dim: m.dim, base: m.base });

// Spike counts -> one token per bin. log1p tames the count range, exactly as the
// Python tokenizer does; the model never sees a raw count.
export function tokenize(counts, m) {
  return counts.map((bin) => {
    const z = new Array(m.dim).fill(0);
    for (let n = 0; n < bin.length; n++) {
      if (bin[n] === 0) continue;                 // most bins are empty; skip the work
      const v = Math.log1p(bin[n]), row = m.embed[n];
      for (let d = 0; d < m.dim; d++) z[d] += v * row[d];
    }
    return z;
  });
}

// Encode a window of recorded spikes. Bidirectional: the window ends at the cut, so
// attending across it uses no information from after the mark.
export function encode(counts, m) {
  return stack(tokenize(counts, m), m.encoder.blocks, m.encoder.norm, cfgOf(m), false);
}

// World model over a latent sequence -> the predicted next latent. Causal.
export function worldStep(zSeq, m) {
  const h = stack(zSeq, m.world.blocks, m.world.norm, cfgOf(m), true);
  return linear(h[h.length - 1], m.world.head.W, m.world.head.b);
}

export function decode(z, m) {   // latent -> per-unit firing rate
  return m.readout.map((row, u) => Math.exp(dot(z, row) * m.scale + m.bias[u]));
}

// Roll the world model forward from a window of real spikes, returning the predicted
// firing rate for each of the next `steps` bins. Mirrors noema.sim.rollout.imagine with
// no action channel: FALCON has none, and inventing one would be a claim about data
// that does not contain it.
export function forecast(counts, steps, m) {
  const z = encode(counts, m);
  const rates = [];
  for (let i = 0; i < steps; i++) {
    const next = worldStep(z, m);
    z.push(next);
    rates.push(decode(next, m));
  }
  return rates;
}

// Population correlation across channels within one bin, and the centred form that
// removes the static per-channel firing profile. The raw number is roughly twice the
// centred one on this data, so which is reported is not a presentation choice.
export function correlate(a, b) {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sa = 0, sb = 0, sab = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sa += da * da; sb += db * db; sab += da * db;
  }
  return sa < 1e-12 || sb < 1e-12 ? NaN : sab / Math.sqrt(sa * sb);
}

// Unpack the float16 block into named tensors. Slicing by name rather than by
// iteration order is deliberate: two languages agreeing on dict order is not a
// property worth trusting a silent weight-shuffle to.
export function unpack(flat, layout) {
  const named = {};
  let off = 0;
  for (const { name, shape, n } of layout) {
    const flatSlice = flat.subarray(off, off + n); off += n;
    named[name] = shape.length === 1 ? Array.from(flatSlice) : reshape(flatSlice, shape);
  }
  return named;
}

function reshape(flat, [rows, cols]) {
  const out = new Array(rows);
  for (let r = 0; r < rows; r++) out[r] = Array.from(flat.subarray(r * cols, (r + 1) * cols));
  return out;
}

/** Assemble the named tensors into the structure the forward pass expects. */
export function build(named, config) {
  const blocks = (prefix, depth) => Array.from({ length: depth }, (_, i) => ({
    n1: { w: named[`${prefix}.${i}.norm1.weight`], b: named[`${prefix}.${i}.norm1.bias`] },
    n2: { w: named[`${prefix}.${i}.norm2.weight`], b: named[`${prefix}.${i}.norm2.bias`] },
    qkv: named[`${prefix}.${i}.attn.qkv.weight`],
    proj: named[`${prefix}.${i}.attn.proj.weight`],
    mlp0: { W: named[`${prefix}.${i}.mlp.0.weight`], b: named[`${prefix}.${i}.mlp.0.bias`] },
    mlp2: { W: named[`${prefix}.${i}.mlp.2.weight`], b: named[`${prefix}.${i}.mlp.2.bias`] },
  }));
  return {
    dim: config.dim, heads: config.heads, head_dim: config.dim / config.heads,
    base: 10000.0, scale: config.dim ** -0.5,
    embed: named["tokenizer.embed.weight"],
    readout: named["tokenizer.readout.weight"],
    bias: named["tokenizer.bias.weight"].map((r) => r[0]),
    encoder: {
      blocks: blocks("encoder.blocks", config.enc_depth),
      norm: { w: named["encoder.norm.weight"], b: named["encoder.norm.bias"] },
    },
    world: {
      blocks: blocks("world.core.blocks", config.wm_depth),
      norm: { w: named["world.core.norm.weight"], b: named["world.core.norm.bias"] },
      head: { W: named["world.head.weight"], b: named["world.head.bias"] },
    },
  };
}
