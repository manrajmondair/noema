// Three gates between the browser and PyTorch. Run: node demo/parity.mjs
//
// Reported as measured-against-gate rather than as the word "verified": a passing
// number nobody can see the size of is not evidence.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { build, correlate, forecast, unpack } from "./model.mjs";

const GATES = { forward: 2e-3, metric: 1e-6 };

// Decoded by hand rather than with Float16Array, which is not in every runtime this
// has to run in. The browser page uses this same routine.
export function decodeF16(bytes) {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024)
      : e === 31 ? (f ? NaN : s * Infinity)
        : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

const raw = (b64) => new Uint8Array(gunzipSync(Buffer.from(b64, "base64")));
const rows = (flat, cols) => {
  const out = [];
  for (let i = 0; i < flat.length; i += cols) out.push(Array.from(flat.subarray(i, i + cols)));
  return out;
};

const bundle = JSON.parse(readFileSync(new URL("./assets.json", import.meta.url)));
const { config, weights, fixtures } = bundle;

let failed = 0;
const report = (name, measured, gate) => {
  const ok = measured <= gate;
  if (!ok) failed++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name.padEnd(30)} ${measured.toExponential(2)} against ${gate.toExponential(0)}`);
};

// Gate 1 — the forward pass. The weights here are float16 because float16 is what the
// browser runs, and the references were generated from the same rounded weights.
console.log("[1] forward pass, browser against PyTorch");
for (const [tag, packed] of Object.entries(weights)) {
  const model = build(unpack(decodeF16(raw(packed.b64)), packed.layout), config);
  let worst = 0;
  for (const fx of fixtures) {
    const seed = rows(raw(fx.seed.b64), config.channels);
    const got = forecast(seed, config.horizon, model);
    const want = rows(new Float32Array(raw(fx[tag].b64).buffer), config.channels);
    for (let h = 0; h < want.length; h++)
      for (let c = 0; c < want[h].length; c++)
        worst = Math.max(worst, Math.abs(got[h][c] - want[h][c]));
  }
  report(`${tag}, ${fixtures.length} cuts`, worst, GATES.forward);
}

// Gate 2 — the metric. A wrong correlation does not look wrong on screen; it looks like
// a slightly different result. So it gets its own tolerance rather than inheriting the
// forward pass's, which is three orders of magnitude looser.
console.log("[2] correlation, browser against an independent implementation");
{
  const a = [1, 2, 3, 4, 5, 6], b = [2, 1, 4, 3, 6, 5];
  const mean = (x) => x.reduce((s, v) => s + v, 0) / x.length;
  const ma = mean(a), mb = mean(b);
  const num = a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0);
  const den = Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0))
    * Math.sqrt(b.reduce((s, v) => s + (v - mb) ** 2, 0));
  report("pearson", Math.abs(correlate(a, b) - num / den), GATES.metric);
  // A constant channel has no correlation to report. Returning 0 would read as
  // "no relationship" when the truth is "undefined", so it must be NaN.
  report("constant input is NaN", Number.isNaN(correlate([1, 1, 1], [1, 2, 3])) ? 0 : 1, GATES.metric);
}

// Gate 3 — float16 decoding. Everything above rests on it, so it is checked against
// values whose exact representation is known.
console.log("[3] float16 decode");
{
  const bytes = new Uint8Array(new Uint16Array([0x3c00, 0xc000, 0x0000, 0x3555]).buffer);
  const got = decodeF16(bytes);
  const want = [1, -2, 0, 0.333251953125];
  report("known bit patterns", Math.max(...got.map((v, i) => Math.abs(v - want[i]))), GATES.metric);
}

console.log(failed ? `\n${failed} gate(s) failed` : "\nall gates passed");
process.exit(failed ? 1 : 0);
