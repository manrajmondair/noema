// Proves the browser forward pass matches PyTorch: run the JS rollout on the
// exported reference actions and compare to the reference rates and behavior.
import { readFileSync } from "node:fs";
import { rollout } from "./model.mjs";

const m = JSON.parse(readFileSync(new URL("./model.json", import.meta.url)));
const out = rollout(m, m.reference.actions);

const maxDiff = (A, B) => {
  let d = 0;
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[i].length; j++) d = Math.max(d, Math.abs(A[i][j] - B[i][j]));
  return d;
};

const rateDiff = maxDiff(out.rates, m.reference.rates);
const behDiff = maxDiff(out.behavior, m.reference.behavior);
console.log(`rate max|Δ|=${rateDiff.toExponential(2)}  behavior max|Δ|=${behDiff.toExponential(2)}`);

const tol = 2e-3;  // float32 export -> float64 JS, accumulated over the rollout
if (rateDiff > tol || behDiff > tol) {
  console.error(`PARITY FAILED (tol ${tol})`);
  process.exit(1);
}
console.log("parity OK");
