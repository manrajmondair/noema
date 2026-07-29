# Demo — steer the neural world model

`noema.html` is a self-contained interactive page (open it directly, no server):
drag the pad to command a movement and the action-conditioned world model
imagines the neural population's response and decodes it back into motion, live.

The browser runs the model's **exact** forward pass — `model.mjs` mirrors the
PyTorch encoder/world-model/tokenizer, and `parity.mjs` measures a maximum
difference of 3 × 10⁻⁵ against an enforced 2 × 10⁻³ gate. The neural encoder
isn't needed in the browser: the seed latents are precomputed, so only the world
model rolls forward.

```bash
scripts/build_fonts.sh     # rebuild the subset faces the page inlines
python demo/export.py      # train a small model, write model.json + noema.html
node demo/parity.mjs       # verify the JS forward pass matches PyTorch
```

`noema.html` is generated. Edit the template in `export.py` and regenerate — a
hand-edit is lost on the next build, and `tests/test_demo_template.py` guards the
properties that regression would break.

The demo decodes the *intended* movement, so steering the pad moves the cursor
that way. It is an illustration of the mechanism; the quantitative results live
in the tests and benchmarks.

Released under the MIT License; see [LICENSE](../LICENSE) at the repository root.
