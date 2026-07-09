# Demo

`python demo/generate.py` trains a small action-conditioned model, imagines a
population forward from a seed window, and writes a self-contained `noema.html`
(open it directly — no server, no dependencies).

The page animates the imagined firing against ground truth and the decoded
behavior against the true trajectory. One rollout is shown, with ground truth
alongside so the match is self-evident. Open-loop neural rollouts stay faithful
over a short horizon and drift over longer ones.
