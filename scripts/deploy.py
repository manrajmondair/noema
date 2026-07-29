#!/usr/bin/env python3
"""Stage the public site and emit a policy that matches what is actually being served.

    python scripts/deploy.py            # stage only, print the directory
    python scripts/deploy.py --ship     # stage, then deploy to production

Two pages ship: the site at `/`, and the self-contained figure at `/forecast`. The
figure carries its script and style inline — it has to, since it is also handed around
as a single file — so the policy has to name their hashes. Those hashes change every
time the figure is regenerated, which is why this is a script and not a checked-in
constant: the last deployment's hashes were set by hand and could not be reproduced.

Nothing is staged that was not asked for. `demo/site` accumulates a Vercel link file
and an OIDC token during deployment, and uploading those would leak a credential.
"""

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "demo" / "site"
FIGURE = ROOT / "demo" / "noema.html"
# Each deployment gets its own URL, so the public address has to be re-pointed at it.
ALIAS = "noema-bci.vercel.app"
PROJECT = "noema"
CLI = "58.1.0"  # 58.3.0 resolves a dependency that is not published

# Everything the browser is allowed to ask for, named one by one. A directory copy
# would sweep up .env.local and .vercel/ along with it.
ASSETS = ["index.html", "styles.css", "study.mjs", "mode.mjs",
          "study-data.json", "noema.woff2"]

INLINE = re.compile(r"<(script|style)(?![^>]*\bsrc=)[^>]*>(.*?)</\1>", re.S)


def hashes(html):
    """CSP source expressions for every inline script and style in the document."""
    found = {"script": [], "style": []}
    for kind, body in INLINE.findall(html):
        digest = hashlib.sha256(body.encode()).digest()
        found[kind].append(f"'sha256-{base64.b64encode(digest).decode()}'")
    return found


def policy(html):
    found = hashes(html)
    script = " ".join(["'self'", *found["script"]])
    style = " ".join(["'self'", *found["style"]])
    return (f"default-src 'self'; script-src {script}; style-src {style}; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")


def config(html):
    """One policy over both pages. A hash only admits content that matches it, so
    naming the figure's inline blocks loosens nothing for the page without any."""
    return {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "cleanUrls": True,
        "trailingSlash": False,
        "headers": [
            {"source": "/(.*)", "headers": [
                {"key": "Content-Security-Policy", "value": policy(html)},
                {"key": "Referrer-Policy", "value": "no-referrer"},
                {"key": "Permissions-Policy",
                 "value": "camera=(), microphone=(), geolocation=()"},
                {"key": "X-Content-Type-Options", "value": "nosniff"},
            ]},
            {"source": "/study-data.json", "headers": [
                {"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"},
            ]},
        ],
    }


def stage(into):
    figure = FIGURE.read_text()
    for name in ASSETS:
        shutil.copy2(SITE / name, into / name)
    # cleanUrls turns forecast.html into /forecast, which is what the site links to.
    (into / "forecast.html").write_text(figure)
    (into / "vercel.json").write_text(json.dumps(config(figure), indent=2) + "\n")

    return into


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", action="store_true", help="deploy the staged directory")
    args = ap.parse_args()

    missing = [n for n in ASSETS if not (SITE / n).exists()]
    if missing or not FIGURE.exists():
        sys.exit(f"missing: {', '.join(missing) or FIGURE}")

    out = stage(Path(tempfile.mkdtemp(prefix="noema-site-")))
    total = sum(p.stat().st_size for p in out.iterdir())
    for p in sorted(out.iterdir()):
        print(f"  {p.name:<20} {p.stat().st_size:>9,}")
    print(f"  {'':<20} {total:>9,} total\n{out}")

    if args.ship:
        vercel = ["npx", f"vercel@{CLI}"]
        # Link by name, not by copying a project.json around: the one in demo/site
        # pointed at a project that no longer exists, and an unresolvable link makes
        # the CLI quietly create a new project named after the temporary directory.
        # Everything below runs inside the staged directory — given a path argument
        # instead, the CLI reads the link from the working directory and ignores it.
        subprocess.run([*vercel, "link", "--yes", "--project", PROJECT], cwd=out, check=True)
        # Linking drops an OIDC token here, and unlike .vercel/ this one would upload.
        for leaked in out.glob(".env*.local"):
            leaked.unlink()
        # The deployment URL is the only thing the CLI writes to stdout. `alias set`
        # cannot infer it, so name the deployment explicitly rather than letting it
        # guess and fail after the upload has already happened.
        done = subprocess.run([*vercel, "deploy", "--prod", "--yes"],
                              cwd=out, check=True, capture_output=True, text=True)
        url = done.stdout.strip().splitlines()[-1]
        subprocess.run([*vercel, "alias", "set", url, ALIAS], cwd=out, check=True)
        print(f"{ALIAS} -> {url}")


if __name__ == "__main__":
    main()
