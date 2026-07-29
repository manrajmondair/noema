#!/usr/bin/env bash
# Rebuild the subset faces the demo inlines.
#
#   scripts/build_fonts.sh          # -> demo/fonts/*.woff2
#
# One face: Cormorant, the canted high-contrast serif the design's voice rests on.
# It covers every character the page renders except superscript minus, which the copy
# states in words instead. Its figures default to oldstyle at nine different widths,
# so lnum and tnum are retained and applied wherever numbers are compared — without
# them the page's numerals would shift as values change.
#
# Needs brotli for woff2, which the project venv deliberately does not carry, so
# this builds its own throwaway environment.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=demo/fonts
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python3 -m venv "$WORK/v"
"$WORK/v/bin/pip" -q install "fonttools[woff]" brotli

GF=https://github.com/google/fonts/raw/main/ofl
curl -sfL "$GF/cormorant/Cormorant%5Bwght%5D.ttf"            -o "$WORK/display.ttf"

inst() { "$WORK/v/bin/fonttools" varLib.instancer -q "$1" "${@:3}" -o "$2"; }
# The weight axis stays live: one face now carries the headline and the small labels,
# and that range is the only hierarchy left once there is a single voice.
inst "$WORK/display.ttf" "$WORK/display-vf.ttf" wght=400:700

# --name-IDs+= ADDS to the default set. Overriding it outright drops name ID 13,
# the licence record, which OFL section 1 requires to travel with every copy.
COMMON=(--flavor=woff2 --no-hinting --drop-tables+=DSIG,MVAR,meta,vhea,vmtx
        --name-IDs+=13,14 --notdef-outline)

# The page's own character set, extracted from what it actually renders.
FULL="U+0020-007E,U+00A0,U+00A7,U+00B0-00B3,U+00B7,U+00B9,U+00D7,U+03B8,U+2009,U+2011,\
U+2013-2014,U+2016,U+2018-2019,U+201C-201D,U+2026,U+2032,U+2070-2071,U+2074-2079,U+207B,U+2192,U+2212"

mkdir -p "$OUT"
# lnum and tnum are load-bearing: pyftsubset drops unlisted features silently, which
# would turn font-variant-numeric into a no-op with no warning at all. cv01/cv02 carry
# the long sheared-tail Q the voice is built around.
"$WORK/v/bin/pyftsubset" "$WORK/display-vf.ttf" --output-file="$OUT/noema.woff2" \
  --unicodes="$FULL" --layout-features="kern,liga,lnum,tnum,cv01,cv02" "${COMMON[@]}"

"$WORK/v/bin/python" - "$OUT" <<'PY'
import base64, glob, os, sys
from fontTools.ttLib import TTFont
total = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.woff2"))):
    t = TTFont(f); n = t["name"]; cmap = t.getBestCmap(); hm = t["hmtx"]
    b64 = len(base64.b64encode(open(f, "rb").read())); total += b64
    adv = {hm[cmap[c]][0] for c in map(ord, "0123456789") if c in cmap}
    lic = "ok" if n.getDebugName(13) else "MISSING LICENCE RECORD"
    print(f"{os.path.basename(f):<22} {os.path.getsize(f):>7,} raw  {b64/1024:>6.1f} KB b64  "
          f"{len(cmap):>4} glyphs  {len(adv)} digit advance(s)  licence {lic}")
print(f"{'TOTAL':<22} {'':>7}      {total/1024:>6.1f} KB b64")
PY
