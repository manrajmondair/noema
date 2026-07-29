#!/usr/bin/env bash
# Rebuild the two subset faces the demo inlines.
#
# Both are shipped base64-inlined so the standalone page opens with no network at
# all. That costs 38 KB against a 2.5 MB page, which buys a display face whose
# canted axis and triangular serifs the design actually depends on.
#
#   scripts/build_fonts.sh          # -> demo/fonts/*.woff2
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
curl -sL "$GF/cormorant/Cormorant%5Bwght%5D.ttf"        -o "$WORK/display.ttf"
curl -sL "$GF/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf" -o "$WORK/data.ttf"

# Clip the weight axis to what the stylesheet uses. Shipping the full 100-800 range
# costs more than two static instances would; clipped, the variable file wins.
"$WORK/v/bin/fonttools" varLib.instancer -q "$WORK/display.ttf" wght=400:700 -o "$WORK/display-vf.ttf"
"$WORK/v/bin/fonttools" varLib.instancer -q "$WORK/data.ttf"    wght=400:700 -o "$WORK/data-vf.ttf"

COMMON=(--flavor=woff2 --no-hinting --drop-tables+=DSIG,MVAR,meta,vhea,vmtx
        --name-IDs=1,2,3,6 --no-name-legacy --notdef-outline)

# lnum+tnum are load-bearing: the default figures are oldstyle, and pyftsubset drops
# unlisted features silently, which makes tabular-nums a no-op with no warning.
# cv01/cv02 carry the long sheared-tail Q the display voice is built around.
mkdir -p "$OUT"
"$WORK/v/bin/pyftsubset" "$WORK/display-vf.ttf" --output-file="$OUT/noema-display.woff2" \
  --unicodes="U+0020-007E,U+00B0,U+00D7,U+2013,U+2014,U+2018,U+2019,U+201C,U+201D,U+2026,U+2212" \
  --layout-features="kern,liga,lnum,tnum,cv01,cv02" "${COMMON[@]}"

# calt is dropped on purpose: it is the code-ligature machinery, which would rewrite
# >= and != inside axis labels, and it accounts for most of the file.
"$WORK/v/bin/pyftsubset" "$WORK/data-vf.ttf" --output-file="$OUT/noema-data.woff2" \
  --unicodes="U+0020-007E,U+00B0,U+00B1,U+00B5,U+00B7,U+00D7,U+0394,U+03A3,U+03A9,U+03BC,U+03C0,U+03C3,U+2013,U+2014,U+2018,U+2019,U+201C,U+201D,U+2026,U+2032,U+2033,U+2190,U+2192,U+2211,U+2212,U+221A,U+221E,U+2248,U+2260,U+2264,U+2265" \
  --layout-features="zero" "${COMMON[@]}"

ls -l "$OUT"/*.woff2
