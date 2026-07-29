#!/usr/bin/env bash
# Rebuild the subset faces the demo inlines.
#
#   scripts/build_fonts.sh          # -> demo/fonts/*.woff2
#
# Three roles, because one face cannot do all three jobs on this page:
#
#   display  Cormorant     — canted axis and high stroke contrast, which is what
#                            the design's voice rests on, but it lacks theta, the
#                            double bar and superscript minus and has nine digit
#                            advances, so it is confined to headings.
#   text     EB Garamond   — the same canted axis with full coverage of the page's
#                            character set, and figures that are lining and tabular
#                            with no feature settings at all.
#   data     Noto Sans Mono — the only monospace measured that covers every glyph
#                            the live readout renders (notably U+2016) while keeping
#                            one advance class, so the readout cannot shift.
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
curl -sfL "$GF/ebgaramond/EBGaramond%5Bwght%5D.ttf"          -o "$WORK/text.ttf"
curl -sfL "$GF/notosansmono/NotoSansMono%5Bwdth,wght%5D.ttf" -o "$WORK/data.ttf"

inst() { "$WORK/v/bin/fonttools" varLib.instancer -q "$1" "${@:3}" -o "$2"; }
# The display face keeps a live weight axis because the headline uses it. Text and
# data are pinned: the design sets one weight everywhere and disables synthesis, so
# shipping a range they never reach is pure bytes. Pinning roughly halves each.
inst "$WORK/display.ttf" "$WORK/display-vf.ttf" wght=400:700
inst "$WORK/text.ttf"    "$WORK/text-vf.ttf"    wght=400
inst "$WORK/data.ttf"    "$WORK/data-vf.ttf"    wdth=100 wght=400

# --name-IDs+= ADDS to the default set. Overriding it outright drops name ID 13,
# the licence record, which OFL section 1 requires to travel with every copy.
COMMON=(--flavor=woff2 --no-hinting --drop-tables+=DSIG,MVAR,meta,vhea,vmtx
        --name-IDs+=13,14 --notdef-outline)

# The page's own character set, extracted from the rendered copy. U+2016 is the
# readout's double bar and U+00B7 is its separator, used 41 times; both were missing
# from the first cut of this build. U+2155 is deliberately absent — the copy says
# "one fifth" rather than a vulgar fraction, which no serif here covers anyway.
FULL="U+0020-007E,U+00A0,U+00A7,U+00B0-00B3,U+00B7,U+00B9,U+00D7,U+03B8,U+2009,U+2011,\
U+2013-2014,U+2016,U+2018-2019,U+201C-201D,U+2026,U+2032,U+2070-2071,U+2074-2079,U+207B,U+2192,U+2212"
# Headings never carry a readout glyph, so the display cut sidesteps its own gaps.
HEAD="U+0020-007E,U+00B7,U+2013-2014,U+2018-2019,U+201C-201D,U+2026"

mkdir -p "$OUT"
# lnum and tnum are load-bearing on the serifs: default figures are oldstyle on
# Cormorant, and pyftsubset drops unlisted features silently, which would turn
# font-variant-numeric into a no-op with no warning. cv01/cv02 carry the long
# sheared-tail Q the display voice is built around.
"$WORK/v/bin/pyftsubset" "$WORK/display-vf.ttf" --output-file="$OUT/noema-display.woff2" \
  --unicodes="$HEAD" --layout-features="kern,liga,lnum,tnum,cv01,cv02" "${COMMON[@]}"
"$WORK/v/bin/pyftsubset" "$WORK/text-vf.ttf" --output-file="$OUT/noema-text.woff2" \
  --unicodes="$FULL" --layout-features="kern,liga,clig,ccmp,locl,lnum,tnum,sups" "${COMMON[@]}"
# A monospace needs no figure features — one advance class is a metric property of
# the face. Everything else is dropped so no alternate can reach the labels.
"$WORK/v/bin/pyftsubset" "$WORK/data-vf.ttf" --output-file="$OUT/noema-data.woff2" \
  --unicodes="$FULL" --layout-features="" "${COMMON[@]}"

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
