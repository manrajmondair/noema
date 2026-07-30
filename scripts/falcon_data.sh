#!/usr/bin/env bash
# Fetch the FALCON H1 dandiset (~98 MB over 40 files, three splits).
#
#   scripts/falcon_data.sh              # -> data/000954
#   scripts/falcon_data.sh /some/where
#
# Every FALCON entry point defaults to data/000954. Nothing recorded how to obtain it,
# so reproducing any FALCON figure meant guessing the command.
#
# Expect 13 held-in-calib, 13 held-in-minival, 14 held-out-calib. The minival files are
# byte-identical prefixes of the matching calib recordings — that is a property of the
# dandiset, not a download error, and disjoint_calib() in noema/eval/falcon.py excises
# the overlap. Do not "fix" it here.
#
# Copying this tree off a Mac instead of downloading it: use COPYFILE_DISABLE=1 with
# tar, or macOS writes ._* xattr sidecars that the loader's *.nwb glob picks up as
# extra sessions.
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
cd "$DEST"

# dandi 0.75.x shipped lowercase defaults for --existing/--format that its own parser
# rejects, so both are named explicitly rather than left to the installed version.
dandi download --existing refresh --format pyout DANDI:000954

found=$(find 000954 -name '*.nwb' ! -name '._*' | wc -l | tr -d ' ')
[ "$found" -eq 40 ] || { echo "expected 40 recordings, found $found" >&2; exit 1; }
du -sh 000954
