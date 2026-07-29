# Fonts

Subsets built by [`scripts/build_fonts.sh`](../../scripts/build_fonts.sh), all under
the SIL Open Font License 1.1 (<https://scripts.sil.org/OFL>), which permits the
embedding, subsetting and redistribution a self-contained page requires.

| File | Role | Upstream | Copyright |
|---|---|---|---|
| `noema-display.woff2` | headings | [Cormorant](https://github.com/CatharsisFonts/Cormorant) | Copyright 2015 The Cormorant Project Authors |
| `noema-text.woff2` | body | [EB Garamond](https://github.com/octaviopardo/EBGaramond12) | Copyright 2017 The EB Garamond Project Authors |
| `noema-data.woff2` | labels, readout | [Noto Sans Mono](https://github.com/notofonts/latin-greek-cyrillic) | Copyright The Noto Project Authors |

No upstream here declares a Reserved Font Name, so the subsets keep their original
name and copyright records; `Noema Display`, `Noema Text` and `Noema Data` are local
CSS aliases, not a renaming of the font software. Full licence text ships alongside,
and name IDs 13 and 14 are retained inside every binary, as OFL section 1 requires.

## Why three faces

One face could not do all three jobs on this page.

- **Cormorant** carries the canted axis and the high stroke contrast the design's
  voice rests on, but it lacks `θ`, `‖` and superscript minus, and its digits have
  nine different advances. It is therefore confined to headings, where none of those
  characters appear.
- **EB Garamond** has the same canted axis with full coverage of the page's character
  set, and its figures are lining and tabular with no feature settings at all.
- **Noto Sans Mono** is the only monospace measured that covers every glyph the live
  readout renders — including `‖`, which most monospaces omit — while keeping a single
  advance class, so the readout cannot shift as it updates.

## Choices that are load-bearing, not cosmetic

- `--name-IDs+=13,14`. The default set drops name ID 13, the licence record. Overriding
  it outright ships an OFL face with its licence stripped.
- `lnum` and `tnum` on the serifs. Cormorant's default figures are oldstyle, and
  `pyftsubset` drops unlisted features silently, so omitting them turns
  `font-variant-numeric: tabular-nums` into a no-op with no warning.
- No layout features at all on the monospace. One advance class is a metric property
  of the face, not a feature, so nothing needs to survive subsetting for the readout
  to stay steady.
- `U+2016` and `U+00B7` are in the character set. The first is the readout's double
  bar; the second is the page's separator, used 41 times. Both were missing from the
  first cut of this build.
