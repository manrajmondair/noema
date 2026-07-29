# Fonts

Subsets built by [`scripts/build_fonts.sh`](../../scripts/build_fonts.sh), all under
the SIL Open Font License 1.1 (<https://scripts.sil.org/OFL>), which permits the
embedding, subsetting and redistribution a self-contained page requires.

| File | Upstream | Copyright |
|---|---|---|
| `noema.woff2` | [Cormorant](https://github.com/CatharsisFonts/Cormorant) | Copyright 2015 The Cormorant Project Authors |

The upstream does not declare a Reserved Font Name, so the subsets keep their original
name and copyright records; `Noema` is a local CSS alias, not a renaming of the font software. Full licence text ships alongside,
and name IDs 13 and 14 are retained inside every binary, as OFL section 1 requires.

## Why one face

Cormorant is the canted, high-contrast serif the design's voice rests on. It covers
every character the page renders except superscript minus, which the copy states in
words rather than fake. Its figures default to oldstyle at nine different widths, so
`lnum` and `tnum` are retained and applied wherever numbers are compared — without
them the page's numerals would shift horizontally as the values change.

## Choices that are load-bearing, not cosmetic

- `--name-IDs+=13,14`. The default set drops name ID 13, the licence record. Overriding
  it outright ships an OFL face with its licence stripped.
- `lnum` and `tnum`. `pyftsubset` drops unlisted features silently, so omitting them
  turns `font-variant-numeric` into a no-op with no warning at all — and this face's
  default figures are oldstyle at nine different widths.
- The character set is extracted from what the page renders, not assumed. An earlier
  cut of this build omitted the separator the page used forty-one times.
