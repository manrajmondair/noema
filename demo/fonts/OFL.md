# Fonts

Both faces here are subsets built by [`scripts/build_fonts.sh`](../../scripts/build_fonts.sh)
and are licensed under the SIL Open Font License, Version 1.1
(<https://scripts.sil.org/OFL>). The licence permits embedding, subsetting and
redistribution, which is what a self-contained page requires.

| File | Upstream | Copyright |
|---|---|---|
| `noema-display.woff2` | [Cormorant](https://github.com/CatharsisFonts/Cormorant) | Copyright 2015 The Cormorant Project Authors |
| `noema-data.woff2` | [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) | Copyright 2020 The JetBrains Mono Project Authors |

Neither upstream declares a Reserved Font Name, so the subsets keep their original
family and copyright records; `Noema Display` and `Noema Data` are local CSS aliases,
not a renaming of the font software. Full licence text travels with the files in
`OFL-Cormorant.txt` and `OFL-JetBrainsMono.txt`, as the licence requires.

Two subsetting choices are load-bearing rather than cosmetic:

- The display face keeps `lnum` and `tnum`. Its default figures are oldstyle, and
  `pyftsubset` drops unlisted features silently — omitting them turns
  `font-variant-numeric: tabular-nums` into a no-op with no warning.
- The data face drops `calt`. That is the code-ligature machinery, which would
  rewrite sequences like `>=` inside axis labels, and it accounts for most of the
  unsubset file.
