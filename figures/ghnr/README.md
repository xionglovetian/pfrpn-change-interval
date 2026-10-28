# Figures for the *Geomatics, Natural Hazards and Risk* version

Eight figures of the manuscript are rebuilt from the tables in this folder. Figure numbering of
this version differs from the earlier one, so the mapping is given explicitly.

| manuscript figure | script | input | output |
|---|---|---|---|
| Figure 1 | `_scripts/build_fig01.py` | event inventory tree (see below) | `fig01a_event_panel.pdf/png`, `fig01b_events_map.pdf/png`, and `event_geo.csv` |
| Figure 2 | `_scripts/build_fig02.py` | none (vector schematic) | `fig02_framework.pdf/png` |
| Figure 3 | `_scripts/build_fig05_qualitative.py` | `italy_1161/` | `fig05_qualitative.pdf/png` |
| Figure 4 | `_scripts/build_fig03.py` | `fig03_perfold.csv` | `fig03_cross.pdf/png` |
| Figure 8 | `_scripts/build_fig07.py` | `fig07_verifier.csv` | `fig07_verifier.pdf/png` |

Figures 5, 6 and 7 are produced by the analysis code in `scripts/` and `results/`, not from this
folder.

## Requirements

Python 3 with `numpy` and `matplotlib` only. No GPU, no deep-learning framework.

```
python -m pip install numpy matplotlib
```

## Running

Run from anywhere; every script locates its inputs relative to its own path:

```
python figures/ghnr/_scripts/build_fig02.py
python figures/ghnr/_scripts/build_fig03.py
python figures/ghnr/_scripts/build_fig05_qualitative.py
python figures/ghnr/_scripts/build_fig07.py
```

Expected stdout (the numbers are the check that the inputs were read correctly):

| script | expected |
|---|---|
| `build_fig02.py` | `wrote fig02_framework.pdf  (5.67 x 4.35 in = printed size)` |
| `build_fig03.py` | `panel a: LOEO mean proposer = 0.6749 over 7 folds`, `panel b: LORO mean proposer = 0.7262 over 3 folds` |
| `build_fig05_qualitative.py` | `reference object 42 px`, `PF best IoU 0.6705, hits 1/900; YOLOv8s best IoU 0.0000 of 5` |
| `build_fig07.py` | `families 1-3: proposer AR@10 = 1.85-2.89%`, `diagnostics 4-6: proposer AR@10 up to 3.31%` |

`build_fig01.py` additionally needs the event inventory tree:

```
set PFRPN_EVENTS_ROOT=<your copy>\landslide_multimodal_lb_pos\events
python figures/ghnr/_scripts/build_fig01.py
```

It rewrites `event_geo.csv`; the version tracked in this folder is reproduced byte for byte
(SHA-256 `5d8f1e5b…`), which is a convenient check that the event tree is the expected one.

## Reproducibility caveat

PNGs are byte-identical between runs. PDFs are not: matplotlib stamps a creation time into the
PDF, so re-running a builder changes the PDF bytes even though the drawing is identical. Run
`python verify_checksums.py` **before** re-running the builders, otherwise the PDFs will report a
mismatch that reflects the timestamp rather than the content. Setting `SOURCE_DATE_EPOCH` makes the
PDFs deterministic as well:

```
set SOURCE_DATE_EPOCH=1793000000
```
