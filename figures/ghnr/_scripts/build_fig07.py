#!/usr/bin/env python
"""Fig. 7 (GNHR version): what survives ten proposals per image, rebuilt in English as vector.

Why this script exists: the figure shipped in the assembled draft (`fig07_verifier.png`) was a
300-dpi raster exported from the Chinese production run. Its axis label, variant names and title
were CJK and rendered as empty boxes in the PDF, i.e. the figure was illegible in the submitted
file. This script regenerates the same numbers with English labels straight into a vector PDF.

Data: 02_图表/fig07_verifier.csv.
Source: 02_图/figs_论文图_v2/figure_data.json -> figA_shared_scorer_ablation
        (six verifier variants; per-arm recall at k = 10; ratio = class-agnostic / Grounding DINO;
         upper_bound_pf_self_AR10 = 0.2265).
Rows 1-3 are the three verifier families actually tested; rows 4-6 are arm-balance / selection
diagnostics, drawn in a separate block so that the manuscript's "1.9-2.9%" range (rows 1-3) and
the 3.3% reached with arm-balanced selection (rows 4-6) cannot be conflated.

Output: 02_图表/fig07_verifier.pdf (+ .png)
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "fig07_verifier"
UPPER_BOUND = 0.2265  # PF-RPN's own verifier, AR@10; figure_data.json: upper_bound_pf_self_AR10

ARMS = [
    ("pf", "Class-agnostic proposer (this study)", "#0b4f8a"),
    ("gd", "Grounding DINO (zero-shot)", "#d9820a"),
    ("yw", "YOLO-World (zero-shot)", "#8c8c8c"),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

rows = list(csv.DictReader(open(BASE / "fig07_verifier.csv", encoding="utf-8")))
for r in rows:
    for key, _, _ in ARMS:
        r[key] = float(r[key])
    r["ratio"] = float(r["ratio"])

labels = [f"{r['variant']}  {r['label']}" for r in rows]
x = np.arange(len(rows))
width = 0.26

fig, ax = plt.subplots(figsize=(7.0, 3.9))

# diagnostic block shading (drawn first, under the bars)
ax.axvspan(2.5, len(rows) - 0.4, color="#f2f2f0", zorder=0, lw=0)

for i, (key, label, color) in enumerate(ARMS):
    ax.bar(x + (i - 1) * width, [r[key] * 100 for r in rows], width=width * 0.9,
           color=color, edgecolor="white", linewidth=0.35, label=label, zorder=3)

for xi, r in zip(x, rows):
    top = max(r["pf"], r["gd"], r["yw"]) * 100
    ax.text(xi, top + 0.55, f"{r['ratio']:.2f}$\\times$", ha="center", va="bottom",
            fontsize=7.0, color="#222222")

ax.axhline(UPPER_BOUND * 100, color="#c0392b", lw=0.9, ls="--", zorder=4)
ax.text(len(rows) - 0.45, UPPER_BOUND * 100 + 0.5,
        f"the proposer's own verifier: AR@10 = {UPPER_BOUND * 100:.1f}\\%",
        ha="right", va="bottom", fontsize=7.0, color="#c0392b")

# one vertical marker between families 1-3 and diagnostics 4-6
ax.axvline(2.5, color="#bbbbbb", lw=0.6, ls=":", zorder=1)
ax.text(4.5, 9.5, "arm-balance and\nselection diagnostics", ha="center", va="center",
        fontsize=6.8, color="#777777", linespacing=1.35)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.6, len(rows) - 0.4)
ax.set_ylim(0, 26)
ax.set_yticks(np.arange(0, 26, 5))
ax.set_ylabel("Recall at ten proposals per image (\\%)")
ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower center", frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.52),
          handlelength=1.2, columnspacing=1.4)

fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.32)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.14)
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.14)

fam = [r for r in rows if r["block"] == "family"]
dia = [r for r in rows if r["block"] == "diagnostic"]
print("families 1-3: proposer AR@10 = %.2f-%.2f%%" % (
    100 * min(r["pf"] for r in fam), 100 * max(r["pf"] for r in fam)))
print("diagnostics 4-6: proposer AR@10 up to %.2f%%" % (100 * max(r["pf"] for r in dia)))
print("wrote", OUT.with_suffix(".pdf").name)
