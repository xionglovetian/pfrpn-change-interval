#!/usr/bin/env python
"""Fig. 3 (GNHR version): cross-event and cross-region recall, one script, two panels.

Why this script exists: `fig03a_cross_event` / `fig03b_cross_region` in the first assembled
draft came from two different production runs (origin/Fig 2 and origin/Fig 3) with different
figsize, different series sets and values that differed from Table 4 in the third decimal
(PF LOEO mean 0.6762 there vs 0.6749 in the table). This script plots Table 4 and nothing
else, so the figure cannot drift away from the table.

Data: 02_图表/fig03_perfold.csv, transcribed from Table 4 of GNHR_manuscript_v1.tex.
Provenance: 03_文档记录/cross_arm_ar.json (per-fold micro) aggregated image-averaged as
defined in the Method (Section 3.6, "Metric").

Output: 02_图表/fig03_cross.pdf (+ .png for inspection)
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "fig03_cross"

ARMS = [
    ("class_agnostic", "Class-agnostic proposer (this study)", "#0b4f8a"),
    ("yolo_world_ft", "YOLO-World (fine-tuned)", "#c0392b"),
    ("yolov8s", "YOLOv8s (closed-set)", "#7f6a4f"),
    ("grounding_dino", "Grounding DINO (zero-shot)", "#d9820a"),
    ("yolo_world_zs", "YOLO-World (zero-shot)", "#8c8c8c"),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

rows = list(csv.DictReader(open(BASE / "fig03_perfold.csv", encoding="utf-8")))
for r in rows:
    r["n_gt"] = int(r["n_gt"])
    for key, _, _ in ARMS:
        r[key] = float(r[key])
loeo = [r for r in rows if r["partition"] == "LOEO"]
loro = [r for r in rows if r["partition"] == "LORO"]
loeo_mean = {k: sum(r[k] for r in loeo) / len(loeo) for k, _, _ in ARMS}
loro_mean = {k: sum(r[k] for r in loro) / len(loro) for k, _, _ in ARMS}

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.0, 5.9))
width = 0.16

# ---------------- panel (a): leave-one-event-out -------------------------------------
x_a = np.arange(len(loeo))
for i, (key, label, color) in enumerate(ARMS):
    ax_a.bar(x_a + (i - 2) * width, [r[key] for r in loeo], width=width * 0.9,
             color=color, edgecolor="white", linewidth=0.35, label=label, zorder=3)
ax_a.set_xticks(x_a)
ax_a.set_xticklabels([r["fold"].replace(" (LOEO)", "") for r in loeo],
                     rotation=22, ha="right", rotation_mode="anchor")
ax_a.set_xlim(-0.6, len(loeo) - 0.4)
ax_a.set_ylim(0, 0.90)
ax_a.set_yticks(np.arange(0, 0.91, 0.2))
ax_a.set_ylabel("AR@900 (image-averaged)")
ax_a.set_title("(a)  Leave-one-event-out: every tested event held out", loc="left", pad=5)
ax_a.axhline(loeo_mean["class_agnostic"], color=ARMS[0][2], lw=0.9, ls="--",
             alpha=0.8, zorder=2)

# ---------------- panel (b): leave-one-region-out + the two pooled means ---------------
loro_cats = [r["fold"].replace(" (LORO)", "") for r in loro]
cats = loro_cats + ["LOEO\nmean", "LORO\nmean"]
x_b = np.arange(len(cats))
for i, (key, label, color) in enumerate(ARMS):
    vals = [r[key] for r in loro] + [loeo_mean[key], loro_mean[key]]
    ax_b.bar(x_b + (i - 2) * width, vals, width=width * 0.9, color=color,
             edgecolor="white", linewidth=0.35, zorder=3)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(cats, rotation=22, ha="right", rotation_mode="anchor")
ax_b.set_xlim(-0.6, len(cats) - 0.4)
ax_b.set_ylim(0, 0.90)
ax_b.set_yticks(np.arange(0, 0.91, 0.2))
ax_b.set_ylabel("AR@900 (image-averaged)")
ax_b.set_title("(b)  Leave-one-region-out: a whole region withheld, against the "
               "event-level mean", loc="left", pad=5)
for xsep in (len(loro) - 0.5, len(cats) - 1.5):
    ax_b.axvline(xsep, color="#bbbbbb", lw=0.6, ls=":", zorder=1)
ax_b.axhline(loro_mean["class_agnostic"], color=ARMS[0][2], lw=0.9, ls="--",
             alpha=0.8, zorder=2)

for ax in (ax_a, ax_b):
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.005), handlelength=1.3, columnspacing=1.6)
fig.subplots_adjust(left=0.125, right=0.985, top=0.955, bottom=0.175, hspace=0.42)

fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.14)
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.14)
print(f"panel a: LOEO mean proposer = {loeo_mean['class_agnostic']:.4f} over {len(loeo)} folds")
print(f"panel b: LORO mean proposer = {loro_mean['class_agnostic']:.4f} over {len(loro)} folds")
print("arm means @900:", {k: round(v, 4) for k, v in loeo_mean.items()})
print("wrote", OUT.with_suffix(".pdf").name)
