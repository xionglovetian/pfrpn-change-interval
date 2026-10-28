#!/usr/bin/env python
"""Figure 2 (GNHR): pipeline and the two decisions studied. Redrawn at the printed size.

Canvas = 5.67 x 4.35 in = journal text width (408 pt) plus a little height, so every font
size below is the size that appears on the page. The previous version was drawn 13.33 in
wide and scaled to 42 %, which printed its labels at about 4.5 pt.

Design follows the reference template: dashed group panels, one rounded card per module with
the module name inside, vertical stacking inside a group, an orthogonal flow, a symbol legend
in its own band, and no in-figure title (the caption carries it).

Content is aligned with the manuscript:
  15-channel tile at 320x320 (3 optical + 4 SAR + 5 topography + 3 change)      Section 3.1
  Event-Bracketing Rule, 97.93 % exact over 3,135 tiles                          Section 3.2
  five discovery arms at a reported candidate count                              Section 3.3
  verification shared and held fixed                                             Section 3.1
  AR@k at IoU 0.5, k = 100/300/900, image-averaged (micro also reported)         Section 3.4
  three partitions: event-level 5/2/4, LOEO x7, LORO x3 -> ten folds             Section 3.5
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "fig02_framework"

BLUE_F, BLUE_E = "#E3EDF9", "#2F6DA8"
ORANGE_F, ORANGE_E = "#FCE7D6", "#C4703A"
GREY_F, GREY_E = "#F1F1F1", "#8C8C8C"
INK = "#222222"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

FIG_W, FIG_H = 5.67, 4.35
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def panel(x0, y0, x1, y1, title, tag=None, tag_color=INK):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                fc="none", ec="#9A9A9A", lw=0.7, ls=(0, (3, 2)), zorder=1))
    ax.text(x0 + 0.010, y1 - 0.016, title, ha="left", va="top", fontsize=6.5, color="#333333", zorder=5)
    if tag:
        ax.text(x0 + 0.010, y1 - 0.048, tag, ha="left", va="top", fontsize=5.6, color=tag_color, zorder=5)


def card(x0, y0, x1, y1, title, body=None, fc=GREY_F, ec=GREY_E, bold=False, fs=6.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.006,rounding_size=0.008",
                                fc=fc, ec=ec, lw=0.8, zorder=3))
    cy = (y0 + y1) / 2
    if body:
        ax.text((x0 + x1) / 2, cy + 0.014, title, ha="center", va="center", fontsize=fs,
                color=INK, fontweight="bold" if bold else "normal", zorder=4)
        ax.text((x0 + x1) / 2, cy - 0.020, body, ha="center", va="center", fontsize=fs - 0.4,
                color="#444444", zorder=4, linespacing=1.25)
    else:
        ax.text((x0 + x1) / 2, cy, title, ha="center", va="center", fontsize=fs,
                color=INK, fontweight="bold" if bold else "normal", zorder=4, linespacing=1.25)


def elbow(points, color="#555555", lw=0.8):
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=6.5,
                                 lw=lw, color=color, zorder=2, shrinkA=0, shrinkB=1.2))


# ------------------------------------------------------------------ group 1: input
panel(0.010, 0.545, 0.205, 0.985, "Input")
card(0.020, 0.858, 0.196, 0.932, "S2 pre-event", "3 ch")
card(0.020, 0.765, 0.196, 0.839, "S1 ascending", "4 ch: VV/VH, pre+post", fs=5.9)
card(0.020, 0.658, 0.196, 0.746, "DEM topography", "5 ch: elevation, slope,\naspect, curvature, wetness", fs=5.6)
ax.text(0.107, 0.572, "15 channels, 320 x 320 px", ha="center", va="center",
        fontsize=5.7, color="#555555", style="italic")

# ------------------------------------------------------ group 2: event window (fixed)
panel(0.222, 0.545, 0.434, 0.985, "Event window", tag="fixed by the rule", tag_color=BLUE_E)
card(0.232, 0.830, 0.424, 0.903, "tile acquisition dates", "+ event date", fc=BLUE_F, ec=BLUE_E, fs=5.9)
card(0.232, 0.720, 0.424, 0.793, "Event-Bracketing Rule", "last before / first after",
     fc=BLUE_F, ec=BLUE_E, bold=True, fs=5.9)
card(0.232, 0.610, 0.424, 0.683, "pre / post pair", "one adjacent pair", fc=BLUE_F, ec=BLUE_E, fs=5.9)
ax.text(0.328, 0.572, "97.93 % exact, 3,135 tiles", ha="center", va="center",
        fontsize=5.6, color=BLUE_E, style="italic")

# ----------------------------------------------------- group 3: change channels
panel(0.451, 0.545, 0.632, 0.985, "Change channels")
card(0.461, 0.775, 0.622, 0.900, "event change (3 ch)",
     "$\\Delta$NDVI, $\\Delta$NDWI, $\\Delta$NBR", fs=5.9)
ax.text(0.542, 0.735, "differenced inside\nthe event window;\nno label is used\nto build them",
        ha="center", va="top", fontsize=5.5, color="#555555", linespacing=1.45)

# -------------------------------------------------- group 4: discovery (compared)
panel(0.647, 0.385, 0.990, 0.985, "Candidate discovery", tag="compared across five arms",
      tag_color=ORANGE_E)
arms = [("class-agnostic proposer", "prompt-free, no class name"),
        ("YOLO-World, fine-tuned", "text-conditioned"),
        ("YOLOv8s", "closed-set, class labels"),
        ("Grounding DINO", "text-conditioned, zero-shot"),
        ("YOLO-World", "text-conditioned, zero-shot")]
for (name, note), y in zip(arms, [0.838, 0.750, 0.662, 0.574, 0.486]):
    card(0.657, y, 0.980, y + 0.066, name, note, fc=ORANGE_F, ec=ORANGE_E, fs=5.8)
ax.text(0.818, 0.408, "candidate count reported for every arm", ha="center", va="center",
        fontsize=5.6, color=ORANGE_E, style="italic")

# ----------------------------------------------- group 5: verification (held fixed)
panel(0.300, 0.115, 0.632, 0.385, "Verification", tag="held fixed", tag_color=GREY_E)
card(0.310, 0.160, 0.622, 0.290, "one shared verifier",
     "identical for every arm;\nno class information in the comparison", fs=6.0)

# ------------------------------------------------------------------ group 6: scoring
panel(0.647, 0.115, 0.990, 0.385, "Scoring")
card(0.657, 0.150, 0.980, 0.300, "AR@k at IoU 0.5",
     "k = 100 / 300 / 900 per image\nimage-averaged; micro also kept", fs=5.7)

# --------------------------------------------------------------- group 7: partitions
panel(0.010, 0.115, 0.288, 0.385, "Partitions")
ax.text(0.020, 0.340, "10 folds, no event shared", ha="left", va="top", fontsize=5.5,
        color="#555555", style="italic")
for i, (name, note) in enumerate([("event-level", "5 train / 2 val / 4 test"),
                                  ("leave-one-event-out", "7 folds"),
                                  ("leave-one-region-out", "3 folds")]):
    y = 0.262 - i * 0.064
    card(0.020, y, 0.278, y + 0.056, name, note, fs=5.6)

# --------------------------------------------------------------------- legend band
for i, (label, fc, ec) in enumerate([("fixed by the rule", BLUE_F, BLUE_E),
                                     ("compared across arms", ORANGE_F, ORANGE_E),
                                     ("shared and held fixed", GREY_F, GREY_E)]):
    x = 0.020 + i * 0.315
    ax.add_patch(FancyBboxPatch((x, 0.048), 0.026, 0.026,
                                boxstyle="round,pad=0.002,rounding_size=0.004",
                                fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.text(x + 0.034, 0.061, label, ha="left", va="center", fontsize=5.7, color="#333333")

# ---------------------------------------------------------------------------- flow
elbow([(0.198, 0.792), (0.228, 0.792)])                                        # input -> window
elbow([(0.424, 0.662), (0.457, 0.662)])                                        # pair -> change
elbow([(0.622, 0.845), (0.639, 0.845), (0.639, 0.870), (0.653, 0.870)])        # change -> arms
elbow([(0.700, 0.486), (0.700, 0.440), (0.470, 0.440), (0.470, 0.387)])        # arms -> verification
elbow([(0.622, 0.225), (0.653, 0.225)])                                        # verification -> scoring

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUT.with_suffix(".pdf"), pad_inches=0.02)
fig.savefig(OUT.with_suffix(".png"), dpi=300, pad_inches=0.02)
print(f"wrote {OUT.name}.pdf  ({FIG_W} x {FIG_H} in = printed size)")
