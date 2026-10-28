#!/usr/bin/env python
"""GNHR Fig. 5 (qualitative): what one candidate looks like, on one event.

Restores the qualitative panel that existed in the IEEE draft (origin/Fig 6) and was dropped during
the GNHR assembly. Nothing new is computed: the arrays and the two proposal files are the archived
outputs for tile italy_1161.

Reference object: the largest annotated component in the tile (42 px, about 10 x 6 px). The tile
also carries two small components (12 px and 7 px); they are not drawn, because scoring a whole-tile
mask bounding box would put the "hit" box around the image rather than around a landslide. The
caption states this convention.

Panels
  (a) pre-event optical image with the reference outline
  (b) event-change magnitude, same outline
  (c) the class-agnostic proposer: the one candidate at IoU >= 0.5 plus the next five by IoU
  (d) the closed-set YOLOv8s: all five of its candidates for this tile

Output: 02_图表/fig05_qualitative.pdf (+ .png)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "italy_1161"  # patched for the Zenodo package (was an absolute path)
OUT = BASE / "fig05_qualitative"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
BLUE, ORANGE, SKY, YELLOW, DARK, GREY = "#0b4f8a", "#d9820a", "#7fb3d5", "#f0c000", "#5a3a00", "#666666"


def pct_norm(image):
    lo, hi = np.percentile(image, [2, 98])
    if hi <= lo:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def largest_component(m):
    h, w = m.shape
    lab = np.zeros((h, w), dtype=int)
    best, cur = [], 0
    for i in range(h):
        for j in range(w):
            if m[i, j] and lab[i, j] == 0:
                cur += 1
                stack, px = [(i, j)], []
                lab[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    px.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
                if len(px) > len(best):
                    best = px
    out = np.zeros_like(m)
    for y, x in best:
        out[y, x] = True
    return out, len(best)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if (aa + bb - inter) else 0.0


optical = pct_norm(np.load(SRC / "optical_pre.npy").astype(float))
change = pct_norm(np.sqrt(np.sum(np.load(SRC / "change.npy").astype(float) ** 2, axis=2)))
raw_mask = np.load(SRC / "mask.npy").astype(bool)
mask, n_px = largest_component(raw_mask)
ys, xs = np.where(mask)
gt = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

pf = json.loads((SRC / "PF_RPN.proposals.json").read_text(encoding="utf-8"))["candidates"]
y8 = json.loads((SRC / "YOLOv8s.proposals.json").read_text(encoding="utf-8"))["candidates"]
pf_scored = sorted(((iou(c["bbox_xyxy"], gt), c) for c in pf),
                   key=lambda t: (-t[0], -float(t[1].get("proposal_score", 0))))
y8_scored = sorted(((iou(c["bbox_xyxy"], gt), c) for c in y8), key=lambda t: (-t[0],))
pf_iou, pf_best = pf_scored[0]
y8_iou = y8_scored[0][0] if y8_scored else 0.0
pf_hits = sum(1 for i, _ in pf_scored if i >= 0.5)

fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.3))
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def label(ax, panel, title):
    ax.text(0.015, 1.015, f"({panel}) {title}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.8, color="#111111")


def note(ax, text, color):
    ax.text(0.03, 0.05, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
            color=color, bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.4))


axes[0, 0].imshow(optical)
axes[0, 0].contour(mask, levels=[0.5], colors=[YELLOW], linewidths=1.3)
label(axes[0, 0], "a", "Sentinel-2 pre-event, with the reference outline")
note(axes[0, 0], "reference object: 42 px", GREY)

im = axes[0, 1].imshow(change, cmap="Blues", vmin=0, vmax=1)
axes[0, 1].contour(mask, levels=[0.5], colors=[YELLOW], linewidths=1.3)
label(axes[0, 1], "b", "Event-change magnitude")
note(axes[0, 1], "normalised $|\\Delta|$, 0--1", GREY)

axes[1, 0].imshow(optical)
axes[1, 0].contour(mask, levels=[0.5], colors=[YELLOW], linewidths=1.3)
label(axes[1, 0], "c", "Class-agnostic proposer")
for _, cand in pf_scored[:6]:
    x1, y1, x2, y2 = cand["bbox_xyxy"]
    axes[1, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=SKY, linewidth=0.8, alpha=0.9))
x1, y1, x2, y2 = pf_best["bbox_xyxy"]
axes[1, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=BLUE, linewidth=1.9))
note(axes[1, 0], f"best IoU {pf_iou:.2f};  {pf_hits} of {len(pf)} candidates reach IoU 0.5", BLUE)

axes[1, 1].imshow(optical)
axes[1, 1].contour(mask, levels=[0.5], colors=[YELLOW], linewidths=1.3)
label(axes[1, 1], "d", "YOLOv8s (closed-set)")
for cand in y8:
    x1, y1, x2, y2 = cand["bbox_xyxy"]
    axes[1, 1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=ORANGE, linewidth=0.9, alpha=0.95))
x1, y1, x2, y2 = y8_scored[0][1]["bbox_xyxy"]
axes[1, 1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=DARK, linewidth=1.6))
note(axes[1, 1], f"best IoU {y8_iou:.2f};  all {len(y8)} candidates", DARK)

fig.tight_layout(pad=0.45, w_pad=0.25, h_pad=0.55)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.12)
print(f"reference object {n_px} px, box={[round(v,1) for v in gt]}")
print(f"PF best IoU {pf_iou:.4f}, hits {pf_hits}/{len(pf)}; YOLOv8s best IoU {y8_iou:.4f} of {len(y8)}")
print("wrote", OUT.with_suffix(".pdf").name)
