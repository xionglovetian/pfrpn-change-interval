#!/usr/bin/env python
"""Fig. 6 (v6): the event interval as a training-time commitment.

Bars are sorted by AR@900 so the figure reads top to bottom as the argument:
training on the correct interval raises recall above the clean anchor; every
way of changing the interval after training, or of training on the wrong one,
lowers it.  The dashed line marks the clean anchor.

Additions over v5: the three correct-interval arms, one of which carries the
three-seed standard deviation as an error bar, plus the two 6-channel arms that
separate the value of the interval from the value of a wider input (the
duplicated-stack control sits on the anchor line; the complementary stream is
the only arm above the retrained one).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
ORIGIN = BASE.parents[1] / "origin"
sys.path.insert(0, str(ORIGIN))
import _plotstyle as st  # noqa: E402

raw = json.loads((BASE / "source_fig6_v6.json").read_text(encoding="utf-8"))

FAMILY_COLORS = {
    "correct interval (retrained)": "#117733",
    "reference": st.COLORS["blue"],
    "change channels": st.COLORS["sky"],
    "permutation": st.COLORS["purple"],
    "permutation control": st.COLORS["green"],
    "no change": st.COLORS["gray"],
    "outside event window": st.COLORS["orange"],
    "correct interval (frozen)": st.COLORS["yellow"],
    "channel count control": "#8C510A",
    "complementary stream": "#B2182B",
}
FAMILY_LABELS = {
    "correct interval (retrained)": "trained on the correct interval",
    "reference": "deployed baseline",
    "change channels": "change channel zeroed at inference",
    "permutation": "change stack permuted within tile",
    "permutation control": "SAR / topography permuted (control)",
    "no change": "trained without change channels",
    "outside event window": "interval moved outside the event",
    "correct interval (frozen)": "correct interval given to a frozen model",
    "channel count control": "6 channels, both intervals long (control)",
    "complementary stream": "event window added as a second stream",
}

bars = sorted(raw["bars"], key=lambda b: b["ar900"])
labels = [b["name"] for b in bars]
vals = np.array([b["ar900"] for b in bars])
errs = np.array([b.get("err", 0.0) for b in bars])
fam = [b["family"] for b in bars]
colors = [FAMILY_COLORS[f] for f in fam]
y = np.arange(len(bars))

st.apply_rc(font_size=8.0)
fig, ax = plt.subplots(figsize=(6.6, 4.45), dpi=600)
ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.58,
        zorder=2, xerr=errs,
        error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=2.4, capthick=0.8))
for yi, v, e in zip(y, vals, errs):
    txt = "%.4f ± %.4f" % (v, e) if e > 0 else "%.4f" % v
    ax.text(v + e + 0.008, yi, txt, va="center", fontsize=7.0, color="#222222")

anchor = float(raw["anchor"])
ax.axvline(anchor, color=st.COLORS["blue"], linewidth=0.8, linestyle="--",
           alpha=0.55, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.0)
ax.set_xlim(0, 0.86)
st.style_axis(ax, xlabel="Average recall at 900 proposals", ylabel="", grid="x")

handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[k])
           for k in FAMILY_COLORS]
ax.legend(handles, [FAMILY_LABELS[k] for k in FAMILY_COLORS],
          loc="upper center", bbox_to_anchor=(0.5, -0.105), ncol=2,
          fontsize=6.0, handlelength=0.9, labelspacing=0.24,
          columnspacing=1.2, frameon=False)
st.save_figure(fig, BASE, "Fig6_interval_v6")
print("plotted %d bars; anchor = %.4f; top = %s %.4f" %
      (len(bars), anchor, labels[-1], vals[-1]))
