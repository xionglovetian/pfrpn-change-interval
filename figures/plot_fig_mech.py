"""Mechanism figure: the interval assumption and its removal.

Two side-by-side timelines over the same tile, whose 15 acquisitions are the
only x axis.  Panel (a) is the conventional pipeline, which differences the
first and last acquisition during training and again at inference; panel (b)
is the scheme of this paper, which trains on the event-anchored window and
deploys the long baseline unchanged.

Spans are the ones the manuscript reports elsewhere: the long baseline is
678 d in the mean (360 d in the median of the pixel-level subset), the
annotated pair is 10 d in the median, and the pair widened by one acquisition
on each side is 25 d in the median.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.4,
    "savefig.dpi": 400,
})

GREY = "#6f6f6f"
GREEN = "#2f7d4f"
BAND = "#e2f0e6"
EVENT = "#b4351f"
INK = "#1d1d1d"

N = 15                 # acquisitions per tile
PAIR = (8, 9)          # annotated window: one adjacent pair, median position 0.50
WIDEN = (7, 10)        # pair widened by one acquisition on each side
LONG = (1, N)

Y_TRAIN, Y_DEPLOY, Y_SERIES = 3.05, 1.55, 0.78


def draw_window(ax, x0, x1, y, color, thick=6.5):
    ax.plot([x0, x1], [y, y], color=color, lw=thick, solid_capstyle="butt",
            zorder=4)
    for x in (x0, x1):
        ax.plot([x, x], [y - 0.17, y + 0.17], color=color, lw=1.3, zorder=4)


def draw_series(ax):
    ax.fill_between([PAIR[0] - 0.5, PAIR[1] + 0.5], 0.95, 3.95,
                    color=BAND, zorder=0, lw=0)
    ax.plot([1, N], [Y_SERIES, Y_SERIES], color="#a2a2a2", lw=0.7, zorder=1)
    for i in range(1, N + 1):
        ax.plot([i, i], [Y_SERIES - 0.09, Y_SERIES + 0.09], color="#3f3f3f",
                lw=0.85, zorder=2)
        ax.text(i, Y_SERIES - 0.16, str(i), ha="center", va="top",
                color="#4a4a4a", fontsize=5.6)
    ax.text((1 + N) / 2.0, 0.30, "acquisition index within the tile "
            "(15 steps per tile; the same tile in both panels)",
            ha="center", va="bottom", color="#4a4a4a", fontsize=6.1,
            style="italic")
    # the annotated event, as a calendar fact
    ax.plot([sum(PAIR) / 2.0] * 2, [Y_DEPLOY - 0.28, 4.02], color=EVENT,
            lw=0.9, ls=(0, (2.2, 1.7)), zorder=3)
    ax.text(sum(PAIR) / 2.0, 4.08, "event date (inventory)", ha="center",
            va="bottom", color=EVENT, fontsize=6.3)


def frame(ax, title):
    ax.set_xlim(-1.15, 16.85)
    ax.set_ylim(0.20, 4.32)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=7.9, color=INK, pad=3.0, fontweight="bold")


def rows(ax, train_color, deploy_color):
    ax.text(-0.35, Y_TRAIN, "train", ha="right", va="center", color=train_color,
            fontsize=7.0, fontweight="bold")
    ax.text(-0.35, Y_DEPLOY, "deploy", ha="right", va="center",
            color=deploy_color, fontsize=7.0, fontweight="bold")


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.05, 2.42))
fig.subplots_adjust(left=0.055, right=0.995, top=0.905, bottom=0.03, wspace=0.13)

# ---------------------------------------------------------------- panel (a)
frame(axA, "(a)  Conventional: one interval for both phases")
draw_series(axA)
rows(axA, GREY, GREY)
draw_window(axA, *LONG, Y_TRAIN, GREY)
draw_window(axA, *LONG, Y_DEPLOY, GREY)
axA.text(4.4, Y_TRAIN + 0.30, "678 d  (mean span)", ha="center", va="bottom",
         color=GREY, fontsize=7.0, fontweight="bold")
axA.text(4.4, Y_DEPLOY + 0.30, "678 d  (mean span)", ha="center", va="bottom",
         color=GREY, fontsize=7.0, fontweight="bold")
axA.annotate("", xy=(15.85, Y_TRAIN - 0.10), xytext=(15.85, Y_DEPLOY + 0.10),
             arrowprops=dict(arrowstyle="<->", lw=0.8, color=GREY,
                             shrinkA=0, shrinkB=0))
axA.text(16.35, (Y_TRAIN + Y_DEPLOY) / 2.0, "same window in both phases",
         rotation=90, ha="center", va="center", color=GREY, fontsize=6.3)
axA.annotate("the annotated pair is\nnever differenced",
             xy=(9.62, 2.42), xytext=(11.4, 2.34), ha="left", va="center",
             fontsize=6.6, color=EVENT, linespacing=1.2,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=EVENT,
                             connectionstyle="arc3,rad=0.0"))

# ---------------------------------------------------------------- panel (b)
frame(axB, "(b)  This work: the interval is chosen per phase")
draw_series(axB)
rows(axB, GREEN, GREY)
draw_window(axB, *WIDEN, Y_TRAIN, GREEN)
draw_window(axB, *LONG, Y_DEPLOY, GREY)
axB.text(10.75, Y_TRAIN, "25 d  (median span)", ha="left", va="center",
         color=GREEN, fontsize=7.0, fontweight="bold")
axB.text(4.6, Y_DEPLOY + 0.30, "678 d  (mean span), unchanged at inference",
         ha="center", va="bottom", color=GREY, fontsize=6.9)
axB.annotate("event-anchored window from\n"
             "the tile's own calendar\n(97.9% exact, no pixel labels)",
             xy=(6.88, Y_TRAIN), xytext=(0.55, 3.48), ha="left",
             va="center", fontsize=6.5, color=GREEN, linespacing=1.22,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=GREEN,
                             connectionstyle="arc3,rad=-0.16"))

out = r"C:\Users\ASUS\Desktop\PFRPN\01_论文\figs_v6\Fig_mech_v6"
fig.savefig(out + ".png", facecolor="white")
fig.savefig(out + ".pdf", facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote Fig_mech_v6.png/.pdf/.svg")
