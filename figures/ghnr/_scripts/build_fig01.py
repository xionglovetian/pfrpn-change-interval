"""Build Fig. 1 for the GNHR version.

fig01a : 11-panel of one representative tile per event (optical + annotated landslide outline)
fig01b : global distribution map with Natural Earth coastlines
also writes event_geo.csv (map-ready table)

Run:  python build_fig01.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shp_lite import read_shp_parts  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

EVENTS_ROOT = os.environ.get("PFRPN_EVENTS_ROOT",
    r"D:\PFRPN论文相关材料\10_数据与实验产物\dataset\landslide_multimodal_lb_pos\events")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # figures/ghnr
OUT = BASE
COAST = os.path.join(OUT, "_geo", "ne_110m_coastline.shp")

# event -> (label, lon, lat, status, split, country, trigger, label_dxy)
GEO = {
    "italy":       ("Emilia-Romagna, Italy",      11.85,  44.35, "verified", "train", "Italy",             "rainfall / flood (May 2023)",    (6, 5)),
    "chimanimani": ("Chimanimani, Zimbabwe",      32.87, -19.80, "verified", "test",  "Zimbabwe",          "Cyclone Idai (Mar 2019)",        (6, 5)),
    "hiroshima":   ("Hiroshima, Japan",          132.80,  34.60, "verified", "train", "Japan",             "heavy rainfall (Jun 2018)",      (6, -12)),
    "hokkaido":    ("Iburi, Hokkaido, Japan",    141.95,  42.65, "verified", "val",   "Japan",             "Mj 6.7 earthquake (Sep 2018)",   (6, 5)),
    "itogon":      ("Itogon, Philippines",       120.68,  16.42, "verified", "train", "Philippines",       "Typhoon Mangkhut (Sep 2018)",    (6, 5)),
    "china":       ("China (multi-event)",       105.00,  35.00, "country",  "test",  "China",             "multiple",                       (6, -14)),
    "indonesia":   ("Indonesia (multi-event)",   118.00,  -2.50, "country",  "test",  "Indonesia",         "multiple",                       (6, 5)),
    "kyrgyzstan1": ("Kyrgyzstan (multi-event)",   74.60,  41.60, "country",  "train", "Kyrgyzstan",        "multiple",                       (6, 5)),
    "kyrgyzstan2": ("Kyrgyzstan (multi-event)",   74.60,  40.10, "country",  "train", "Kyrgyzstan",        "multiple",                       (6, -12)),
    "newzealand":  ("New Zealand (multi-event)", 172.80, -41.50, "country",  "val",   "New Zealand",       "multiple",                       (-92, 6)),
    "usa":         ("Puerto Rico, USA",          -66.70,  18.00, "verified", "test",  "USA (Puerto Rico)", "earthquake sequence (Jan 2020)", (6, 5)),
}
ORDER = ["italy", "chimanimani", "hiroshima", "hokkaido", "itogon",
         "china", "indonesia", "kyrgyzstan1", "kyrgyzstan2", "newzealand", "usa"]

SPLIT_COLOR = {"train": "#4C72B0", "val": "#DD8452", "test": "#C44E52"}
SPLIT_LABEL = {"train": "training", "val": "validation", "test": "held-out test"}


def stretch(rgb, lo=2, hi=98):
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[2]):
        ch = rgb[..., c]
        a, b = np.percentile(ch, [lo, hi])
        out[..., c] = np.clip((ch - a) / max(b - a, 1e-6), 0, 1)
    return out


def pick_tile(ev):
    p = os.path.join(EVENTS_ROOT, ev, "tiles")
    best, best_area = None, 0.0
    for t in sorted(os.listdir(p)):
        m = os.path.join(p, t, "mask.npy")
        if not os.path.exists(m):
            continue
        frac = float((np.load(m) > 0).mean())
        if 0.02 <= frac <= 0.40 and frac > best_area:
            best, best_area = t, frac
    if best is None:
        for t in sorted(os.listdir(p)):
            m = os.path.join(p, t, "mask.npy")
            if os.path.exists(m) and (np.load(m) > 0).any():
                return t
        return sorted(os.listdir(p))[0]
    return best


def fig_panel():
    fig, axes = plt.subplots(3, 4, figsize=(13.33, 8.4))
    axes = axes.ravel()
    for k, ev in enumerate(ORDER):
        ax = axes[k]
        tile = pick_tile(ev)
        d = os.path.join(EVENTS_ROOT, ev, "tiles", tile)
        rgb = np.load(os.path.join(d, "optical_pre.npy"))
        mk = np.load(os.path.join(d, "mask.npy"))
        ax.imshow(stretch(rgb))
        if (mk > 0).any():
            ax.contour(mk, levels=[0.5], colors=["#FF2D2D"], linewidths=0.9)
        ax.set_title(f"{GEO[ev][0]}\n{GEO[ev][6]}", fontsize=8.2, linespacing=1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.03, tile, transform=ax.transAxes, fontsize=6.2, color="w",
                bbox=dict(fc="k", alpha=0.45, lw=0))
    for j in range(len(ORDER), len(axes)):
        axes[j].axis("off")
    fig.legend(handles=[Line2D([], [], color="#FF2D2D", lw=1.2, label="annotated landslide outline")],
               loc="lower center", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig01a_event_panel.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig01a_event_panel.{pdf,png}")


def fig_map():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    note = ""
    if os.path.exists(COAST):
        for xs, ys in read_shp_parts(COAST):
            ax.plot(xs, ys, color="0.45", lw=0.5, zorder=1)
    else:
        note = " (coastline file missing)"
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-60, 81, 30))
    ax.grid(color="0.9", lw=0.5, zorder=0)
    ax.set_facecolor("#F7F9FB")
    for ev in ORDER:
        label, lon, lat, status, split, country, trig, dxy = GEO[ev]
        c = SPLIT_COLOR[split]
        if status == "verified":
            ax.scatter(lon, lat, s=62, c=c, edgecolor="k", linewidth=0.6, zorder=5)
        else:
            ax.scatter(lon, lat, s=80, facecolor="none", edgecolor=c, linewidth=1.7,
                       marker="s", zorder=4)
        disp = {"usa": "usa (Puerto Rico)"}.get(ev, ev)
        ax.annotate(disp, (lon, lat), textcoords="offset points", xytext=dxy,
                    fontsize=8, color="0.12", zorder=6)
    handles = [Line2D([], [], marker="o", ls="", mfc=SPLIT_COLOR[k], mec="k", ms=7,
                      label=SPLIT_LABEL[k]) for k in ("train", "val", "test")]
    handles += [Line2D([], [], marker="s", ls="", mfc="none", mec="0.35", ms=8,
                       label="multi-event inventory (country level)")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True, framealpha=0.95)
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.set_title(f"Geographic distribution of the eleven events{note}", fontsize=11)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig01b_events_map.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig01b_events_map.{pdf,png}")


def write_csv():
    p = os.path.join(OUT, "event_geo.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event", "location_label", "lon", "lat", "location_status", "split", "country", "trigger"])
        for ev in ORDER:
            label, lon, lat, status, split, country, trig, _ = GEO[ev]
            w.writerow([ev, label, lon, lat, status, split, country, trig])
    print("wrote event_geo.csv")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write_csv()
    fig_panel()
    fig_map()

