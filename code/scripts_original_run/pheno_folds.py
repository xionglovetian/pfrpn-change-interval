#!/usr/bin/env python3
"""Cross-event / cross-region propagation of the change-signal specificity test.

Same computation as tools/baselines/pheno_test.py (data-only: reads change.npy + mask.npy,
no trained model, no GPU), but run per held-out fold for LOEO (7) and LORO (3), plus the
main event-level test split as the reference row.

Per split it reports:
  * median |change| INSIDE landslide mask vs OUTSIDE  -> "landslide vs background ratio"
  * AUC(inside landslide change, same-tile background change)
  * AUC(inside landslide change, negative-tile background change)
  * background share  = mean(neg tiles)/mean(inside)   -> "non-purity" (~55% on main test)
  * landslide share   = 1 - background share           (~45% on main test)
"""
import json, os, sys
import numpy as np

ROOT = "/root/autodl-tmp/PF-RPN"
DATA = os.path.join(ROOT, "data", "landslide_multimodal_hn_lb")
OUTDIR = "/root/autodl-tmp/baselines/analysis"
OUT = os.path.join(OUTDIR, "phenology_folds.json")

SPLITS = (
    [("main_test", "manifests/test.json")]
    + [(f"loeo_{f}", f"manifests/loeo_{f}.json") for f in
       ["italy", "kyrgyzstan1", "newzealand", "chimanimani", "hiroshima", "indonesia", "hokkaido"]]
    + [(f"loro_{f}", f"manifests/loro_{f}.json") for f in
       ["EastAsia", "CentralAsia", "SEAsia"]]
)


def auc(x, y):
    """Area under ROC; x = class-1 scores (landslide change), y = class-0 (background)."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    v = np.concatenate([x, y])
    r = np.empty(v.size)
    o = np.argsort(v, kind="mergesort")
    r[o] = np.arange(1, v.size + 1)
    u = r[:x.size].sum() - x.size * (x.size + 1) / 2.0
    return float(u / (x.size * y.size))


def analyse(manifest_rel):
    man = json.load(open(os.path.join(DATA, manifest_rel)))
    ins, outs, neg = [], [], []
    miss = 0
    for e in man:
        tile = os.path.dirname(os.path.join(DATA, e["modalities"]["change"]))
        a = np.load(os.path.join(DATA, e["modalities"]["change"])).astype(np.float32)
        m = np.abs(a).mean(axis=-1)
        has_gt = bool(e["labels"]["annotations"])
        mk_path = os.path.join(tile, "mask.npy")
        if not os.path.exists(mk_path):
            miss += 1
            if not has_gt:
                neg.append(float(m.mean()))
            continue
        mk = np.load(mk_path) > 0
        if has_gt and mk.any():
            ins.append(float(m[mk].mean()))
            bg = m[~mk]
            if bg.size:
                outs.append(float(bg.mean()))
        else:
            neg.append(float(m.mean()))
    ins = np.asarray(ins); outs = np.asarray(outs); neg = np.asarray(neg)
    res = {
        "n_pos_tiles": int(ins.size), "n_neg_tiles": int(neg.size), "missing_masks": miss,
        "inside_median": float(np.median(ins)) if ins.size else float("nan"),
        "outside_median": float(np.median(outs)) if outs.size else float("nan"),
        "inside_mean": float(ins.mean()) if ins.size else float("nan"),
        "outside_mean": float(outs.mean()) if outs.size else float("nan"),
        "neg_mean": float(neg.mean()) if neg.size else float("nan"),
        "ratio_median_inside_outside": float(np.median(ins) / max(np.median(outs), 1e-9))
            if ins.size and outs.size else float("nan"),
        "auc_inside_vs_tilebg": auc(ins, outs),
        "auc_inside_vs_negtiles": auc(ins, neg),
    }
    # non-purity: how much of the observed change at landslide locations is background
    if ins.size and neg.size and ins.mean() > 0:
        bg_share = float(neg.mean() / ins.mean())
        res["background_share"] = bg_share
        res["landslide_share"] = float(1.0 - bg_share)
    else:
        res["background_share"] = float("nan"); res["landslide_share"] = float("nan")
    return res


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    all_res = {}
    for name, rel in SPLITS:
        try:
            r = analyse(rel)
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}", flush=True)
            continue
        all_res[name] = r
        print(f"[{name}] pos={r['n_pos_tiles']} neg={r['n_neg_tiles']} "
              f"ratio={r['ratio_median_inside_outside']:.2f}x "
              f"AUC(bg)={r['auc_inside_vs_tilebg']:.3f} AUC(neg)={r['auc_inside_vs_negtiles']:.3f} "
              f"bg_share={r['background_share']:.3f}", flush=True)
    json.dump(all_res, open(OUT, "w"), indent=1)

    print("\n| split | ratio (median) | AUC vs tile-bg | AUC vs neg-tiles | background share |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    order = [n for n, _ in SPLITS if n in all_res]
    for n in order:
        r = all_res[n]
        print("| %s | %.2fx | %.3f | %.3f | %.1f%% |" % (
            n, r["ratio_median_inside_outside"], r["auc_inside_vs_tilebg"],
            r["auc_inside_vs_negtiles"], 100 * r["background_share"]), flush=True)
    print("\nwrote " + OUT, flush=True)


if __name__ == "__main__":
    main()

