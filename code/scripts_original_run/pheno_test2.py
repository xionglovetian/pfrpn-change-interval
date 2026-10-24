"""Is the change signal landslide-specific, or is it background (phenological)?

Data-only test on the test split:
  positives: |change| INSIDE the landslide mask  vs  OUTSIDE it
  negatives: whole-tile |change|
If inside >> outside, the landslide change dominates the seasonal background and
the model has a separable signal.  If inside ~ outside, the change channel is
dominated by non-landslide (e.g. seasonal) variation.
"""
import json, os
import numpy as np
ROOT = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
man = json.load(open(os.path.join(ROOT, "manifests/test.json")))

ins, outs, neg, miss = [], [], [], 0
for e in man:
    tile = os.path.dirname(os.path.join(ROOT, e["modalities"]["change"]))
    mp = os.path.join(tile, "mask.npy")
    a = np.load(os.path.join(ROOT, e["modalities"]["change"])).astype(np.float32)
    m = np.abs(a).mean(axis=-1)
    has_gt = bool(e["labels"]["annotations"])
    if not os.path.exists(mp):
        miss += 1
        if not has_gt: neg.append(float(m.mean()))
        continue
    mk = np.load(mp) > 0
    if has_gt and mk.any():
        ins.append(float(m[mk].mean()))
        bg = m[~mk]
        if bg.size: outs.append(float(bg.mean()))
    else:
        neg.append(float(m.mean()))

ins, outs, neg = np.array(ins), np.array(outs), np.array(neg)
print("positive tiles with GT mask: %d | negative tiles: %d | missing masks: %d" % (len(ins), len(neg), miss))
print()
print("=== |change| : INSIDE landslide vs OUTSIDE (positives) ===")
for q in (10, 25, 50, 75, 90):
    print("  %2d%%: inside=%.4f  outside=%.4f  ratio=%.2fx" % (
        q, np.percentile(ins, q), np.percentile(outs, q), np.percentile(ins, q) / max(np.percentile(outs, q), 1e-9)))
print("  mean: inside=%.4f  outside=%.4f  ratio=%.2fx" % (ins.mean(), outs.mean(), ins.mean()/outs.mean()))
print()
print("=== background |change| : positive-outside vs negative-tiles ===")
print("  positive background: mean=%.4f median=%.4f" % (outs.mean(), np.median(outs)))
print("  negative tiles     : mean=%.4f median=%.4f" % (neg.mean(), np.median(neg)))
print()
# 可分性：滑坡处变化 vs 背景变化
from numpy import argsort
def auc(x, y):
    # x = score for class1 (landslide change), y = class0 (background)
    v = np.concatenate([x, y]); r = np.empty(len(v)); o = argsort(v, kind="mergesort"); r[o] = np.arange(1, len(v)+1)
    u = r[:len(x)].sum() - len(x)*(len(x)+1)/2
    return u / (len(x)*len(y))
print("=== separability (AUC): inside-landslide change vs background change ===")
print("  AUC = %.3f  (0.5 = indistinguishable, 1.0 = perfectly separable)" % auc(ins, outs))
print("  AUC = %.3f  vs negative-tile background" % auc(ins, neg))
frac = float((ins > np.percentile(outs, 90)).mean())
print("  fraction of landslides whose inside-change exceeds the 90th pct of background: %.1f%%" % (100*frac))
