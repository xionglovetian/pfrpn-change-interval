import os, json, glob
import numpy as np
DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
# 1) GT masks: real annotations?
masks = glob.glob(DATA + "/events/*/tiles/*/mask.npy") + glob.glob(DATA + "/../landslide_multimodal_lb_pos/events/*/tiles/*/mask.npy")
print("mask files found:", len(masks))
for p in masks[:5]:
    m = np.load(p)
    print("  %-70s shape=%s dtype=%s nonzero=%d max=%s" % (p.replace(DATA,''), m.shape, m.dtype, int((m>0).sum()), m.max()))
# 2) modal npy sanity
for name in ("optical_pre","change","topography"):
    fs = glob.glob(DATA + "/events/hokkaido/tiles/*/" + name + ".npy") + glob.glob(DATA + "/../landslide_multimodal_lb_pos/events/hokkaido/tiles/*/" + name + ".npy")
    if fs:
        a = np.load(fs[0])
        print("  %-14s shape=%s dtype=%s mean=%.3f std=%.3f" % (name, a.shape, a.dtype, float(a.mean()), float(a.std())))
# 3) RoI features real (not zeros/nan)?
import numpy as np
r = np.load("/root/autodl-tmp/m2_roi_cache/A4_train_pos/train/itogon/" + os.path.basename(glob.glob("/root/autodl-tmp/m2_roi_cache/A4_train_pos/train/itogon/*.roi.npz")[0]))
for k in r.files:
    v = r[k]
    if v.dtype.kind in "f":
        print("  %-16s shape=%s mean=%.4f std=%.4f nan=%d" % (k, v.shape, float(v.mean()), float(v.std()), int(np.isnan(v).sum())))