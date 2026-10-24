import os, glob
import numpy as np
p = glob.glob("/root/autodl-tmp/m2_roi_cache/A4_train_pos/train/itogon/*.roi.npz")[0]
d = np.load(p)
print("file:", os.path.basename(p), "dtype:", d["roi__optical"].dtype)
for k in d.files:
    v = d[k]
    if v.dtype.kind != "f":
        continue
    f = v.astype(np.float32)
    print("%-16s fp16[min=%8.1f max=%8.1f]  inf=%d nan=%d | fp32 mean=%9.3f std=%9.3f  p999=%.1f" % (
        k, float(v.min()), float(v.max()),
        int(np.isinf(f).sum()), int(np.isnan(f).sum()),
        float(f.mean()), float(f.std()), float(np.percentile(np.abs(f), 99.9))))
print()
print("fp16 representable max =", np.finfo(np.float16).max)