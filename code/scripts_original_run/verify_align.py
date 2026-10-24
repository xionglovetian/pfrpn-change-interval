import os, glob, json
import numpy as np
from src.training_data import resize_nearest
from src.mine_hard_negatives import box_iou_matrix, load_gt_from_manifest

DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
ROOT = "/root/autodl-tmp/m2_fix_check"
gt = load_gt_from_manifest(DATA + "/manifests/train.json", box_scale=1.0)

old_cov, new_cov, n = [], [], 0
for p in sorted(glob.glob(ROOT + "/train/itogon/*.proposals.json")):
    doc = json.load(open(p, encoding="utf-8")); iid = doc["image_id"]
    e = gt.get(iid, {})
    gb = np.array(e.get("gt_boxes", []), float).reshape(-1, 4)
    td = e.get("tile_dir")
    if td and not td.startswith("/"):
        td = os.path.normpath(os.path.join(DATA, td))
    if len(gb) == 0 or not td: continue
    mask = np.load(os.path.join(td, "mask.npy"))
    m320 = resize_nearest(mask, (320, 320)) > 0          # native 128 -> model 320
    b = np.array([c["bbox_xyxy"] for c in doc["candidates"]], float)
    iou = box_iou_matrix(b, gb).max(1)
    for i in np.where(iou >= 0.5)[0]:
        box_old = b[i]                                    # used as-is (buggy)
        box_new = b[i] * 2.5                              # back to model frame (fixed)
        ys, xs = np.where(m320)
        if len(ys) == 0: continue
        for box, acc in ((box_old, old_cov), (box_new, new_cov)):
            inside = ((xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3]))
            acc.append(inside.mean())
        n += 1
print("matched positives checked: %d" % n)
print("mean GT-mask coverage inside RoI:")
print("  OLD (buggy: native box used on model-frame features) = %.3f" % np.mean(old_cov))
print("  NEW (fixed: box scaled back to model frame)          = %.3f" % np.mean(new_cov))