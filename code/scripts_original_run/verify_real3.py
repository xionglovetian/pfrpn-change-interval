import json, glob
import numpy as np
from src.mine_hard_negatives import box_iou_matrix, load_gt_from_manifest, DEFAULT_EXPORT_SCALE
DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
ROOT = "/root/autodl-tmp/m2_roi_cache/A4_train_pos"

allmax = []
for p in sorted(glob.glob(ROOT + "/train/itogon/*.proposals.json"))[:5]:
    doc = json.load(open(p, encoding="utf-8"))
    b = np.array([c["bbox_xyxy"] for c in doc["candidates"]], float)
    allmax.append(b.max())
    print("%-22s n=%d  bbox min=%.1f max=%.1f  (>128 count=%d)" % (
        doc["image_id"], len(b), b.min(), b.max(), int((b > 128).sum())))
print("overall max coord =", max(allmax))

# IoU of proposals vs GT under scale 1.0 vs 2.5
gt = load_gt_from_manifest(DATA + "/manifests/train.json", box_scale=1.0)
g25 = load_gt_from_manifest(DATA + "/manifests/train.json", box_scale=2.5)
best = {1.0: [], 2.5: []}
for p in sorted(glob.glob(ROOT + "/train/itogon/*.proposals.json"))[:20]:
    doc = json.load(open(p, encoding="utf-8")); iid = doc["image_id"]
    b = np.array([c["bbox_xyxy"] for c in doc["candidates"]], float)
    for s, g in ((1.0, gt), (2.5, g25)):
        gb = np.array(g.get(iid, {}).get("gt_boxes", []), float).reshape(-1, 4)
        if len(gb) and len(b):
            best[s].extend(box_iou_matrix(b, gb).max(1).tolist())
for s in (1.0, 2.5):
    a = np.array(best[s])
    print("GT_scale=%.1f  n=%d  maxIoU p50=%.3f p90=%.3f p99=%.3f  >=0.5 count=%d" % (
        s, len(a), np.percentile(a,50), np.percentile(a,90), np.percentile(a,99), int((a>=0.5).sum())))