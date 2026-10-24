import os, json, glob
import numpy as np
from src.mine_hard_negatives import DEFAULT_EXPORT_SCALE, box_iou_matrix, load_gt_from_manifest
from src.metrics import average_precision
DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
ROOT = "/root/autodl-tmp/m2_roi_cache/A4_train_pos"
gt = load_gt_from_manifest(DATA + "/manifests/train.json", box_scale=DEFAULT_EXPORT_SCALE)

def pool_ap(pred_labels, tag):
    sc, lb = [], []
    for p in glob.glob(ROOT + "/train/*/*.proposals.json"):
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("event_id") != "itogon":
            continue
        e = gt.get(doc["image_id"], {}); g = np.array(e.get("gt_boxes", []), float).reshape(-1, 4)
        boxes = np.array([c["bbox_xyxy"] for c in doc["candidates"]], float)
        obj = np.array([float(c["objectness"]) for c in doc["candidates"]])
        if len(boxes) == 0 or len(g) == 0:
            continue
        iou = box_iou_matrix(boxes, g).max(1)
        keep = pred_labels(iou)
        sc.extend(obj[keep]); lb.extend([1 if iou[i] >= 0.5 else 0 for i in np.where(keep)[0]])
    lb = np.array(lb)
    print("%-38s n=%5d pos=%4d chance=%.3f  M1_objectness_AP=%.4f" % (
        tag, len(lb), int(lb.sum()), lb.mean() if len(lb) else 0, average_precision(lb.tolist(), sc)))

# (a) my mining rule: positives IoU>=.5 (any score) + negatives IoU<.3 & score>=.4
def rule_a(iou):
    return (iou >= 0.5)
print("--- (a) positives only, my rule: any score ---")
pool_ap(rule_a, "positives(iou>=.5, any score)")

# (b) full pool, no score truncation: positives iou>=.5, negatives iou<.3
def rule_b(iou):
    return (iou >= 0.5) | (iou < 0.3)
print("--- (b) full pool, NO score truncation ---")
pool_ap(rule_b, "pos(iou>=.5)+neg(iou<.3), any score")