import os, json, glob
import numpy as np
from src.mine_hard_negatives import box_iou_matrix, load_gt_from_manifest

DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
V2 = "/root/autodl-tmp/m2_roi_cache_v2"
SETS = [("A4_hokkaido","val","event_hokkaido.json","val.json"),
        ("A4_newzealand","val","event_newzealand.json","val.json"),
        ("A4_train_pos","train","train_pos.json","train.json")]

print("== M1 候选效用：每个 GT 是否被至少一个候选以 IoU>=0.5 框住 ==")
print("%-16s %7s %7s %8s %8s %8s %10s" % ("set","#GT","covered","oracleR","AR@100","AR@300","AR@900"))
for name, split, man, gtman in SETS:
    root = f"{V2}/{name}/{split}"
    gt = load_gt_from_manifest(f"{DATA}/manifests/{gtman}", box_scale=1.0)
    gt_cov = gt_tot = 0
    per_img = []
    for p in sorted(glob.glob(root + "/*/*.proposals.json")):
        doc = json.load(open(p, encoding="utf-8")); iid = doc["image_id"]
        gb = np.array(gt.get(iid, {}).get("gt_boxes", []), float).reshape(-1,4)
        b = np.array([c["bbox_xyxy"] for c in doc["candidates"]], float).reshape(-1,4)
        obj = np.array([float(c["objectness"]) for c in doc["candidates"]])
        if len(gb) == 0: continue
        gt_tot += len(gb)
        if len(b):
            iou = box_iou_matrix(b, gb)          # (N, M)
            gt_cov += int((iou.max(0) >= 0.5).sum())
            mx = iou.max(1); per_img.append((obj, mx, len(gb)))
        else:
            per_img.append((obj, np.zeros(0), len(gb)))
    def ark(k):
        rs=[]
        for s, mx, n in per_img:
            if n<=0 or s.size==0: continue
            o=np.argsort(-s)[:k]; rs.append(min(int((mx[o]>=0.5).sum()), n)/n)
        return float(np.mean(rs)) if rs else float("nan")
    print("%-16s %7d %7d %8.3f %8.3f %8.3f %8.3f" % (
        name, gt_tot, gt_cov, gt_cov/max(gt_tot,1), ark(100), ark(300), ark(900)))