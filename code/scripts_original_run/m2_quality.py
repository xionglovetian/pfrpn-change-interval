import os, json, glob
import numpy as np
from src.mine_hard_negatives import box_iou_matrix, load_gt_from_manifest
from src.evaluate_cascade import ar_at_k, oracle_recall, localization_quality
DATA="/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"; V2="/root/autodl-tmp/m2_roi_cache_v2"
SETS=[("A4_hokkaido","val","val.json"),("A4_newzealand","val","val.json"),("A4_train_pos","train","train.json")]
print("%-16s %7s | %7s %8s | %7s %7s %7s %7s | %8s" % ("set","#GT","cover","oracleR","AR@100","AR@300","IoU>=.5","IoU>=.75","meanIoU"))
for name,split,gtman in SETS:
    root=f"{V2}/{name}/{split}"; gt=load_gt_from_manifest(f"{DATA}/manifests/{gtman}", box_scale=1.0)
    per=[]; tot=0; cov=0
    for p in sorted(glob.glob(root+"/*/*.proposals.json")):
        doc=json.load(open(p,encoding="utf-8")); iid=doc["image_id"]
        gb=np.array(gt.get(iid,{}).get("gt_boxes",[]),float).reshape(-1,4)
        if len(gb)==0: continue
        b=np.array([c["bbox_xyxy"] for c in doc["candidates"]],float).reshape(-1,4)
        obj=np.array([float(c["objectness"]) for c in doc["candidates"]])
        iou=box_iou_matrix(b,gb) if len(b) else np.zeros((0,len(gb)))
        tot+=len(gb); cov+=int((iou.max(0)>=0.5).sum()) if iou.size else 0
        per.append((obj,iou,len(gb)))
    loc=localization_quality(per)
    print("%-16s %7d | %7d %8.3f | %7.3f %7.3f %7.3f %7.3f | %8.3f" % (
        name,tot,cov,oracle_recall(per),ar_at_k(per,100),ar_at_k(per,300),
        loc["frac_best_iou_ge_0.50"],loc["frac_best_iou_ge_0.75"],loc["mean_best_iou"]))