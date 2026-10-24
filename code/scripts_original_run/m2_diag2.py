import os, json, glob
import numpy as np
from src.mine_hard_negatives import DEFAULT_EXPORT_SCALE, MiningConfig, load_gt_from_manifest, mine_proposal_dir, box_iou_matrix
from src.metrics import average_precision
DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
ROOT = "/root/autodl-tmp/m2_roi_cache/A4_train_pos"
gt = load_gt_from_manifest(DATA + "/manifests/train.json", box_scale=DEFAULT_EXPORT_SCALE)
lab = mine_proposal_dir(ROOT, "/tmp/x", gt, split="train",
                        config=MiningConfig(score_thr=0.40), write=False)["labels"]

scores, labels, objs = [], [], []
for p in glob.glob(ROOT + "/train/*/*.proposals.json"):
    doc = json.load(open(p, encoding="utf-8"))
    if doc.get("event_id") != "itogon":
        continue
    for c in doc["candidates"]:
        cid = str(c["candidate_id"])
        if cid in lab:
            labels.append(int(lab[cid])); scores.append(float(c["objectness"]))
print("itogon labelled dev: n=%d pos=%d neg=%d" % (len(labels), sum(labels), len(labels)-sum(labels)))
print("  dev chance (pos rate) = %.4f" % (np.mean(labels)))
print("  M1 objectness  dev_AP = %.4f" % average_precision(labels, scores))
pos = [s for s,l in zip(scores,labels) if l==1]; neg=[s for s,l in zip(scores,labels) if l==0]
print("  objectness pos: mean %.3f  p50 %.3f | neg: mean %.3f  p50 %.3f" % (
    np.mean(pos), np.median(pos), np.mean(neg), np.median(neg)))