import os, json, glob, random
import numpy as np
from src.mine_hard_negatives import box_iou_matrix, load_gt_from_manifest

DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
V2 = "/root/autodl-tmp/m2_roi_cache_v2"

def check(name, split, man, gtman):
    root = f"{V2}/{name}/{split}"
    exp = json.load(open(man, encoding="utf-8"))
    files = {os.path.basename(p)[:-len(".roi.npz")]: p for p in glob.glob(root + "/*/*.roi.npz")}
    print("== %s (%s): expect %d, have %d" % (name, split, len(exp), len(files)))
    per_ev_exp, per_ev_act = {}, {}
    for e in exp: per_ev_exp[e["event_id"]] = per_ev_exp.get(e["event_id"], 0) + 1
    for p in files.values():
        ev = os.path.basename(os.path.dirname(p)); per_ev_act[ev] = per_ev_act.get(ev, 0) + 1
    miss = []
    for ev in sorted(per_ev_exp):
        a, b = per_ev_exp[ev], per_ev_act.get(ev, 0)
        flag = "" if a == b else "  <-- MISSING %d" % (a - b)
        if a != b: miss.append(ev)
        print("   %-16s %5d / %5d%s" % (ev, b, a, flag))
    # GT frame check: which scale matches?
    g1 = load_gt_from_manifest(gtman, box_scale=1.0)
    g25 = load_gt_from_manifest(gtman, box_scale=2.5)
    random.seed(0); sample = random.sample(sorted(files), min(40, len(files)))
    m1 = m25 = npos1 = 0
    for iid in sample:
        doc = json.load(open(files[iid][:-len(".roi.npz")] + ".proposals.json", encoding="utf-8"))
        b = np.array([c["bbox_xyxy"] for c in doc["candidates"]], float).reshape(-1,4)
        if not len(b): continue
        for g, key in ((g1, 1), (g25, 25)):
            gb = np.array(g.get(iid, {}).get("gt_boxes", []), float).reshape(-1,4)
            if len(gb):
                mx = box_iou_matrix(b, gb).max(1)
                if key == 1: m1 += int((mx >= 0.5).sum()); npos1 += len(gb)
                else: m25 += int((mx >= 0.5).sum())
    print("   GT match (40 imgs): scale1.0 -> %d hits | scale2.5 -> %d hits  => %s" % (
        m1, m25, "original frame OK" if m1 > m25 else "!! scale looks wrong"))
    return not miss

a = check("A4_hokkaido", "val", f"{DATA}/manifests/event_hokkaido.json", f"{DATA}/manifests/val.json")
b = check("A4_newzealand", "val", f"{DATA}/manifests/event_newzealand.json", f"{DATA}/manifests/val.json")
print()
print("B2 partial (completed splits):", "PASS" if (a and b) else "FAIL")