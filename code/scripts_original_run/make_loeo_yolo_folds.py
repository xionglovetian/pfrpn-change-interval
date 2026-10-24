"""Prepare per-fold YOLO train lists + data yaml for leave-one-event-out."""
import json, os
M = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests"
DS = "/root/autodl-tmp/yolo_ds/all"
FOLDS = ["italy", "kyrgyzstan1", "newzealand", "chimanimani", "hiroshima", "indonesia", "hokkaido"]
pooled = json.load(open(os.path.join(M, "loeo_all.json")))
img_event = {str(e["image_id"]): str(e["event_id"]) for e in pooled}
for f in FOLDS:
    keep = [i for i, ev in img_event.items() if ev != f]
    lst = os.path.join(DS, "train_%s.txt" % f)
    with open(lst, "w") as fh:
        for i in keep:
            p = os.path.join(DS, "images", i + ".png")
            if os.path.exists(p):
                fh.write(p + "\n")
    with open(os.path.join(DS, "data_%s.yaml" % f), "w") as fh:
        fh.write("path: %s\ntrain: %s\nval: %s\nnames:\n  0: landslide\n" % (DS, lst, lst))
    print("fold %-14s train images = %d" % (f, len(keep)))
