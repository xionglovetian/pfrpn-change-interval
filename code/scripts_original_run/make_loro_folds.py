"""Build leave-one-region-out fold manifests + per-fold YOLO lists."""
import json, os
M = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb/manifests"
DS = "/root/autodl-tmp/yolo_ds/all"
FOLDS = {
    "EastAsia":    ["china", "hiroshima", "hokkaido"],
    "CentralAsia": ["kyrgyzstan1", "kyrgyzstan2"],
    "SEAsia":      ["indonesia", "itogon"],
}
pooled = json.load(open(os.path.join(M, "loeo_all.json")))
os.makedirs(os.path.join(M, "loro"), exist_ok=True)
for name, evs in FOLDS.items():
    tr = [e for e in pooled if str(e["event_id"]) not in evs]
    te = [e for e in pooled if str(e["event_id"]) in evs]
    json.dump(tr, open(os.path.join(M, "loro", "%s_train.json" % name), "w"), ensure_ascii=False)
    json.dump(te, open(os.path.join(M, "loro_%s.json" % name), "w"), ensure_ascii=False)
    lst = os.path.join(DS, "train_loro_%s.txt" % name)
    with open(lst, "w") as fh:
        for e in tr:
            p = os.path.join(DS, "images", str(e["image_id"]) + ".png")
            if os.path.exists(p):
                fh.write(p + "\n")
    with open(os.path.join(DS, "data_loro_%s.yaml" % name), "w") as fh:
        fh.write("path: %s\ntrain: %s\nval: %s\nnames:\n  0: landslide\n" % (DS, lst, lst))
    ngt = sum(len(e["labels"]["annotations"]) for e in te)
    print("%-12s train=%5d  test=%4d  test_gt=%5d  events=%s" % (name, len(tr), len(te), ngt, ",".join(evs)))
