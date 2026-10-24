import os, json, glob, random
import numpy as np
from src.mine_hard_negatives import box_iou_matrix, load_gt_from_manifest

DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
V2 = "/root/autodl-tmp/m2_roi_cache_v2"
MODS = ("optical","sar","topography","change")
EXPECT_DIM = {"optical":256,"sar":192,"topography":256,"change":256}

SETS = [
    ("A4_train_pos", "train", f"{DATA}/manifests/train_pos.json", f"{DATA}/manifests/train.json"),
    ("A4_hokkaido",  "val",   f"{DATA}/manifests/event_hokkaido.json",  f"{DATA}/manifests/val.json"),
    ("A4_newzealand","val",   f"{DATA}/manifests/event_newzealand.json",f"{DATA}/manifests/val.json"),
    ("A4_test",      "test",  f"{DATA}/manifests/test.json",            f"{DATA}/manifests/test.json"),
]

ok_all = True
print("%-16s %-6s %8s %8s  %-8s %-10s %-10s %-8s" % ("split_set","split","expect","actual","ids","shapes","frame","nan/inf"))
for name, split, man, gtman in SETS:
    root = f"{V2}/{name}/{split}"
    exp = len(json.load(open(man, encoding="utf-8")))
    files = glob.glob(root + "/*/*.roi.npz")
    act = len(files)
    # manifest fields
    mf = f"{V2}/{name}/roi_feature_manifest.json"
    mfj = json.load(open(mf, encoding="utf-8")) if os.path.exists(mf) else {}
    frame = "%s/%s" % (mfj.get("bbox_frame","?"), mfj.get("roi_frame","?"))
    # sample deep checks
    random.seed(0)
    sample = random.sample(files, min(25, len(files)))
    ids_ok = shapes_ok = frame_ok = num_ok = True
    for f in sample:
        base = f[:-len(".roi.npz")]
        pj = base + ".proposals.json"
        if not os.path.exists(pj): ids_ok = False; continue
        doc = json.load(open(pj, encoding="utf-8"))
        pcid = [str(c["candidate_id"]) for c in doc["candidates"]]
        with np.load(f, allow_pickle=False) as d:
            rcid = [str(v) for v in d["candidate_id"]]
            if rcid != pcid: ids_ok = False
            for m in MODS:
                a = d[f"roi__{m}"]
                if a.shape[1:] != (EXPECT_DIM[m], 6, 6): shapes_ok = False
                if a.dtype != np.float16: shapes_ok = False
            b = np.array([c["bbox_xyxy"] for c in doc["candidates"]], float).reshape(-1,4)
            if len(b) and b.max() > 128.5: frame_ok = False
            pick = min(50, d["roi__optical"].shape[0])
            for m in MODS:
                a = d[f"roi__{m}"][:pick].astype(np.float32)
                if not np.isfinite(a).all(): num_ok = False
    status = "PASS" if (act==exp and ids_ok and shapes_ok and frame_ok and num_ok and mfj.get("bbox_frame")=="original") else "FAIL"
    if status=="FAIL": ok_all = False
    print("%-16s %-6s %8d %8d  %-8s %-10s %-10s %-8s  -> %s" % (
        name, split, exp, act, "OK" if ids_ok else "BAD", "OK" if shapes_ok else "BAD",
        frame, "OK" if num_ok else "BAD", status))
print()
print("B2 GATE:", "PASS" if ok_all else "FAIL (see rows above)")