import os, json, glob, random
import numpy as np
from src.mine_hard_negatives import box_iou_matrix, load_gt_from_manifest
DATA="/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"; V2="/root/autodl-tmp/m2_roi_cache_v2"
MODS=("optical","sar","topography","change"); DIM={"optical":256,"sar":192,"topography":256,"change":256}
SETS=[("A4_train_pos","train","train_pos.json","train.json"),
      ("A4_hokkaido","val","event_hokkaido.json","val.json"),
      ("A4_newzealand","val","event_newzealand.json","val.json")]
allok=True
for name,split,man,gtman in SETS:
    root=f"{V2}/{name}/{split}"
    exp=len(json.load(open(f"{DATA}/manifests/{man}",encoding="utf-8")))
    files=glob.glob(root+"/*/*.roi.npz"); act=len(files)
    mf=json.load(open(f"{V2}/{name}/roi_feature_manifest.json",encoding="utf-8"))
    random.seed(0); sample=random.sample(files,min(20,len(files)))
    ids=shp=frame=num=True
    for f in sample:
        pj=f[:-len(".roi.npz")]+".proposals.json"
        if not os.path.exists(pj): ids=False; continue
        doc=json.load(open(pj,encoding="utf-8"))
        pcid=[str(c["candidate_id"]) for c in doc["candidates"]]
        with np.load(f,allow_pickle=False) as d:
            if [str(v) for v in d["candidate_id"]]!=pcid: ids=False
            for m in MODS:
                a=d[f"roi__{m}"]
                if a.shape[1:]!=(DIM[m],6,6) or a.dtype!=np.float16: shp=False
            b=np.array([c["bbox_xyxy"] for c in doc["candidates"]],float)
            if len(b) and b.max()>128.5: frame=False
            if not np.isfinite(d["roi__optical"][:20].astype(np.float32)).all(): num=False
    ok = (act==exp) and ids and shp and frame and num and mf.get("bbox_frame")=="original"
    allok &= ok
    print("%-16s %5d/%-5d  ids=%s shapes=%s frame=%s(%s) finite=%s -> %s" % (
        name, act, exp, "OK" if ids else "BAD", "OK" if shp else "BAD",
        "OK" if frame else "BAD", mf.get("bbox_frame"), "OK" if num else "BAD",
        "PASS" if ok else "FAIL"))
print("B2 (training splits):", "PASS" if allok else "FAIL")