import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, "/root/autodl-tmp/landslide-framework/module2")
from src.mine_hard_negatives import DEFAULT_EXPORT_SCALE, MiningConfig, load_gt_from_manifest, mine_proposal_dir
from src.training_data import build_samples, preload_items
from src.metrics import average_precision

DATA = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
OUT  = "/root/autodl-tmp/m2_roi_cache"
MODS = ("optical", "sar", "topography", "change")

def gt_split(split):
    g = load_gt_from_manifest(f"{DATA}/manifests/{split}.json", box_scale=DEFAULT_EXPORT_SCALE)
    for v in g.values():
        td = v.get("tile_dir")
        if td and not td.startswith("/"):
            v["tile_dir"] = os.path.normpath(os.path.join(DATA, td))
    return g

def mine(root, split, g):
    p = mine_proposal_dir(root, "/tmp/d", g, split=split, config=MiningConfig(score_thr=0.40), write=False)
    return p["labels"]

t0 = time.time()
gt = {s: gt_split(s) for s in ("train", "val")}
train_all = build_samples(f"{OUT}/A4_train_pos", "train", mine(f"{OUT}/A4_train_pos", "train", gt["train"]), gt_index=gt["train"])
dev_all   = build_samples(f"{OUT}/A4_hokkaido",  "val",   mine(f"{OUT}/A4_hokkaido",  "val",   gt["val"]),  gt_index=gt["val"])
train_s = [s for s in train_all if s.label == 1 and s.event_id != "itogon"]
neg_s   = [s for s in train_all if s.label == 0 and s.event_id != "itogon"]
dev_s   = [s for s in dev_all if s.event_id == "hokkaido"][:0]   # placeholder
# dev = itogon from train export (same as training run)
dev_s   = [s for s in train_all if s.event_id == "itogon"]
train_s = train_s + neg_s
print("train=%d (pos %d / neg %d)  dev=%d (pos %d / neg %d)" % (
    len(train_s), sum(1 for s in train_s if s.label==1), sum(1 for s in train_s if s.label==0),
    len(dev_s), sum(1 for s in dev_s if s.label==1), sum(1 for s in dev_s if s.label==0)), flush=True)

tr = preload_items(train_s, MODS, 28)
dv = preload_items(dev_s, MODS, 28)
print("preload %.0fs" % (time.time()-t0), flush=True)

def feat(items):
    X = np.stack([np.concatenate([it["roi_feats"][m].astype(np.float32).mean(axis=(1,2)) for m in MODS]) for it in items])
    y = np.array([int(it["cls_label"]) for it in items])
    return X, y

Xtr, ytr = feat(tr); Xdv, ydv = feat(dv)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr_n = (Xtr-mu)/sd; Xdv_n = (Xdv-mu)/sd
print("feature dim=%d  dev pos rate=%.4f" % (Xtr.shape[1], ydv.mean()), flush=True)

def fit_eval(Xa, ya, Xb, yb, steps=800, lr=1e-2):
    torch.manual_seed(0)
    model = torch.nn.Linear(Xa.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    xa = torch.tensor(Xa, dtype=torch.float32); ya_t = torch.tensor(ya, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad(); loss = lossf(model(xa).squeeze(1), ya_t); loss.backward(); opt.step()
    with torch.no_grad():
        pa = torch.sigmoid(model(torch.tensor(Xa, dtype=torch.float32)).squeeze(1)).numpy()
        pb = torch.sigmoid(model(torch.tensor(Xb, dtype=torch.float32)).squeeze(1)).numpy()
    return average_precision(ya.tolist(), pa.tolist()), average_precision(yb.tolist(), pb.tolist())

ap_tr, ap_dv = fit_eval(Xtr_n, ytr, Xdv_n, ydv)
print("LINEAR PROBE  train_AP=%.4f  dev_AP=%.4f   (dev chance=%.4f)" % (ap_tr, ap_dv, ydv.mean()), flush=True)

# objectness statistics
import json as _j
def obj_stats(items):
    pos = [it for it in items if int(it["cls_label"])==1]; neg=[it for it in items if int(it["cls_label"])==0]
    return len(pos), len(neg)
print("done %.0fs" % (time.time()-t0))