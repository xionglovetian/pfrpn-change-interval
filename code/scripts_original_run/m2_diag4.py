import os, sys, json, glob, random, time
import numpy as np, torch
sys.path.insert(0,"/root/autodl-tmp/landslide-framework/module2")
from src.mine_hard_negatives import (DEFAULT_EXPORT_SCALE, MiningConfig,
    load_gt_from_manifest, mine_proposal_dir)
from src.training_data import build_samples, preload_items
from src.train import build_model
from src.metrics import average_precision
import yaml

DATA="/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"; OUT="/root/autodl-tmp/m2_roi_cache_v2"
MODS=("optical","sar","topography","change")
def gt_of(split):
    g=load_gt_from_manifest(f"{DATA}/manifests/{split}.json",box_scale=1.0)
    for v in g.values():
        td=v.get("tile_dir")
        if td and not td.startswith("/"): v["tile_dir"]=os.path.normpath(os.path.join(DATA,td))
    return g
gtr, gva = gt_of("train"), gt_of("val")
def mine(root,split,g):
    return mine_proposal_dir(root,"/tmp/d",g,split=split,config=MiningConfig(score_thr=0.40),write=False)["labels"]
tr_all=build_samples(f"{OUT}/A4_train_pos","train",mine(f"{OUT}/A4_train_pos","train",gtr),gt_index=gtr)
dv_all=build_samples(f"{OUT}/A4_hokkaido","val",mine(f"{OUT}/A4_hokkaido","val",gva),gt_index=gva)
pos=[s for s in tr_all if s.label==1]; neg=[s for s in tr_all if s.label==0]
rnd=random.Random(0); rnd.shuffle(pos); rnd.shuffle(neg)
tr=pos[:3000]+neg[:3000]
dv=[s for s in dv_all if s.event_id=="hokkaido"]
print("diag train=%d (pos %d)  dev=%d (pos %d)" % (len(tr),sum(1 for s in tr if s.label==1),len(dv),sum(1 for s in dv if s.label==1)),flush=True)
t0=time.time(); itr=preload_items(tr,MODS,28); idv=preload_items(dv,MODS,28)
print("preload %.0fs"%(time.time()-t0),flush=True)

# 1) trained model, if available
ck="/root/autodl-tmp/m2_train_v3/last.pth"
cfg=yaml.safe_load(open("/root/autodl-tmp/landslide-framework/module2/configs/module2.yaml",encoding="utf-8"))
if os.path.exists(ck):
    st=torch.load(ck,map_location="cpu"); st=st.get("model",st)
    dims={m:int(st[f"encoder.proj.{m}.0.weight"].shape[0]) for m in MODS}
    model=build_model(cfg,dims,MODS); model.load_state_dict(st); model.eval()
    def score(items):
        out=[]
        with torch.no_grad():
            for i in range(0,len(items),64):
                ch=items[i:i+64]
                roi={m:torch.stack([torch.as_tensor(x["roi_feats"][m]) for x in ch]).float() for m in MODS}
                out.extend(torch.sigmoid(model(roi_feats=roi)["cls_logit"]).reshape(-1).tolist())
        return out
    ytr=[int(x["cls_label"]) for x in itr]; ydv=[int(x["cls_label"]) for x in idv]
    print("MODEL(in-sample train) AP=%.4f  |  MODEL(dev) AP=%.4f" % (
        average_precision(ytr,score(itr)), average_precision(ydv,score(idv))),flush=True)

# 2) linear probe on pooled features
def feat(items):
    X=np.stack([np.concatenate([x["roi_feats"][m].astype(np.float32).mean(axis=(1,2)) for m in MODS]) for x in items])
    y=np.array([int(x["cls_label"]) for x in items]); return X,y
Xtr,ytr=feat(itr); Xdv,ydv=feat(idv)
mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xdv=(Xdv-mu)/sd
torch.manual_seed(0); lin=torch.nn.Linear(Xtr.shape[1],1); opt=torch.optim.Adam(lin.parameters(),lr=1e-2,weight_decay=1e-3)
xa=torch.tensor(Xtr,dtype=torch.float32); ya=torch.tensor(ytr,dtype=torch.float32)
lf=torch.nn.BCEWithLogitsLoss()
for _ in range(800):
    opt.zero_grad(); loss=lf(lin(xa).squeeze(1),ya); loss.backward(); opt.step()
with torch.no_grad():
    p_tr=torch.sigmoid(lin(xa).squeeze(1)).numpy()
    p_dv=torch.sigmoid(lin(torch.tensor(Xdv,dtype=torch.float32)).squeeze(1)).numpy()
print("LINEAR PROBE  train AP=%.4f | dev AP=%.4f | dev chance=%.4f" % (
    average_precision(ytr,p_tr), average_precision(ydv,p_dv), np.mean(ydv)),flush=True)