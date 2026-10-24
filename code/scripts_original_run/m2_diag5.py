import os, sys, random, time
import numpy as np, torch, yaml
sys.path.insert(0,"/root/autodl-tmp/landslide-framework/module2")
from src.mine_hard_negatives import MiningConfig, load_gt_from_manifest, mine_proposal_dir
from src.training_data import build_samples, preload_items
from src.train import build_model
from src.metrics import average_precision

DATA="/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"; OUT="/root/autodl-tmp/m2_roi_cache_v2"
MODS=("optical","sar","topography","change")
def gt_of(s):
    g=load_gt_from_manifest(f"{DATA}/manifests/{s}.json",box_scale=1.0)
    for v in g.values():
        td=v.get("tile_dir")
        if td and not td.startswith("/"): v["tile_dir"]=os.path.normpath(os.path.join(DATA,td))
    return g
gtr,gva=gt_of("train"),gt_of("val")
L=lambda r,s,g: mine_proposal_dir(r,"/tmp/d",g,split=s,config=MiningConfig(score_thr=0.40),write=False)["labels"]
tr_all=build_samples(f"{OUT}/A4_train_pos","train",L(f"{OUT}/A4_train_pos","train",gtr),gt_index=gtr)
# sample a FEW IMAGES only -> fast preload
rnd=random.Random(0)
imgs=sorted({s.image_id for s in tr_all if s.event_id!="itogon"})
pick=set(rnd.sample(imgs,120))
tr=[s for s in tr_all if s.image_id in pick and s.event_id!="itogon"]
dv=[s for s in tr_all if s.event_id=="itogon"]
print("fast diag train=%d (pos %d) imgs=%d | dev=%d (pos %d)"%(
  len(tr),sum(1 for s in tr if s.label==1),len(pick),len(dv),sum(1 for s in dv if s.label==1)),flush=True)
t0=time.time(); itr=preload_items(tr,MODS,28); idv=preload_items(dv,MODS,28)
print("preload %.0fs"%(time.time()-t0),flush=True)
cfg=yaml.safe_load(open("/root/autodl-tmp/landslide-framework/module2/configs/module2.yaml",encoding="utf-8"))
def feat(items):
    X=np.stack([np.concatenate([x["roi_feats"][m].astype(np.float32).mean(axis=(1,2)) for m in MODS]) for x in items])
    y=np.array([int(x["cls_label"]) for x in items]); return X,y
Xtr,ytr=feat(itr); Xdv,ydv=feat(idv)
mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xdv=(Xdv-mu)/sd
torch.manual_seed(0); lin=torch.nn.Linear(Xtr.shape[1],1); opt=torch.optim.Adam(lin.parameters(),lr=1e-2,weight_decay=1e-3)
xa=torch.tensor(Xtr,dtype=torch.float32); ya=torch.tensor(ytr,dtype=torch.float32); lf=torch.nn.BCEWithLogitsLoss()
for _ in range(600):
    opt.zero_grad(); lf(lin(xa).squeeze(1),ya).backward(); opt.step()
with torch.no_grad():
    ptr=torch.sigmoid(lin(xa).squeeze(1)).numpy()
    pdv=torch.sigmoid(lin(torch.tensor(Xdv,dtype=torch.float32)).squeeze(1)).numpy()
print("LINEAR PROBE  train AP=%.4f | dev AP=%.4f | dev chance=%.4f"%(average_precision(ytr,ptr),average_precision(ydv,pdv),np.mean(ydv)),flush=True)
# also: trained model in-sample vs dev (if ckpt)
ck="/root/autodl-tmp/m2_train_v3/last.pth"
if os.path.exists(ck):
    st=torch.load(ck,map_location="cpu"); st=st.get("model",st)
    dims={m:int(st[f"encoder.proj.{m}.0.weight"].shape[0]) for m in MODS}
    mdl=build_model(cfg,dims,MODS); mdl.load_state_dict(st); mdl.eval()
    def sc(items):
        o=[]
        with torch.no_grad():
            for i in range(0,len(items),64):
                ch=items[i:i+64]; roi={m:torch.stack([torch.as_tensor(x["roi_feats"][m]) for x in ch]).float() for m in MODS}
                o.extend(torch.sigmoid(mdl(roi_feats=roi)["cls_logit"]).reshape(-1).tolist())
        return o
    print("TRAINED MODEL  train AP=%.4f | dev AP=%.4f"%(average_precision(ytr,sc(itr)),average_precision(ydv,sc(idv))),flush=True)