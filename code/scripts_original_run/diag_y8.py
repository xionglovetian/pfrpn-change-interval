import json, os, numpy as np
from PIL import Image
from ultralytics import YOLO
ROOT = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
def iou_mat(a,b):
    a=np.asarray(a,float).reshape(-1,4); b=np.asarray(b,float).reshape(-1,4)
    if a.size==0 or b.size==0: return np.zeros((len(a),len(b)))
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip(a[:,2]-a[:,0],0,None)*np.clip(a[:,3]-a[:,1],0,None)
    bb=np.clip(b[:,2]-b[:,0],0,None)*np.clip(b[:,3]-b[:,1],0,None)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-6)
m = YOLO("/root/autodl-tmp/yolo_runs/yolov8s_200e/weights/best.pt")
for split in ["train", "test"]:
    ents = [e for e in json.load(open(os.path.join(ROOT,"manifests/%s.json"%split))) if e["labels"]["annotations"]][:12]
    tot=0; hit=0; boxhit=0
    for e in ents:
        p = os.path.join(ROOT, e["modalities"]["optical_pre"])
        a = np.load(p).astype(np.float32)
        out = np.empty_like(a)
        for c in range(3):
            lo,hi=np.percentile(a[...,c],[2,98]); out[...,c]=np.clip((a[...,c]-lo)/max(hi-lo,1e-6),0,1)
        rgb=(out*255).astype(np.uint8)
        rgb = np.asarray(Image.fromarray(rgb).resize((320,320), Image.NEAREST))
        r = m.predict(source=Image.fromarray(rgb), conf=0.001, imgsz=320, verbose=False, device=0, max_det=300)[0]
        b = np.zeros((0,4)) if r.boxes is None else r.boxes.xyxy.cpu().numpy()
        gt = np.asarray([an["bbox_xyxy"] for an in e["labels"]["annotations"]], float)  # native 128 frame
        b128 = b * (128.0/320.0)
        tot += len(b)
        M = iou_mat(gt, b128)
        if M.size:
            hit += int((M.max(axis=1) >= 0.5).sum())
            boxhit += len(b)
    print("%-6s tiles=%d  dets=%d (%.1f/tile)  GT matched@0.5=%d" % (split, len(ents), tot, tot/len(ents), hit), flush=True)
