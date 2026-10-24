import json, os, numpy as np
from PIL import Image, ImageDraw
ROOT="/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
m=json.load(open(ROOT+"/manifests/test.json"))
# 取 4 个含 GT 的瓦片
sel=[e for e in m if e["labels"]["annotations"]][:4]
tiles=[]
for e in sel:
    a=np.load(os.path.join(ROOT, e["modalities"]["optical_pre"])).astype(np.float32)
    # mode A: percentile
    out=np.empty_like(a)
    for c in range(a.shape[-1]):
        lo,hi=np.percentile(a[...,c],[2,98]); 
        out[...,c]=np.clip((a[...,c]-lo)/max(hi-lo,1e-6),0,1)
    A=(out*255).astype(np.uint8)
    # mode B: plain  /10000*3
    B=(np.clip(a/10000.0*3.0,0,1)*255).astype(np.uint8)
    tiles.append((e["image_id"],e,A,B))
for name,e,A,B in tiles:
    for tag,img in (("pct",A),("plain",B)):
        im=Image.fromarray(img).resize((256,256), Image.NEAREST)
        d=ImageDraw.Draw(im)
        for an in e["labels"]["annotations"]:
            x1,y1,x2,y2=an["bbox_xyxy"]
            d.rectangle([x1*2,y1*2,x2*2,y2*2], outline=(255,0,0), width=2)
        im.save("/root/autodl-tmp/rgbcheck_%s_%s.png"%(name,tag))
print("saved", len(tiles)*2, "pngs")
for name,e,A,B in tiles:
    print(name, "native", A.shape, "pct range", A.min(), A.max(), "| plain range", B.min(), B.max(), "| gt", len(e["labels"]["annotations"]))
