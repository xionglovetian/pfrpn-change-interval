import json, os, sys, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
sys.path.insert(0, "/root/autodl-tmp")
from run_grounding_dino import to_rgb_uint8, resize_nearest
MODEL="/root/autodl-tmp/models/grounding-dino-base"
ROOT="/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
INPUT=320; SB=128/INPUT
m=json.load(open(os.path.join(ROOT,"manifests/test.json")))
sel=[e for e in m if e["labels"]["annotations"]][:8]
proc=AutoProcessor.from_pretrained(MODEL)
model=GroundingDinoForObjectDetection.from_pretrained(MODEL).to("cuda").eval()

e=sel[0]
arr=np.load(os.path.join(ROOT,e["modalities"]["optical_pre"]))
print("tile", e["image_id"], "img", arr.shape, "GT(native 128):", [an["bbox_xyxy"] for an in e["labels"]["annotations"]])
rgb=resize_nearest(to_rgb_uint8(arr,"plain"),(INPUT,INPUT))
for th in (0.10, 0.20, 0.35):
    inp=proc(images=Image.fromarray(rgb), text="landslide.", return_tensors="pt").to("cuda")
    with torch.no_grad(): out=model(**inp)
    r=proc.post_process_grounded_object_detection(out, inp["input_ids"], box_threshold=th,
        text_threshold=0.15, target_sizes=[(INPUT,INPUT)])[0]
    b=(r["boxes"].detach().cpu().numpy()*SB); s=r["scores"].detach().cpu().numpy()
    print(" thr=%.2f n=%d" % (th, len(b)))
    for bb,ss in list(zip(b,s))[:5]:
        print("    box", [round(float(x),1) for x in bb], "score %.3f" % ss)
print()
print("=== prompt 扫描（8 图, box_thr=0.15）===")
for pr in ["landslide.","rockslide.","landslide scar.","mudslide.","object.","terrain.","bare ground.","water."]:
    tot=0; best=[]
    for e in sel:
        arr=np.load(os.path.join(ROOT,e["modalities"]["optical_pre"]))
        rgb=resize_nearest(to_rgb_uint8(arr,"plain"),(INPUT,INPUT))
        inp=proc(images=Image.fromarray(rgb), text=pr, return_tensors="pt").to("cuda")
        with torch.no_grad(): out=model(**inp)
        r=proc.post_process_grounded_object_detection(out, inp["input_ids"], box_threshold=0.15,
            text_threshold=0.15, target_sizes=[(INPUT,INPUT)])[0]
        tot+=len(r["boxes"])
    print("  %-18s total dets over 8 tiles = %d" % (pr, tot))
