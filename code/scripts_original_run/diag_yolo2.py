import json, os, numpy as np
from PIL import Image
from ultralytics import YOLO
import sys
sys.path.insert(0, "/root/autodl-tmp")
from run_yolo_world import to_rgb_uint8, resize_nearest

ROOT = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
m = json.load(open(os.path.join(ROOT, "manifests/test.json")))
sel = [x for x in m if x["labels"]["annotations"]][:4]
model = YOLO("/root/autodl-tmp/yolov8s-worldv2.pt")
model.set_classes(["landslide"])

for mode in ["plain", "percentile"]:
    for sz in (320, 640):
        ninf = 0; mx = 0.0; ntot = 0
        for e in sel:
            arr = np.load(os.path.join(ROOT, e["modalities"]["optical_pre"]))
            rgb = resize_nearest(to_rgb_uint8(arr, mode), (sz, sz))
            r = model.predict(source=Image.fromarray(rgb), conf=0.0005, imgsz=sz,
                              verbose=False, device=0)[0]
            n = 0 if r.boxes is None else len(r.boxes)
            ntot += n
            if n: mx = max(mx, float(r.boxes.conf.max()))
            ninf += 1
        print("mode=%-11s imgsz=%3d  dets(4 tiles)=%d  maxconf=%.4f  mean_rgb=%s" % (
            mode, sz, ntot, mx,
            [int(v) for v in np.array(Image.fromarray(
                resize_nearest(to_rgb_uint8(np.load(os.path.join(ROOT, sel[0]["modalities"]["optical_pre"])), mode), (8,8)))).mean(axis=(0,1))]),
            flush=True)
