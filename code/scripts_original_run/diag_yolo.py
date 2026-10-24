import json, os, numpy as np
from PIL import Image
from ultralytics import YOLO
import sys
sys.path.insert(0, "/root/autodl-tmp")
from run_yolo_world import to_rgb_uint8, resize_nearest

ROOT = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
m = json.load(open(os.path.join(ROOT, "manifests/test.json")))
e = [x for x in m if x["labels"]["annotations"]][0]
arr = np.load(os.path.join(ROOT, e["modalities"]["optical_pre"]))
rgb = resize_nearest(to_rgb_uint8(arr, "plain"), (320, 320))
print("tile", e["image_id"], "rgb", rgb.shape, "range", int(rgb.min()), int(rgb.max()), flush=True)

model = YOLO("/root/autodl-tmp/yolov8s-worldv2.pt")
for cls in ["landslide", "rock", "tree", "car", "building", "landslide scar"]:
    model.set_classes([cls])
    res = model.predict(source=Image.fromarray(rgb), conf=0.001, verbose=False, device=0)[0]
    n = 0 if res.boxes is None else len(res.boxes)
    mx = 0.0 if (res.boxes is None or n == 0) else float(res.boxes.conf.max())
    print("  class=%-16s dets=%3d  maxconf=%.4f" % (cls, n, mx), flush=True)
