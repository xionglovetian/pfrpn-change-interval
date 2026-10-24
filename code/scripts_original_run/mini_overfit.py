import os, glob, shutil, random
os.environ["YOLO_CONFIG_DIR"] = "/root/autodl-tmp/.yolo"
SRC = "/root/autodl-tmp/yolo_ds/train"
DST = "/root/autodl-tmp/yolo_ds/mini"
random.seed(0)
for sub in ("images", "labels"):
    os.makedirs(os.path.join(DST, sub), exist_ok=True)
imgs = sorted(glob.glob(os.path.join(SRC, "images", "*.png")))
pos = [p for p in imgs if os.path.getsize(os.path.join(SRC, "labels", os.path.basename(p).replace(".png", ".txt"))) > 0]
pick = random.sample(pos, 100)
for p in pick:
    b = os.path.basename(p)
    shutil.copy(p, os.path.join(DST, "images", b))
    shutil.copy(os.path.join(SRC, "labels", b.replace(".png", ".txt")),
                os.path.join(DST, "labels", b.replace(".png", ".txt")))
open(os.path.join(DST, "dataset.yaml"), "w").write(
    "path: %s\ntrain: images\nval: images\nnames:\n  0: landslide\n" % DST)
print("mini dataset:", len(pick), "images (positives only)")

from ultralytics import YOLO
m = YOLO("/root/autodl-tmp/yolov8s.pt")
m.train(data=os.path.join(DST, "dataset.yaml"), epochs=120, imgsz=320, batch=8,
        device=0, project="/root/autodl-tmp/yolo_runs", name="mini_overfit",
        exist_ok=True, workers=4, patience=0, seed=0, cache=True,
        plots=False, amp=False, verbose=False)
print("MINI TRAIN DONE", flush=True)
