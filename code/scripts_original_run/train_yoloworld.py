import os
os.environ["YOLO_CONFIG_DIR"] = "/root/autodl-tmp/.yolo"
from ultralytics import YOLO
m = YOLO("/root/autodl-tmp/yolov8s-worldv2.pt")
m.set_classes(["landslide"])
print("start training (amp=False)", flush=True)
m.train(
    data="/root/autodl-tmp/yolo_ds/train/dataset.yaml",
    epochs=50, imgsz=320, batch=16, device=0,
    project="/root/autodl-tmp/yolo_runs", name="yoloworld_ft",
    exist_ok=True, workers=8, patience=0, seed=42,
    cache=True, plots=False, amp=False, verbose=True,
)
print("TRAIN DONE", flush=True)
