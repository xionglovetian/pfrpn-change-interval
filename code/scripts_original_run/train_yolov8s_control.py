import os
os.environ["YOLO_CONFIG_DIR"] = "/root/autodl-tmp/.yolo"
from ultralytics import YOLO
m = YOLO("yolov8s.pt")
print("start CONTROL training: plain YOLOv8s (COCO) on same data", flush=True)
m.train(
    data="/root/autodl-tmp/yolo_ds/train/dataset.yaml",
    epochs=50, imgsz=320, batch=16, device=0,
    project="/root/autodl-tmp/yolo_runs", name="yolov8s_ft",
    exist_ok=True, workers=8, patience=0, seed=42,
    cache=True, plots=False, amp=False, verbose=True,
)
print("CONTROL TRAIN DONE", flush=True)
