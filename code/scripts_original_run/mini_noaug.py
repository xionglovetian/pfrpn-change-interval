import os
os.environ["YOLO_CONFIG_DIR"] = "/root/autodl-tmp/.yolo"
from ultralytics import YOLO
m = YOLO("/root/autodl-tmp/yolov8s.pt")
m.train(
    data="/root/autodl-tmp/yolo_ds/mini/dataset.yaml",
    epochs=120, imgsz=320, batch=8, device=0,
    project="/root/autodl-tmp/yolo_runs", name="mini_noaug",
    exist_ok=True, workers=4, patience=0, seed=0, cache=True,
    plots=False, amp=False, verbose=False,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.0, translate=0.0,
    fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, shear=0.0, perspective=0.0, close_mosaic=0,
)
print("NOAUG DONE", flush=True)
