"""Train one YOLOv8s fold (leave-one-event-out) with data yaml from argv[6]."""
import os, sys
os.environ["YOLO_CONFIG_DIR"] = "/root/autodl-tmp/.yolo"
from ultralytics import YOLO
AUG = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.0, translate=0.0,
           fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
           degrees=0.0, shear=0.0, perspective=0.0, close_mosaic=0)
base, name, imgsz, epochs, batch, data_yaml = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
m = YOLO(base)
m.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=0,
        project="/root/autodl-tmp/yolo_runs", name=name, exist_ok=True,
        workers=6, patience=0, seed=42, cache=False, plots=False,
        amp=False, verbose=False, **AUG)
print("DONE", name, flush=True)
