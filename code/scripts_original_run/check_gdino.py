import json, os, sys, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
sys.path.insert(0, "/root/autodl-tmp")
from run_grounding_dino import to_rgb_uint8, resize_nearest

MODEL = "/root/autodl-tmp/models/grounding-dino-base"
ROOT = "/root/autodl-tmp/PF-RPN/data/landslide_multimodal_hn_lb"
N = 8; INPUT = 320; NATIVE = 128; SB = NATIVE / INPUT

def iou_mat(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if a.size == 0 or b.size == 0: return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2]-a[:, 0], 0, None)*np.clip(a[:, 3]-a[:, 1], 0, None)
    bb = np.clip(b[:, 2]-b[:, 0], 0, None)*np.clip(b[:, 3]-b[:, 1], 0, None)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)

m = json.load(open(os.path.join(ROOT, "manifests/test.json")))
sel = [e for e in m if e["labels"]["annotations"]][:N]
print("tiles=%d  GT boxes=%s" % (len(sel), [len(e["labels"]["annotations"]) for e in sel]), flush=True)

proc = AutoProcessor.from_pretrained(MODEL)
model = GroundingDinoForObjectDetection.from_pretrained(MODEL).to("cuda").eval()
print("model loaded", flush=True)

for mode in ["percentile", "plain"]:
    dets, ious = [], []
    for e in sel:
        arr = np.load(os.path.join(ROOT, e["modalities"]["optical_pre"]))
        rgb = resize_nearest(to_rgb_uint8(arr, mode), (INPUT, INPUT))
        inp = proc(images=Image.fromarray(rgb), text="landslide.", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_grounded_object_detection(
            out, inp["input_ids"], box_threshold=0.2, text_threshold=0.2,
            target_sizes=[(INPUT, INPUT)])[0]
        boxes = r["boxes"].detach().cpu().numpy() * SB
        gt = np.asarray([an["bbox_xyxy"] for an in e["labels"]["annotations"]], float)
        dets.append(len(boxes))
        M = iou_mat(gt, boxes)
        ious.append(float(M.max()) if M.size else 0.0)
    print("MODE=%-11s dets/img=%.1f  GT bestIoU mean=%.3f max=%.3f  #tiles>=0.5=%d" % (
        mode, np.mean(dets), np.mean(ious), np.max(ious), sum(1 for v in ious if v >= 0.5)), flush=True)
