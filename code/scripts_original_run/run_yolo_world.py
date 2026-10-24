"""External baseline: YOLO-World (ultralytics) -> proposal.v1.

Same contract as run_grounding_dino.py: optical-pre -> 8-bit RGB at
--input-size, boxes mapped back to the native 128x128 frame that the GT and
the evaluator use.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--classes', nargs='+', default=['landslide'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conf', type=float, default=0.01)
    p.add_argument('--input-size', type=int, default=320)
    p.add_argument('--native-size', type=int, default=128)
    p.add_argument('--max-per-img', type=int, default=300)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--rgb-mode', default='plain', choices=['percentile', 'plain'])
    p.add_argument('--model-version', default='yolo-world')
    p.add_argument('--device', default='0')
    return p.parse_args()


def to_rgb_uint8(arr, mode='plain'):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if mode == 'plain':
        x = np.clip(a / 10000.0 * 3.0, 0.0, 1.0)
    else:
        out = np.empty_like(a)
        for c in range(a.shape[-1]):
            lo, hi = np.percentile(a[..., c], [2, 98])
            if hi <= lo:
                lo, hi = float(a[..., c].min()), float(a[..., c].max()) or 1.0
            out[..., c] = np.clip((a[..., c] - lo) / (hi - lo), 0.0, 1.0)
        x = out
    return (x * 255.0).astype(np.uint8)


def resize_nearest(img, size):
    h, w = size
    ys = np.clip(((np.arange(h) + 0.5) * img.shape[0] / h).astype(int), 0, img.shape[0] - 1)
    xs = np.clip(((np.arange(w) + 0.5) * img.shape[1] / w).astype(int), 0, img.shape[1] - 1)
    return img[ys][:, xs]


def main():
    args = parse_args()
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.weights)
    # YOLO-World exposes set_classes(); plain YOLOv8 keeps its trained head.
    if hasattr(model, 'set_classes'):
        model.set_classes([c for c in args.classes])

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    entries = manifest[: args.max_images] if args.max_images else manifest
    out_root = Path(args.out)
    sb = float(args.native_size) / float(args.input_size)

    n_done = n_cand = 0
    for entry in entries:
        image_id = str(entry['image_id'])
        event_id = str(entry.get('event_id', ''))
        rel = (entry.get('modalities') or {}).get('optical_pre')
        if rel is None:
            continue
        arr = np.load(os.path.join(args.data_root, rel))
        rgb = resize_nearest(to_rgb_uint8(arr, args.rgb_mode),
                             (args.input_size, args.input_size))
        res = model.predict(source=Image.fromarray(rgb), conf=args.conf,
                            imgsz=args.input_size, verbose=False,
                            device=args.device, max_det=args.max_per_img)[0]
        boxes = res.boxes.xyxy.detach().cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.detach().cpu().numpy() if res.boxes is not None else np.zeros(0)
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]

        cands = []
        for b, s in zip(boxes, scores):
            if len(cands) >= args.max_per_img:
                break
            x1, y1, x2, y2 = [float(v) * sb for v in b]
            x1 = max(0.0, min(x1, args.native_size)); y1 = max(0.0, min(y1, args.native_size))
            x2 = max(0.0, min(x2, args.native_size)); y2 = max(0.0, min(y2, args.native_size))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            cid = '%s_%06d' % (image_id, len(cands))
            cands.append({
                'candidate_id': cid,
                'bbox_xyxy': [x1, y1, x2, y2],
                'objectness': float(s),
                'proposal_score': float(s),
                'level': 0,
                'uncertainty': float(1.0 - s),
                'roi_feature_ref': '%s.features.npz:%s' % (image_id, cid[-6:]),
            })
        payload = {
            'schema_version': 'proposal.v1',
            'image_id': image_id,
            'event_id': event_id,
            'model_version': args.model_version,
            'candidates': cands,
            'meta': {'classes': list(args.classes), 'conf': args.conf,
                     'input_size': args.input_size, 'native_size': args.native_size,
                     'rgb_mode': args.rgb_mode, 'bbox_frame': 'original'},
        }
        d = out_root / event_id
        d.mkdir(parents=True, exist_ok=True)
        (d / ('%s.proposals.json' % image_id)).write_text(
            json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        n_done += 1; n_cand += len(cands)
        if n_done % 100 == 0:
            print('[%d/%d] images, %d candidates' % (n_done, len(entries), n_cand), flush=True)
    print('DONE images=%d candidates=%d -> %s' % (n_done, n_cand, out_root), flush=True)


if __name__ == '__main__':
    main()
