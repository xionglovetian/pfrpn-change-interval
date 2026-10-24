# Copyright (c) OpenMMLab. All rights reserved.
"""External prompt-based baseline: Grounding DINO -> proposal.v1.

Runs the HuggingFace Grounding-DINO checkpoint on the optical-pre channel of the
event-level split and writes proposal.v1 JSON so the SAME evaluator
(``tools/eval_proposal_metrics.py``) can score it against the same GT.

Frame convention
----------------
The evaluator IoU-matches proposals against GT that lives in the **native 128x128
tile frame**, so this script maps model boxes back to that frame.  The image fed
to the detector is upsampled to ``--input-size`` (default 320) to match Module 1's
model-input resolution, so the comparison is resolution-fair.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True,
                   help='local HF model dir (or hub id)')
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-root', required=True,
                   help='dataset root the manifest modality paths are relative to')
    p.add_argument('--out', required=True)
    p.add_argument('--prompt', default='landslide.',
                   help='period-separated text prompt, e.g. "landslide."')
    p.add_argument('--box-threshold', type=float, default=0.20)
    p.add_argument('--text-threshold', type=float, default=0.20)
    p.add_argument('--input-size', type=int, default=320)
    p.add_argument('--native-size', type=int, default=128)
    p.add_argument('--max-per-img', type=int, default=300)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--model-version', default='grounding-dino-base')
    p.add_argument('--rgb-mode', default='percentile',
                   choices=['percentile', 'plain'],
                   help='how to turn reflectance into an 8-bit RGB image')
    return p.parse_args()


def to_rgb_uint8(arr, mode='percentile', lo=2.0, hi=98.0):
    """(H,W,3) reflectance -> (H,W,3) uint8."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if mode == 'plain':
        x = np.clip(a / 10000.0 * 3.0, 0.0, 1.0)
    else:
        out = np.empty_like(a)
        for c in range(a.shape[-1]):
            ch = a[..., c]
            p_lo, p_hi = np.percentile(ch, [lo, hi])
            if p_hi <= p_lo:
                p_lo, p_hi = float(ch.min()), max(float(ch.max()), float(ch.min()) + 1)
            out[..., c] = np.clip((ch - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        x = out
    return (x * 255.0).astype(np.uint8)


def resize_nearest(img, size):
    h, w = size
    ys = np.clip(((np.arange(h) + 0.5) * img.shape[0] / h).astype(int), 0, img.shape[0] - 1)
    xs = np.clip(((np.arange(w) + 0.5) * img.shape[1] / w).astype(int), 0, img.shape[1] - 1)
    return img[ys][:, xs]


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    device = args.device if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model_dir).to(device).eval()

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    entries = manifest[: args.max_images] if args.max_images else manifest
    out_root = Path(args.out)
    scale_back = float(args.native_size) / float(args.input_size)

    n_done = 0
    n_cand = 0
    for entry in entries:
        image_id = str(entry['image_id'])
        event_id = str(entry.get('event_id', ''))
        rel = (entry.get('modalities') or {}).get('optical_pre')
        if rel is None:
            continue
        tile = np.load(os.path.join(args.data_root, rel))
        rgb = to_rgb_uint8(tile, args.rgb_mode)
        rgb = resize_nearest(rgb, (args.input_size, args.input_size))
        pil = Image.fromarray(rgb)

        inputs = processor(images=pil, text=args.prompt,
                           return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'],
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(args.input_size, args.input_size)])[0]

        boxes = res['boxes'].detach().cpu().numpy()
        scores = res['scores'].detach().cpu().numpy()
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]

        cands = []
        for i, (b, s) in enumerate(zip(boxes, scores)):
            if len(cands) >= args.max_per_img:
                break
            x1, y1, x2, y2 = [float(v) * scale_back for v in b]
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
            'meta': {
                'prompt': args.prompt,
                'box_threshold': args.box_threshold,
                'text_threshold': args.text_threshold,
                'input_size': args.input_size,
                'native_size': args.native_size,
                'rgb_mode': args.rgb_mode,
                'bbox_frame': 'original',
            },
        }
        d = out_root / event_id
        d.mkdir(parents=True, exist_ok=True)
        (d / ('%s.proposals.json' % image_id)).write_text(
            json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        n_done += 1
        n_cand += len(cands)
        if n_done % 100 == 0:
            print('[%d/%d] images, %d candidates' % (n_done, len(entries), n_cand), flush=True)

    print('DONE images=%d candidates=%d -> %s' % (n_done, n_cand, out_root), flush=True)


if __name__ == '__main__':
    main()
