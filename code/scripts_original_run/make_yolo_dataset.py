"""Build a YOLO-format dataset from the event-level split (for fine-tuned baselines).

Renders ``optical_pre`` to 8-bit RGB at ``--input-size`` and writes the GT boxes
(native 128x128 frame) as normalised YOLO ``class cx cy w h`` labels.  GT-free
tiles are emitted with an empty label file so the detector also learns background
(the same negatives Module 1 trains on).

The event-level split is preserved exactly, so the fine-tuned baseline sees the
same train events as Module 1 and is evaluated on the same held-out test events.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--input-size', type=int, default=320)
    p.add_argument('--native-size', type=int, default=128)
    p.add_argument('--rgb-mode', default='percentile', choices=['percentile', 'plain'])
    p.add_argument('--include-negatives', action='store_true', default=True)
    p.add_argument('--max-images', type=int, default=None)
    return p.parse_args()


def to_rgb_uint8(arr, mode='percentile'):
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
                lo, hi = float(a[..., c].min()), max(float(a[..., c].max()), 1.0)
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

    entries = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if args.max_images:
        entries = entries[: args.max_images]
    root = Path(args.out)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'labels').mkdir(parents=True, exist_ok=True)

    n_img = n_box = n_neg = 0
    s = float(args.input_size) / float(args.native_size)
    for e in entries:
        rel = (e.get('modalities') or {}).get('optical_pre')
        if rel is None:
            continue
        anns = (e.get('labels') or {}).get('annotations') or []
        if not anns and not args.include_negatives:
            continue
        arr = np.load(os.path.join(args.data_root, rel))
        rgb = resize_nearest(to_rgb_uint8(arr, args.rgb_mode),
                             (args.input_size, args.input_size))
        stem = str(e['image_id'])
        Image.fromarray(rgb).save(root / 'images' / f'{stem}.png', optimize=False)

        lines = []
        for an in anns:
            x1, y1, x2, y2 = [float(v) * s for v in an['bbox_xyxy']]
            x1 = max(0.0, min(x1, args.input_size)); x2 = max(0.0, min(x2, args.input_size))
            y1 = max(0.0, min(y1, args.input_size)); y2 = max(0.0, min(y2, args.input_size))
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            cx = (x1 + x2) / 2.0 / args.input_size
            cy = (y1 + y2) / 2.0 / args.input_size
            lines.append('0 %.6f %.6f %.6f %.6f' % (cx, cy, w / args.input_size, h / args.input_size))
        (root / 'labels' / f'{stem}.txt').write_text('\n'.join(lines), encoding='utf-8')
        n_img += 1; n_box += len(lines)
        if not lines:
            n_neg += 1
        if n_img % 500 == 0:
            print('[%d/%d] imgs, %d boxes' % (n_img, len(entries), n_box), flush=True)

    yaml = root / 'dataset.yaml'
    yaml.write_text(
        'path: %s\ntrain: images\nval: images\nnames:\n  0: landslide\n' % root.resolve(),
        encoding='utf-8')
    print('DONE images=%d boxes=%d negatives=%d -> %s' % (n_img, n_box, n_neg, root), flush=True)


if __name__ == '__main__':
    main()
