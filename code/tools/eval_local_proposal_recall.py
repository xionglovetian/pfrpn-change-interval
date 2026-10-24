# Copyright (c) OpenMMLab. All rights reserved.
"""Local proposal recall dry-run using exported proposal.v1 files."""
import argparse
import json
from pathlib import Path

import numpy as np


def _iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(
        a[..., 3] - a[..., 1], 0)
    area_b = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(
        b[..., 3] - b[..., 1], 0)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('proposals_dir')
    parser.add_argument('manifest')
    parser.add_argument('--ks', type=int, nargs='+', default=[100, 300, 900])
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--out', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    sample_by_image = {item['image_id']: item for item in manifest}
    totals = {k: 0 for k in args.ks}
    gt_total = 0

    for prop_path in sorted(Path(args.proposals_dir).rglob('*.proposals.json')):
        proposal = json.loads(prop_path.read_text(encoding='utf-8'))
        sample = sample_by_image.get(proposal['image_id'])
        if sample is None:
            continue
        gt_boxes = [
            ann['bbox_xyxy']
            for ann in sample['labels']['annotations']
        ]
        candidates = sorted(
            proposal['candidates'],
            key=lambda item: item['proposal_score'],
            reverse=True)
        pred_boxes = [item['bbox_xyxy'] for item in candidates]
        gt_total += len(gt_boxes)
        if not gt_boxes:
            continue
        for k in args.ks:
            topk = pred_boxes[:k]
            if not topk:
                continue
            ious = _iou_matrix(gt_boxes, topk)
            matched = (ious.max(axis=1) >= args.iou_thr).sum().item()
            totals[k] += matched

    result = {
        'note': 'local dry-run only; official random patch split',
        'iou_threshold': args.iou_thr,
        'gt_boxes': gt_total,
        'recall': {
            str(k): (totals[k] / gt_total if gt_total else 0.0)
            for k in args.ks
        },
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()