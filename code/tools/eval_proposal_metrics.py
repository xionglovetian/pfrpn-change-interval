# Copyright (c) OpenMMLab. All rights reserved.
"""Score exported proposal.v1 artifacts against event-level GT manifests.

Metrics follow RESEARCH_ROADMAP.md section 5.9:

* AR@100 / AR@300 / AR@900  (COCO-style average recall, IoU 0.5)
* COCO-style averaged AR over IoU 0.50:0.05:0.95
* Recall split by GT size (small/medium/large)
* Recall at fixed FPPI (0.1 / 1 / 10)

M0 owns the final AR100/300/900 evaluator; this M1 scorer is intentionally
dependency-light (numpy only) and reads the same ``proposal.v1`` files that
M2 consumes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per_image_io as pio  # noqa: E402

IOU_THRS = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('proposals_dir')
    parser.add_argument('manifest')
    parser.add_argument('--ks', type=int, nargs='+', default=[100, 300, 900])
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--fppi-points', type=float, nargs='+',
                        default=[0.1, 1.0, 10.0])
    parser.add_argument('--out', default=None)
    parser.add_argument('--dump-per-image', default=None,
                        help='write a per-image dump (.npz) for paired stats')
    return parser.parse_args()


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    aa = a[:, None, :]
    bb = b[None, :, :]
    x1 = np.maximum(aa[..., 0], bb[..., 0])
    y1 = np.maximum(aa[..., 1], bb[..., 1])
    x2 = np.minimum(aa[..., 2], bb[..., 2])
    y2 = np.minimum(aa[..., 3], bb[..., 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = np.maximum(aa[..., 2] - aa[..., 0], 0) * np.maximum(aa[..., 3] - aa[..., 1], 0)
    area_b = np.maximum(bb[..., 2] - bb[..., 0], 0) * np.maximum(bb[..., 3] - bb[..., 1], 0)
    return inter / (area_a + area_b - inter + 1e-6)


def size_bucket(box):
    area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    if area < 32 ** 2:
        return 'small'
    if area < 96 ** 2:
        return 'medium'
    return 'large'


def greedy_match(ious, order, thr=0.5):
    """Greedy IoU matching in score order; returns per-detection TP flags."""
    num_gt, num_pred = ious.shape
    flags = np.zeros(num_pred, dtype=bool)
    used = np.zeros(num_gt, dtype=bool)
    for j in order:
        if num_gt == 0:
            break
        col = ious[:, j]
        best = int(np.argmax(col))
        if col[best] >= thr and not used[best]:
            used[best] = True
            flags[j] = True
    return flags


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    by_image = {str(item.get('image_id')): item for item in manifest}

    n_scored = 0
    gt_count = 0
    hit = {k: 0 for k in args.ks}
    img_ar = {k: [] for k in args.ks}
    size_total = {'small': 0, 'medium': 0, 'large': 0}
    size_hit = {'small': 0, 'medium': 0, 'large': 0}
    coco_ar_img = {t: [] for t in IOU_THRS}
    per_image = []  # (scores, match_flags, gt_count)
    image_ids = []
    event_ids = []
    num_gt_list = []
    ar_rows = {k: [] for k in args.ks}
    score_parts = []
    flag_parts = []
    offsets = [0]

    prop_files = sorted(Path(args.proposals_dir).rglob('*.proposals.json'))
    for prop_path in prop_files:
        proposal = json.loads(prop_path.read_text(encoding='utf-8'))
        sample = by_image.get(str(proposal.get('image_id')))
        if sample is None:
            continue
        gt = [a['bbox_xyxy'] for a in sample['labels']['annotations']]
        cands = sorted(proposal['candidates'],
                       key=lambda c: c['proposal_score'], reverse=True)
        pred = np.asarray([c['bbox_xyxy'] for c in cands], dtype=np.float64) \
            if cands else np.zeros((0, 4), dtype=np.float64)
        scores = np.asarray([c['proposal_score'] for c in cands],
                            dtype=np.float64) if cands else np.zeros(0)
        order = np.arange(len(scores))
        ious = iou_matrix(np.asarray(gt, dtype=np.float64), pred)

        n_scored += 1
        gt_count += len(gt)
        for b in gt:
            size_total[size_bucket(b)] += 1

        if len(gt) > 0:
            for k in args.ks:
                matched = int((ious[:, :k].max(axis=1) >= args.iou).sum()) \
                    if ious.size else 0
                hit[k] += matched
                img_ar[k].append(matched / len(gt))
            best = ious.max(axis=1) if ious.size else np.zeros(len(gt))
            for b_idx, b in enumerate(gt):
                if best[b_idx] >= args.iou:
                    size_hit[size_bucket(b)] += 1
            for t in IOU_THRS:
                matched = int((ious.max(axis=1) >= t).sum()) if ious.size else 0
                coco_ar_img[t].append(matched / len(gt))
            flags = greedy_match(ious, order, args.iou)
        else:
            # COCO-style AR ignores images without GT: do NOT append 0.0 here.
            # Appending 0.0 dilutes AR by the negative-tile fraction (the exact
            # bug fixed in eval_m1_split.py).  Only the FPPI curve uses these
            # images, where every proposal counts as a false positive.
            flags = np.zeros(len(scores), dtype=bool)
        image_ids.append(str(proposal.get('image_id')))
        event_ids.append(str(proposal.get('event_id')
                              or sample.get('event_id') or 'unknown'))
        num_gt_list.append(len(gt))
        for k in args.ks:
            ar_rows[k].append(img_ar[k][-1] if len(gt) > 0 else 0.0)
        per_image.append((scores, flags, len(gt)))
        score_parts.append(np.asarray(scores, dtype=np.float64))
        flag_parts.append(np.asarray(flags, dtype=bool))
        offsets.append(offsets[-1] + len(scores))

    result = {
        'proposals_dir': str(args.proposals_dir),
        'manifest': str(args.manifest),
        'iou_threshold': args.iou,
        'num_proposal_files': len(prop_files),
        'num_images_scored': n_scored,
        'num_gt_boxes': gt_count,
        'AR': {f'AR@{k}': (float(np.mean(img_ar[k])) if img_ar[k] else 0.0)
               for k in args.ks},
        'micro_recall_at_iou': {
            f'recall@{k}': (hit[k] / gt_count if gt_count else 0.0)
            for k in args.ks},
        'AR_coco_style': float(
            np.mean([np.mean(v) for v in coco_ar_img.values() if v])
        ) if any(coco_ar_img.values()) else 0.0,
        'recall_by_size': {
            s: (size_hit[s] / size_total[s] if size_total[s] else 0.0)
            for s in ('small', 'medium', 'large')},
        'size_counts': dict(size_total),
    }

    all_scores = np.concatenate([r[0] for r in per_image if r[0].size]) \
        if any(r[0].size for r in per_image) else np.zeros(0)
    if all_scores.size:
        grid = np.unique(np.concatenate([
            np.quantile(all_scores, np.linspace(0.0, 1.0, 200)),
            np.linspace(0.0, all_scores.max() + 1e-6, 50)]))
        curve = []
        for thr in grid:
            tp = fp = 0
            for scores, flags, n_gt in per_image:
                if scores.size == 0:
                    continue
                keep = scores >= thr
                tp += int((flags & keep).sum())
                fp += int((~flags & keep).sum())
            curve.append((fp / max(n_scored, 1), tp / gt_count if gt_count else 0.0))
        curve.sort()
        fppis = np.asarray([c[0] for c in curve])
        recalls = np.asarray([c[1] for c in curve])
        at = {}
        for point in args.fppi_points:
            if fppis.max() < point:
                at[f'recall@{point}FPPI'] = float(recalls[-1])
            else:
                idx = int(np.searchsorted(fppis, point, side='right') - 1)
                at[f'recall@{point}FPPI'] = float(recalls[max(idx, 0)])
        result['recall_at_fppi'] = at

    if args.dump_per_image:
        pio.save_dump(
            args.dump_per_image,
            image_ids=image_ids,
            event_ids=event_ids,
            num_gt=num_gt_list,
            ar_by_k=ar_rows,
            scores=(np.concatenate(score_parts) if score_parts else np.zeros(0)),
            flags=(np.concatenate(flag_parts) if flag_parts
                   else np.zeros(0, dtype=bool)),
            offsets=offsets,
            meta={'proposals_dir': str(args.proposals_dir),
                  'manifest': str(args.manifest), 'ks': list(args.ks),
                  'iou': args.iou})
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()