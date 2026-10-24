# Copyright (c) OpenMMLab. All rights reserved.
"""Fast batched M1 checkpoint evaluation on an event-level split.

Runs the trained detector through a multi-worker data loader and computes the
RESEARCH_ROADMAP.md section 5.9 metrics directly from ``gt_instances`` (no
per-image proposale.v1 round trip), then prints a JSON report.  Use
``tools/inference/export_proposals.py`` for the M2-facing ``proposal.v1``
artifacts; this tool exists because single-image export is too slow to sweep
A0-A9.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.runner import Runner

import mmdet.datasets  # noqa: F401
import mmdet.models  # noqa: F401
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per_image_io as pio  # noqa: E402

IOU_THRS = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--split', default='val')
    parser.add_argument('--scale', type=int, default=320,
                        help='eval resize when the config has no test pipeline')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--ks', type=int, nargs='+', default=[100, 300, 900])
    parser.add_argument('--drop-modalities', nargs='*', default=[],
                        help='modalities to mask out, e.g. sar change')
    parser.add_argument('--zero-model-channels', nargs='*', default=[],
                        help='zero individual input channels, e.g. change:0 '
                             'change:2 (mechanism analysis: which change index '
                             'matters).  Channel indices are within the '
                             'stacked input for that modality.')
    parser.add_argument('--shuffle-modalities', nargs='*', default=[],
                        help='modalities whose channels are rolled across the '
                             'batch, breaking tile/label correspondence '
                             '(leakage sanity check)')
    parser.add_argument('--fppi-points', type=float, nargs='+',
                        default=[0.1, 1.0, 10.0])
    parser.add_argument('--out', default=None)
    parser.add_argument('--dump-per-image', default=None,
                        help='write a per-image dump (.npz) for paired stats')
    parser.add_argument('--cfg-options', nargs='*', default=[])
    return parser.parse_args()


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    aa, bb = a[:, None, :], b[None, :, :]
    x1 = np.maximum(aa[..., 0], bb[..., 0])
    y1 = np.maximum(aa[..., 1], bb[..., 1])
    x2 = np.minimum(aa[..., 2], bb[..., 2])
    y2 = np.minimum(aa[..., 3], bb[..., 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa_ = np.maximum(aa[..., 2] - aa[..., 0], 0) * np.maximum(aa[..., 3] - aa[..., 1], 0)
    bb_ = np.maximum(bb[..., 2] - bb[..., 0], 0) * np.maximum(bb[..., 3] - bb[..., 1], 0)
    return inter / (aa_ + bb_ - inter + 1e-6)


def size_bucket(box):
    area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    if area < 32 ** 2:
        return 'small'
    if area < 96 ** 2:
        return 'medium'
    return 'large'


def _parse_cfg_options(items):
    out = {}
    for item in items:
        key, _, value = item.partition('=')
        parts = key.split('.')
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            node[parts[-1]] = value
    return out


def _build_eval_dataset_cfg(cfg, split, scale):
    """Clone the training dataset config with a deterministic eval pipeline."""
    if cfg.get('test_dataloader'):
        base = dict(cfg.test_dataloader.dataset)
        base['ann_file'] = f'manifests/{split}.json'
        base['test_mode'] = True
        return base
    train_pipe = list(cfg.train_dataloader.dataset.pipeline)
    keep = []
    for step in train_pipe:
        if step['type'] in ('RandomFlip', 'RandomChoiceResize', 'RandomResize',
                            'RandomChoice', 'RandomCrop', 'FixScaleResize'):
            continue
        keep.append(dict(step))
    # Resize must come AFTER LoadAnnotations so that mmdet's Resize also
    # rescales gt_bboxes; inserting it earlier leaves GT in the original
    # frame while the image is resized, which silently destroys every IoU.
    insert_at = next(
        (i for i, s in enumerate(keep) if s['type'] == 'LoadAnnotations'),
        len(keep) - 1) + 1
    keep.insert(insert_at,
                dict(type='FixScaleResize', scale=(scale, scale),
                     keep_ratio=True))
    base = dict(cfg.train_dataloader.dataset)
    base['ann_file'] = f'manifests/{split}.json'
    base['test_mode'] = True
    base['pipeline'] = keep
    return base


def _modality_slices(model):
    """Channel slices for the enabled modalities, in stacked-input order."""
    slices = {}
    offset = 0
    for name in model.modality_order:
        width = int(model.modality_channels[name])
        slices[name] = (offset, offset + width)
        offset += width
    return slices


def _apply_modality_drop(data, dropped, model):
    """Zero the channels of dropped modalities and update modality_mask.

    ``MultiModalDetDataPreprocessor`` already produced a stacked input; the
    model reads ``data_sample.modality_mask`` and gates the fusion branches, so
    zeroing both keeps the ablation faithful to a missing sensor.
    """
    slices = _modality_slices(model)
    idx = [i for i, name in enumerate(model.modality_order) if name in dropped]
    if not idx:
        return
    for i in idx:
        lo, hi = slices[model.modality_order[i]]
        data['inputs'][:, lo:hi] = 0.0
    for sample in data['data_samples']:
        mask = getattr(sample, 'modality_mask', None)
        if mask is None:
            continue
        mask = torch.as_tensor(mask).clone()
        for i in idx:
            mask[i] = 0
        sample.set_metainfo(dict(modality_mask=mask))

def _apply_channel_zero(data, specs, model):
    """Zero individual stacked-input channels given 'modality:index' specs."""
    slices = _modality_slices(model)
    hits = []
    for spec in specs:
        name, _, idx = spec.partition(':')
        if name not in slices or not idx:
            continue
        lo, _hi = slices[name]
        j = lo + int(idx)
        data['inputs'][:, j] = 0.0
        hits.append(spec)
    if hits:
        for sample in data['data_samples']:
            sample.set_metainfo(dict(zero_channels=hits))


def _apply_modality_shuffle(data, shuffled, model):
    """Replace a modality's channels with another tile's within the batch.

    If the model genuinely relies on the per-tile signal this destroys AR; if
    AR is unaffected the modality is not being used as intended (or the
    prediction is driven by something else).
    """
    slices = _modality_slices(model)
    rolled = [name for name in shuffled if name in slices]
    if not rolled:
        return
    batch = data['inputs'].shape[0]
    if batch < 2:
        return
    generator = torch.Generator().manual_seed(1234)
    perm = torch.randperm(batch, generator=generator)
    for name in rolled:
        lo, hi = slices[name]
        data['inputs'][:, lo:hi] = data['inputs'][:, lo:hi][perm]


def _bootstrap_ci(values, n_boot=1000, seed=0):
    """Percentile bootstrap 95% CI over images (per-image AR values)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))
    dataset_cfg = dict(_build_eval_dataset_cfg(cfg, args.split, args.scale))
    cfg.test_dataloader = dict(dataset=dataset_cfg)
    dataset = DATASETS.build(dataset_cfg)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    # When shuffling a modality we also shuffle the loader so that batch-mates
    # come from different tiles/events, making the swap a real mismatch.
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=bool(args.shuffle_modalities),
        num_workers=args.workers, collate_fn=pseudo_collate)

    hit = {k: 0 for k in args.ks}
    img_ar = {k: [] for k in args.ks}
    gt_count = 0
    n_images = 0
    size_total = {'small': 0, 'medium': 0, 'large': 0}
    size_hit = {'small': 0, 'medium': 0, 'large': 0}
    coco_ar = {t: [] for t in IOU_THRS}
    per_image = []
    image_ids = []
    event_ids = []
    num_gt_list = []
    ar_rows = {k: [] for k in args.ks}
    score_parts = []
    flag_parts = []
    offsets = [0]
    ar_by_event = {}
    gt_by_event = {}
    limit = args.max_images

    with torch.no_grad():
        for batch in loader:
            data = model.data_preprocessor(batch, training=False)
            if args.drop_modalities:
                _apply_modality_drop(data, args.drop_modalities, model)
            if args.zero_model_channels:
                _apply_channel_zero(data, args.zero_model_channels, model)
            if args.shuffle_modalities:
                _apply_modality_shuffle(data, args.shuffle_modalities, model)
            results = model.predict(data['inputs'], data['data_samples'],
                                    rescale=False)
            for res in results:
                if limit is not None and n_images >= limit:
                    break
                n_images += 1
                event = res.metainfo.get('event_id', 'unknown')
                ar_by_event.setdefault(str(event), {k: [] for k in args.ks})
                gt_by_event[str(event)] = gt_by_event.get(str(event), 0)
                gt = res.gt_instances.bboxes.detach().cpu().numpy()
                pred = res.pred_instances.bboxes.detach().cpu().numpy()
                scores = res.pred_instances.scores.detach().cpu().numpy()
                order = np.argsort(-scores)
                pred = pred[order]
                scores = scores[order]
                gt_count += len(gt)
                gt_by_event[str(event)] = gt_by_event.get(str(event), 0) + len(gt)
                for b in gt:
                    size_total[size_bucket(b)] += 1
                ious = iou_matrix(gt, pred)
                if len(gt) > 0:
                    for k in args.ks:
                        m = int((ious[:, :k].max(axis=1) >= 0.5).sum()) if ious.size else 0
                        hit[k] += m
                        img_ar[k].append(m / len(gt))
                        ar_by_event[str(event)][k].append(m / len(gt))
                    best = ious.max(axis=1)
                    for i, b in enumerate(gt):
                        if best[i] >= 0.5:
                            size_hit[size_bucket(b)] += 1
                    for t in IOU_THRS:
                        m = int((ious.max(axis=1) >= t).sum()) if ious.size else 0
                        coco_ar[t].append(m / len(gt))
                else:
                    # COCO-style AR ignores images without GT annotations; only
                    # the FPPI curve uses them (all their proposals are FP).
                    # Diluting the average with 0.0 here silently understates
                    # AR by the fraction of negative tiles in the split.
                    pass
                image_ids.append(str(
                    res.metainfo.get('image_id')
                    or res.metainfo.get('sample_id')
                    or res.metainfo.get('img_id') or n_images))
                event_ids.append(str(event))
                num_gt_list.append(len(gt))
                for k in args.ks:
                    if len(gt) > 0 and ious.size:
                        ar_rows[k].append(float(
                            (ious[:, :k].max(axis=1) >= 0.5).sum() / len(gt)))
                    else:
                        ar_rows[k].append(0.0)
                # greedy TP/FP for FPPI
                flags = np.zeros(len(pred), dtype=bool)
                used = np.zeros(len(gt), dtype=bool)
                for j in range(len(pred)):
                    if len(gt) == 0:
                        break
                    col = ious[:, j]
                    b = int(np.argmax(col))
                    if col[b] >= 0.5 and not used[b]:
                        used[b] = True
                        flags[j] = True
                per_image.append((scores, flags, len(gt)))
                score_parts.append(np.asarray(scores, dtype=np.float64))
                flag_parts.append(np.asarray(flags, dtype=bool))
                offsets.append(offsets[-1] + len(scores))
            if limit is not None and n_images >= limit:
                break

    result = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'drop_modalities': list(args.drop_modalities),
        'shuffle_modalities': list(args.shuffle_modalities),
        'zero_channels': list(args.zero_model_channels),
        'num_images': n_images,
        'num_gt_boxes': gt_count,
        'AR': {f'AR@{k}': (float(np.mean(img_ar[k])) if img_ar[k] else 0.0)
               for k in args.ks},
        'micro_recall_at_iou': {
            f'recall@{k}': (hit[k] / gt_count if gt_count else 0.0)
            for k in args.ks},
        'AR_coco_style': float(np.mean([np.mean(v) for v in coco_ar.values() if v]))
        if any(coco_ar.values()) else 0.0,
        'recall_by_size': {s: (size_hit[s] / size_total[s] if size_total[s] else 0.0)
                           for s in ('small', 'medium', 'large')},
        'AR_ci95': {f'AR@{k}': list(_bootstrap_ci(img_ar[k]))
                    for k in args.ks},
        'AR_by_event': {e: {f'AR@{k}': (float(np.mean(v[k])) if v[k] else 0.0)
                            for k in args.ks}
                        for e, v in ar_by_event.items()},
        'gt_by_event': gt_by_event,
        'size_counts': dict(size_total),
    }
    all_scores = np.concatenate([r[0] for r in per_image if r[0].size]) \
        if any(r[0].size for r in per_image) else np.zeros(0)
    if all_scores.size:
        grid = np.unique(np.quantile(all_scores, np.linspace(0, 1, 200)))
        curve = []
        for thr in grid:
            tp = fp = 0
            for scores, flags, n_gt in per_image:
                keep = scores >= thr
                tp += int((flags & keep).sum())
                fp += int((~flags & keep).sum())
            curve.append((fp / max(n_images, 1), tp / gt_count if gt_count else 0.0))
        curve.sort()
        fppis = np.asarray([c[0] for c in curve])
        recalls = np.asarray([c[1] for c in curve])
        result['recall_at_fppi'] = {}
        for point in args.fppi_points:
            if fppis.max() < point:
                result['recall_at_fppi'][f'recall@{point}FPPI'] = float(recalls[-1])
            else:
                idx = int(np.searchsorted(fppis, point, side='right') - 1)
                result['recall_at_fppi'][f'recall@{point}FPPI'] = float(recalls[max(idx, 0)])
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
            meta={'config': args.config, 'checkpoint': args.checkpoint,
                  'split': args.split, 'ks': list(args.ks),
                  'drop_modalities': list(args.drop_modalities)})
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()