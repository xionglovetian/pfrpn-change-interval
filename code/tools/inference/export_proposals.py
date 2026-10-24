# Copyright (c) OpenMMLab. All rights reserved.
"""Export M2-PF-RPN candidates as proposal.v1 JSON + ROI feature NPZ."""
import argparse
import ast
import copy
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import default_collate

from mmdet.apis import init_detector
import mmdet.datasets  # noqa: F401
import mmdet.datasets.transforms  # noqa: F401
import mmdet.models  # noqa: F401
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

from proposal_io import write_proposal_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export M2-PF-RPN proposals.')
    parser.add_argument('config', help='Model config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument(
        '--out',
        default='artifacts/proposals/test',
        help='Output split root, e.g. artifacts/proposals/test')
    parser.add_argument('--split', default='test')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--max-per-img', type=int, default=900)
    parser.add_argument('--rescale', action='store_true', default=True)
    parser.add_argument('--no-roi', action='store_true')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--roi-per-modality', action='store_true',
                        help='also dump per-modality RoI features (M2 cache)')
    parser.add_argument('--roi-size', type=int, default=7,
                        help='RoI output size r for the per-modality dump')
    parser.add_argument('--scale', type=int, default=320,
                        help='eval resize when the config has no test pipeline')
    parser.add_argument('--cfg-options', nargs='*', default=[],
                        help='cfg overrides, e.g. model.use_sar=False')
    return parser.parse_args()


_ROI_STRIDE = 8  # level-0 feature stride for the 320x320 model input


def extract_per_modality_roi(model, inputs, bboxes, output_size=7):
    """Per-modality RoI features at level 0 (stride 8), *before* fusion.

    Mirrors ``M2PFRPN.extract_roi_features`` but keeps modalities separate so
    Module 2 receives ``roi__<modality>`` blocks it can fuse itself.
    """
    from torchvision.ops import roi_align

    feats = model.extract_modality_feats(inputs)
    rois = torch.cat([bboxes.new_zeros((bboxes.shape[0], 1)), bboxes], dim=1)
    out = {}
    for idx, name in enumerate(model.modality_order):
        fmap = feats[idx][0]
        roi = roi_align(fmap, rois, output_size=output_size,
                        spatial_scale=1.0 / _ROI_STRIDE, aligned=True)
        out[name] = roi.detach().cpu().numpy().astype(np.float16)
    return out


def export_one(model,
               dataset,
               index: int,
               out_root: str,
               split: str,
               score_thr: float,
               max_per_img: int,
               rescale: bool,
               save_roi: bool,
               roi_per_modality_flag: bool = False,
               roi_size: int = 7) -> dict:
    data = dataset[index]
    batch = default_collate([data])
    processed = model.data_preprocessor(batch, training=False)
    inputs = processed['inputs']
    data_samples = processed['data_samples']

    with torch.no_grad():
        results = model.predict(inputs, data_samples, rescale=rescale)

    sample = dataset.get_data_info(index)
    event_id = sample.get('event_id') or 'unknown_event'
    image_id = sample.get('image_id') or sample.get('img_id') or str(index)
    pred = results[0].pred_instances
    if len(pred) == 0:
        bboxes = pred.bboxes.new_zeros((0, 4))
        scores = pred.scores.new_zeros((0, ))
    else:
        bboxes = pred.bboxes
        scores = pred.scores

    keep = torch.where(scores >= score_thr)[0]
    if keep.numel() > max_per_img:
        keep = keep[torch.argsort(scores[keep], descending=True)[:max_per_img]]
    else:
        keep = keep[torch.argsort(scores[keep], descending=True)]

    bboxes = bboxes[keep]
    scores = scores[keep]
    objectness = getattr(pred, 'objectness', scores)[keep]
    proposal_score = getattr(pred, 'proposal_score', scores)[keep]
    uncertainty = getattr(pred, 'uncertainty', 1.0 - scores)[keep]
    valid = ((bboxes[:, 2] > bboxes[:, 0]) &
             (bboxes[:, 3] > bboxes[:, 1]))
    bboxes = bboxes[valid]
    scores = scores[valid]
    objectness = objectness[valid]
    proposal_score = proposal_score[valid]
    uncertainty = uncertainty[valid]
    levels = model._assign_levels(bboxes)

    roi_feats = None
    roi_per_modality = None
    modality_mask = sample.get('modality_mask')
    if modality_mask is not None:
        modality_mask = torch.as_tensor(
            modality_mask, dtype=torch.float32, device=inputs.device)
        modality_mask = modality_mask.unsqueeze(0)
    # --- RoI frame fix (2026-10-09) -------------------------------------
    # `rescale=True` returns prediction boxes in the ORIGINAL image frame, but
    # the feature maps come from the RESIZED model input.  RoI-Align must use
    # model-input-frame boxes; otherwise every RoI is taken from the wrong
    # location (shifted toward the origin by 1/scale_factor).
    roi_boxes = bboxes
    if rescale and bboxes.numel() > 0:
        _sf = data_samples[0].scale_factor
        if torch.is_tensor(_sf):
            _sf = _sf.to(device=bboxes.device, dtype=bboxes.dtype)
        else:
            _sf = torch.as_tensor(_sf, dtype=bboxes.dtype, device=bboxes.device)
        roi_boxes = bboxes.clone()
        roi_boxes[:, [0, 2]] = roi_boxes[:, [0, 2]] / _sf[0]
        roi_boxes[:, [1, 3]] = roi_boxes[:, [1, 3]] / _sf[1]
    if save_roi and bboxes.numel() > 0:
        roi_feats = model.extract_roi_features(
            inputs, roi_boxes, modality_mask=modality_mask)
    if roi_per_modality_flag and bboxes.numel() > 0:
        roi_per_modality = extract_per_modality_roi(
            model, inputs, roi_boxes, output_size=roi_size)

    candidates = []
    features = {}
    for order in range(bboxes.shape[0]):
        candidate_id = f'{image_id}_{order + 1:06d}'
        candidates.append(
            dict(
                candidate_id=candidate_id,
                bbox_xyxy=[
                    float(v) for v in bboxes[order].detach().cpu().tolist()
                ],
                objectness=float(objectness[order].detach().cpu()),
                proposal_score=float(proposal_score[order].detach().cpu()),
                level=int(levels[order].detach().cpu()),
                uncertainty=float(uncertainty[order].detach().cpu())))
        if roi_feats is not None:
            features[candidate_id] = (
                roi_feats[order].detach().cpu().numpy().astype(np.float32))
        else:
            features[candidate_id] = np.zeros((1, 1, 1), dtype=np.float32)

    meta = dict(
        num_queries=getattr(model, 'num_queries', 0),
        num_pseudo_tokens=getattr(model, 'num_pseudo_tokens', 0),
        sp_thr=getattr(model, 'sp_thr', 0.0),
        sp_iter_num=getattr(model, 'sp_iter_num', 0),
        topk=getattr(model, 'topk', 0),
        modalities_used=list(getattr(model, 'modality_order', [])))
    write_proposal_artifacts(
        output_dir=out_root,
        split=split,
        event_id=str(event_id),
        image_id=str(image_id),
        candidates=candidates,
        features=features,
        meta=meta)

    info = dict(image_id=str(image_id), event_id=str(event_id),
                num_candidates=len(candidates), dims=None)
    if roi_per_modality is not None:
        roi_dir = Path(out_root) / split / str(event_id)
        roi_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            'candidate_id': np.asarray(
                [item['candidate_id'] for item in candidates], dtype=str)
        }
        arrays.update({f'roi__{key}': value
                       for key, value in roi_per_modality.items()})
        np.savez_compressed(roi_dir / f'{image_id}.roi.npz', **arrays)
        info['dims'] = {key: int(value.shape[1])
                        for key, value in roi_per_modality.items()}
    return info

def main() -> None:
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        merged = {}
        for item in args.cfg_options:
            key, _, value = item.partition('=')
            parts = key.split('.')
            node = merged
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            try:
                node[parts[-1]] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                node[parts[-1]] = value
        cfg.merge_from_dict(merged)
    if not cfg.get('test_dataloader'):
        # Cloud configs keep validation disabled during training; rebuild a
        # deterministic eval pipeline (resize AFTER LoadAnnotations so that
        # gt_bboxes are rescaled too).
        train_pipe = list(cfg.train_dataloader.dataset.pipeline)
        keep = [dict(s) for s in train_pipe
                if s['type'] not in ('RandomFlip', 'RandomChoiceResize',
                                     'RandomResize', 'RandomChoice',
                                     'RandomCrop', 'FixScaleResize')]
        at = next((i for i, s in enumerate(keep)
                   if s['type'] == 'LoadAnnotations'), len(keep) - 1) + 1
        keep.insert(at, dict(type='FixScaleResize',
                             scale=(args.scale, args.scale),
                             keep_ratio=True))
        ds_cfg = copy.deepcopy(dict(cfg.train_dataloader.dataset))
        ds_cfg['pipeline'] = keep
        cfg.test_dataloader = dict(dataset=ds_cfg)
    cfg.test_dataloader.dataset.test_mode = True
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    out_root = args.out
    if osp.basename(osp.normpath(out_root)) == args.split:
        out_root = osp.dirname(osp.normpath(out_root))

    limit = args.max_images if args.max_images else len(dataset)
    infos = []
    for index in range(min(limit, len(dataset))):
        infos.append(export_one(
            model,
            dataset,
            index,
            out_root,
            args.split,
            args.score_thr,
            args.max_per_img,
            args.rescale,
            not args.no_roi,
            args.roi_per_modality,
            args.roi_size))
    if args.roi_per_modality:
        write_roi_manifest(out_root, args, model, infos)


def write_roi_manifest(out_root, args, model, infos):
    """Write the Module-2 alignment manifest for the per-modality RoI dump."""
    import json
    from pathlib import Path

    dims = {}
    for info in infos:
        if info.get('dims'):
            dims = info['dims']
            break
    order = list(getattr(model, 'modality_order', []))
    manifest = dict(
        schema_version='roi_feature.v1',
        producer='module1',
        task='m2-pf-rpn',
        split=args.split,
        backbone='swin-b',
        checkpoint=osp.abspath(args.checkpoint),
        config=osp.abspath(args.config),
        model_version='m2-pf-rpn-v1',
        roi_size=int(args.roi_size),
        bbox_frame='original',
        roi_frame='model_input',
        rescale=bool(args.rescale),
        dtype='float16',
        feature_level=0,
        stride=_ROI_STRIDE,
        spatial_scale={name: 1.0 / _ROI_STRIDE for name in order},
        modalities=order,
        feature_dims=dims,
        input_scale=int(args.scale),
        layout='<split>/<event_id>/<image_id>.roi.npz',
        npz_keys=dict(candidate_id='(N,) <U',
                      roi='roi__<modality> -> (N, C, r, r) float16'),
        roi_feature_ref='<split>/<event_id>/<image_id>.features.npz:<candidate_id>',
        num_images=len(infos),
        num_candidates=int(sum(i['num_candidates'] for i in infos)),
        per_image=[dict(image_id=i['image_id'], event_id=i['event_id'],
                        num_candidates=i['num_candidates']) for i in infos],
        notes=('Per-modality RoI features from the level-0 neck output '
               '(stride 8) in the model input frame, extracted before '
               'cross-modal fusion. spatial_scale is the feature-stride '
               'reciprocal.'),
    )
    path = Path(out_root) / 'roi_feature_manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    print('roi manifest ->', path)
    print('  roi_size=%d dims=%s total_candidates=%d' % (
        manifest['roi_size'], manifest['feature_dims'],
        manifest['num_candidates']))


if __name__ == '__main__':
    main()