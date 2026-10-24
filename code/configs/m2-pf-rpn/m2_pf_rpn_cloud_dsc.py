"""Cloud config: dual-orbit SAR (asc + dsc) -> 19 input channels.

Same protocol as ``m2_pf_rpn_cloud_full.py`` (320x320, batch 4, AMP with
loss_scale=1.0, lr 1e-5, 8 epochs, seed 42) so the dual-orbit run is directly
comparable with the ascending-only runs.  Only the SAR modality changes: it
stacks 4 ascending + 4 descending pre/post channels.

``norm_stats.json`` must exist for this data root (19 channels in the order
optical(3), sar_pre(4), sar_post(4), topography(5), change(3)).
"""
import json as _json
import os as _os

_base_ = ['./m2_pf_rpn_cloud_full.py']

data_root = 'data/landslide_multimodal_dsc/'

_stats_path = _os.path.join(data_root, 'norm_stats.json')
if not _os.path.exists(_stats_path):
    raise FileNotFoundError(
        f'{_stats_path} is missing; run tools/data/compute_sen12_stats.py on '
        f'the dual-orbit conversion before using this config.')
with open(_stats_path, 'r', encoding='utf-8') as _fh:
    _stats = _json.load(_fh)
_fh.close()
del _fh  # MMEngine deep-copies module globals; file handles are not picklable
_mean = _stats['mean']
_std = _stats['std']
del _stats

modality_order = ('optical', 'sar', 'topography', 'change')
modality_channels = dict(optical=3, sar=8, topography=5, change=3)

train_pipeline = [
    dict(
        type='LoadMultiModalImageFromFile',
        modality_paths=dict(
            optical='optical_path',
            sar='sar_path',
            topography='topography_path',
            change='change_path'),
        modality_channels=modality_channels,
        stack_order=modality_order,
        color_type='unchanged',
        to_float32=True,
        allow_missing=True,
        channel_first=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(320, 320), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'sample_id', 'event_id', 'img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'flip',
                   'flip_direction', 'text', 'custom_entities',
                   'modality_channels', 'modality_order', 'modality_mask')),
]

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='manifests/train.json',
        pipeline=train_pipeline))

model = dict(
    modality_channels=modality_channels,
    modality_order=modality_order,
    sar_backbone=dict(in_channels=8),
    data_preprocessor=dict(mean=_mean, std=_std))

work_dir = 'work_dirs/m2_pf_rpn/cloud_dsc'