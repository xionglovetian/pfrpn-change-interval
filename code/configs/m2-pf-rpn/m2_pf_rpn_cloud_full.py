"""Cloud full-data M1 config (event-level split, A0-A9 ablations).

Engineering notes
-----------------
* Resolution 320x320: native patches are 128x128, so 320 is a 2.5x
  upsample that keeps Swin-B token counts and runtime tractable.
* ``data_preprocessor`` normalisation is read from
  ``data/landslide_multimodal_full/norm_stats.json`` produced by
  ``tools/data/compute_sen12_stats.py``; a fallback is kept so that the
  config can also be built before conversion finishes.
* Validation during training is disabled because M0 owns the final
  AR100/300/900 evaluator; metrics are produced by exporting proposals and
  scoring them offline.
"""
import json as _json
import os as _os

_base_ = ['./m2_pf_rpn_multimodal.py']

data_root = 'data/landslide_multimodal_hn_lb/'
load_from = 'checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth'

_stats_path = _os.path.join(data_root, 'norm_stats.json')
if _os.path.exists(_stats_path):
    with open(_stats_path, 'r', encoding='utf-8') as _fh:
        _stats = _json.load(_fh)
    _fh.close()
    del _fh  # MMEngine deep-copies module globals; file handles are not picklable
    del _stats
    _mean = _json.load(open(_stats_path, 'r', encoding='utf-8'))['mean']
    _std = _json.load(open(_stats_path, 'r', encoding='utf-8'))['std']
else:  # fallback: statistics of the 223-patch chimanimani subset
    _mean = [
        905.173601, 1046.769567, 845.795848,
        -9.247413, -15.403178, -7.821083, -13.490451,
        1408.547824, 19.335547, 138.012568,
        0.000014, 1.743614, -0.059221, 0.051515, -0.025532,
    ]
    _std = [
        1570.899165, 1565.830953, 1665.534114,
        2.859251, 2.930799, 2.720607, 2.847586,
        230.571459, 14.761215, 108.444818,
        0.030068, 1.819913, 0.324543, 0.282677, 0.200726,
    ]

val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

_modality_order = ('optical', 'sar', 'topography', 'change')
_modality_channels = dict(optical=3, sar=4, topography=5, change=3)

train_pipeline = [
    dict(
        type='LoadMultiModalImageFromFile',
        modality_paths=dict(
            optical='optical_path',
            sar='sar_path',
            topography='topography_path',
            change='change_path'),
        modality_channels=_modality_channels,
        stack_order=_modality_order,
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
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiModalLandslideDataset',
        data_root=data_root,
        ann_file='manifests/train.json',
        pipeline=train_pipeline))

model = dict(
    encoder=dict(num_cp=0),
    data_preprocessor=dict(mean=_mean, std=_std))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=1.0,
    optimizer=dict(lr=1e-5))

max_epochs = 8
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[6, 7],
        gamma=0.1),
]

# Keep only the most recent checkpoint: each epoch file is ~2.4 GB and ten
# ablations would otherwise need ~190 GB of disk.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1000)
work_dir = 'work_dirs/m2_pf_rpn/cloud_full'
randomness = dict(seed=42)