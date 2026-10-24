_base_ = ['./m2_pf_rpn_multimodal.py']

# Local ablation dry-run only.  Uses the official random patch split and must
# not be reported as a paper result.  Final experiments require M0 event-level
# or region-level splits.
load_from = 'checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth'
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
    dict(type='FixScaleResize', scale=(320, 320), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'sample_id', 'event_id', 'img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'text',
                   'custom_entities', 'modality_channels', 'modality_order',
                   'modality_mask')),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=train_pipeline))

model = dict(encoder=dict(num_cp=0))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(lr=1e-5))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_interval=1000)

work_dir = 'work_dirs/m2_pf_rpn/ablations/base'
randomness = dict(seed=42)
