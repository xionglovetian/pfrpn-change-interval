_base_ = ['./m2_pf_rpn_multimodal.py']

# Local proposal export config.  The official random patch split is used only
# for pipeline validation, not for paper metrics.
model = dict(encoder=dict(num_cp=0))

_modality_order = ('optical', 'sar', 'topography', 'change')
_modality_channels = dict(optical=3, sar=4, topography=5, change=3)

test_pipeline = [
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
    dict(type='FixScaleResize', scale=(320, 320), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'sample_id', 'event_id', 'img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'text',
                   'custom_entities', 'modality_channels', 'modality_order',
                   'modality_mask')),
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))