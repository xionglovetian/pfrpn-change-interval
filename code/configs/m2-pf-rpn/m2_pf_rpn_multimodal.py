_base_ = [
    '../pf-rpn/grounding_dino_swin-b_pf-rpn_base.py',
]

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.m2_pf_rpn',
        'mmdet.models.data_preprocessors.multimodal_preprocessor',
        'mmdet.datasets.multimodal_landslide',
    ],
    allow_failed_imports=False)

data_root = 'data/landslide_multimodal/'

# optical: pre-event RGB; sar: pre/post VV-VH; topography: DEM factors;
# change: precomputed event-change channels.
modality_order = ('optical', 'sar', 'topography', 'change')
modality_channels = dict(optical=3, sar=4, topography=5, change=3)
custom_classes = ('landslide', )

model = dict(
    type='M2PFRPN',
    num_pseudo_tokens=3,
    sp_thr=0.3,
    sp_iter_num=3,
    topk=3,
    candidate_name='landslide_candidate',
    modality_channels=modality_channels,
    modality_order=modality_order,
    use_sar=True,
    use_topography=True,
    use_change=True,
    use_modality_tokens=True,
    use_cross_modal_router=True,
    use_router=True,
    data_preprocessor=dict(
        type='MultiModalDetDataPreprocessor',
        mean=[
            905.173601, 1046.769567, 845.795848,
            -9.247413, -15.403178, -7.821083, -13.490451,
            1408.547824, 19.335547, 138.012568,
            0.000014, 1.743614, -0.059221, 0.051515, -0.025532,
        ],
        std=[
            1570.899165, 1565.830953, 1665.534114,
            2.859251, 2.930799, 2.720607, 2.847586,
            230.571459, 14.761215, 108.444818,
            0.030068, 1.819913, 0.324543, 0.282677, 0.200726,
        ],
        bgr_to_rgb=False,
        pad_mask=False,
    ),
    sar_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        in_channels=4,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=False),
    fusion_cfg=dict(
        in_channels=[
            [256, 512, 1024],
            [192, 384, 768],
            [256, 512, 1024],
            [256, 512, 1024],
        ],
        out_channels=[256, 512, 1024],
        token_dim=256,
        dropout=0.0),
    router_loss_weight=0.01,
    router_z_loss_weight=0.001,
    cross_modal_align_loss_weight=0.01,
    pseudo_token_diversity_loss_weight=0.001,
    missing_modality_loss_weight=0.01,
    modality_dropout_prob=0.1,
    bbox_head=dict(
        num_classes=1,
        contrastive_cfg=dict(max_text_len=3),
        rm_lang_embed=True,
        use_conf_branch=True,
        loss_ctr=dict(type='L1Loss', loss_weight=5.0),
    ),
)

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
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (640, 1333), (800, 1333)],
        keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'sample_id', 'event_id', 'img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'flip',
                   'flip_direction', 'text', 'custom_entities',
                   'modality_channels', 'modality_order', 'modality_mask')),
]

test_pipeline = [
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
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'sample_id', 'event_id', 'img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'text',
                   'custom_entities', 'modality_channels', 'modality_order',
                   'modality_mask')),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiModalLandslideDataset',
        data_root=data_root,
        ann_file='manifests/train.json',
        modality_keys=dict(
            optical_pre='optical_pre',
            optical_post='optical_post',
            sar_pre='sar_pre',
            sar_post='sar_post',
            topography='topography',
            change='change',
            valid_mask='valid_mask'),
        metainfo=dict(classes=custom_classes),
        filter_cfg=None,
        return_classes=True,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MultiModalLandslideDataset',
        data_root=data_root,
        ann_file='manifests/val.json',
        modality_keys=dict(
            optical_pre='optical_pre',
            optical_post='optical_post',
            sar_pre='sar_pre',
            sar_post='sar_post',
            topography='topography',
            change='change',
            valid_mask='valid_mask'),
        metainfo=dict(classes=custom_classes),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline))
test_dataloader = dict(val_dataloader)
test_dataloader['dataset'] = dict(val_dataloader['dataset'])
test_dataloader['dataset']['ann_file'] = 'manifests/test.json'

# M0 owns the final AR100/300/900 evaluator.  The placeholder keeps the
# standard MMDetection validation loop runnable.
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'manifests/val_coco.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone': dict(lr_mult=0.5),
            'modality_encoders': dict(lr_mult=1.0),
            'fusion': dict(lr_mult=1.0),
            'bbox_head': dict(lr_mult=1.0),
            'meta_net': dict(lr_mult=0.1),
            'router': dict(lr_mult=1.0),
            'learnable_text_embedding': dict(lr_mult=0.1),
            'absolute_pos_embed': dict(decay_mult=0.0),
            'decoder': dict(lr_mult=0.0),
        })))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(base_batch_size=4)
work_dir = 'work_dirs/m2_pf_rpn/multimodal'