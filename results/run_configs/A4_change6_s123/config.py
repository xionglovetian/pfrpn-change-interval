_mean = [
    1029.5342453632275,
    1079.4336683754002,
    863.1984791522611,
    -10.18005422358765,
    -17.324075998159785,
    -10.231501132774872,
    -18.23429157162832,
    670.8182674056893,
    15.3440504864745,
    117.1159613205886,
    5.874625648929065e-07,
    2.329307080777685,
    -0.015688989228987836,
    -0.0213319460430245,
    -0.07851966630286088,
    0.13970811786762777,
    -0.11653654212697553,
    0.054745577970949845,
]
_modality_channels = dict(change=6, optical=3, sar=4, topography=5)
_modality_order = (
    'optical',
    'sar',
    'topography',
    'change',
)
_stats_path = '/root/autodl-tmp/data_change6/norm_stats.json'
_std = [
    1726.2366983271515,
    1698.3143203905574,
    1761.615413933271,
    4.232201262342382,
    6.394913164598099,
    3.6253600375179915,
    7.13013461710153,
    719.8863912485601,
    14.247911873340039,
    113.38233961541157,
    0.023993401696555776,
    2.0896870326599877,
    0.22499136112504256,
    0.19411379954787653,
    0.228413854419202,
    0.35052120254789665,
    0.3314732651237574,
    0.2553263322001407,
]
auto_scale_lr = dict(base_batch_size=4, enable=False)
backend_args = None
custom_classes = ('landslide', )
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.models.detectors.m2_pf_rpn',
        'mmdet.models.data_preprocessors.multimodal_preprocessor',
        'mmdet.datasets.multimodal_landslide',
    ])
data_root = '/root/autodl-tmp/data_change6'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 8
modality_channels = dict(change=6, optical=3, sar=4, topography=5)
modality_order = (
    'optical',
    'sar',
    'topography',
    'change',
)
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=False,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=128,
        mlp_ratio=4,
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=dict(
        contrastive_cfg=dict(bias=False, log_scale=0.0, max_text_len=3),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_ctr=dict(loss_weight=5.0, type='L1Loss'),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        rm_lang_embed=True,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead',
        use_conf_branch=True),
    candidate_name='landslide_candidate',
    cross_modal_align_loss_weight=0.01,
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            1029.5342453632275,
            1079.4336683754002,
            863.1984791522611,
            -10.18005422358765,
            -17.324075998159785,
            -10.231501132774872,
            -18.23429157162832,
            670.8182674056893,
            15.3440504864745,
            117.1159613205886,
            5.874625648929065e-07,
            2.329307080777685,
            -0.015688989228987836,
            -0.0213319460430245,
            -0.07851966630286088,
            0.13970811786762777,
            -0.11653654212697553,
            0.054745577970949845,
        ],
        pad_mask=False,
        std=[
            1726.2366983271515,
            1698.3143203905574,
            1761.615413933271,
            4.232201262342382,
            6.394913164598099,
            3.6253600375179915,
            7.13013461710153,
            719.8863912485601,
            14.247911873340039,
            113.38233961541157,
            0.023993401696555776,
            2.0896870326599877,
            0.22499136112504256,
            0.19411379954787653,
            0.228413854419202,
            0.35052120254789665,
            0.3314732651237574,
            0.2553263322001407,
        ],
        type='MultiModalDetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=0,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    fusion_cfg=dict(
        dropout=0.0,
        in_channels=[
            [
                256,
                512,
                1024,
            ],
            [
                192,
                384,
                768,
            ],
            [
                256,
                512,
                1024,
            ],
            [
                256,
                512,
                1024,
            ],
        ],
        out_channels=[
            256,
            512,
            1024,
        ],
        token_dim=256),
    missing_modality_loss_weight=0.01,
    modality_channels=dict(change=6, optical=3, sar=4, topography=5),
    modality_dropout_prob=0.1,
    modality_order=(
        'optical',
        'sar',
        'topography',
        'change',
    ),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            256,
            512,
            1024,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_pseudo_tokens=3,
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    pseudo_token_diversity_loss_weight=0.001,
    router_loss_weight=0.01,
    router_z_loss_weight=0.001,
    sar_backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=False,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        in_channels=4,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=True),
    sp_iter_num=3,
    sp_thr=0.3,
    test_cfg=dict(max_per_img=900),
    topk=3,
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='M2PFRPN',
    use_change=True,
    use_cross_modal_router=True,
    use_modality_tokens=True,
    use_router=True,
    use_sar=True,
    use_topography=True,
    with_box_refine=True)
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    loss_scale=1.0,
    optimizer=dict(lr=1e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.5),
            bbox_head=dict(lr_mult=1.0),
            decoder=dict(lr_mult=0.0),
            fusion=dict(lr_mult=1.0),
            learnable_text_embedding=dict(lr_mult=0.1),
            meta_net=dict(lr_mult=0.1),
            modality_encoders=dict(lr_mult=1.0),
            router=dict(lr_mult=1.0))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=8,
        gamma=0.1,
        milestones=[
            6,
            7,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=123)
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(
        allow_missing=True,
        channel_first=False,
        color_type='unchanged',
        modality_channels=dict(change=3, optical=3, sar=4, topography=5),
        modality_paths=dict(
            change='change_path',
            optical='optical_path',
            sar='sar_path',
            topography='topography_path'),
        stack_order=(
            'optical',
            'sar',
            'topography',
            'change',
        ),
        to_float32=True,
        type='LoadMultiModalImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        1333,
    ), type='FixScaleResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'sample_id',
            'event_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
            'modality_channels',
            'modality_order',
            'modality_mask',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=8, type='EpochBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='manifests/train.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='/root/autodl-tmp/data_change6',
        filter_cfg=None,
        metainfo=dict(classes=('landslide', )),
        modality_keys=dict(
            change='change',
            optical_post='optical_post',
            optical_pre='optical_pre',
            sar_post='sar_post',
            sar_pre='sar_pre',
            topography='topography',
            valid_mask='valid_mask'),
        pipeline=[
            dict(
                allow_missing=True,
                channel_first=False,
                color_type='unchanged',
                modality_channels=dict(
                    change=6, optical=3, sar=4, topography=5),
                modality_paths=dict(
                    change='change_path',
                    optical='optical_path',
                    sar='sar_path',
                    topography='topography_path'),
                stack_order=(
                    'optical',
                    'sar',
                    'topography',
                    'change',
                ),
                to_float32=True,
                type='LoadMultiModalImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(keep_ratio=True, scale=(
                320,
                320,
            ), type='FixScaleResize'),
            dict(
                meta_keys=(
                    'img_id',
                    'sample_id',
                    'event_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                    'modality_channels',
                    'modality_order',
                    'modality_mask',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        type='MultiModalLandslideDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        allow_missing=True,
        channel_first=False,
        color_type='unchanged',
        modality_channels=dict(change=6, optical=3, sar=4, topography=5),
        modality_paths=dict(
            change='change_path',
            optical='optical_path',
            sar='sar_path',
            topography='topography_path'),
        stack_order=(
            'optical',
            'sar',
            'topography',
            'change',
        ),
        to_float32=True,
        type='LoadMultiModalImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(keep_ratio=True, scale=(
        320,
        320,
    ), type='FixScaleResize'),
    dict(
        meta_keys=(
            'img_id',
            'sample_id',
            'event_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'custom_entities',
            'modality_channels',
            'modality_order',
            'modality_mask',
        ),
        type='PackDetInputs'),
]
val_cfg = None
val_dataloader = None
val_evaluator = None
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/root/autodl-tmp/work_dirs/w1/A4_change6_s123'
