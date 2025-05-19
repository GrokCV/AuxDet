METAINFO = dict(
    classes=('Target', ), palette=[
        (
            128,
            0,
            0,
        ),
    ])
auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
data_root = '/data/dataset/syt/data/VOCdevkit/'
dataset_type = 'VSBWILDVOCDetDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
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
load_from = '/data/dataset/syt/PTH/0324/c+s_vsbd_e_oldalphas_c0/epoch_12.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        end_level=3,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type='AuxFPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.04,
                    0.04,
                    0.08,
                    0.08,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.5, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        num_stages=2,
        stages=[
            dict(
                adapt_cfg=dict(dilation=3, type='dilation'),
                anchor_generator=dict(
                    ratios=[
                        1.0,
                    ],
                    scales=[
                        8,
                    ],
                    strides=[
                        4,
                        8,
                    ],
                    type='AnchorGenerator'),
                bbox_coder=dict(
                    target_means=(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    target_stds=(
                        0.1,
                        0.1,
                        0.5,
                        0.5,
                    ),
                    type='DeltaXYWHBBoxCoder'),
                bridged_feature=True,
                feat_channels=256,
                in_channels=256,
                loss_bbox=dict(linear=True, loss_weight=7.0, type='IoULoss'),
                reg_decoded_bbox=True,
                type='StageCascadeRPNHead',
                with_cls=False),
            dict(
                adapt_cfg=dict(type='offset'),
                bbox_coder=dict(
                    target_means=(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    target_stds=(
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ),
                    type='DeltaXYWHBBoxCoder'),
                bridged_feature=False,
                feat_channels=256,
                in_channels=256,
                loss_bbox=dict(linear=True, loss_weight=7.0, type='IoULoss'),
                loss_cls=dict(
                    loss_weight=0.7, type='CrossEntropyLoss',
                    use_sigmoid=True),
                reg_decoded_bbox=True,
                type='StageCascadeRPNHead',
                with_cls=True),
        ],
        type='CascadeRPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.001),
        rpn=dict(
            max_per_img=300,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.65,
                neg_iou_thr=0.65,
                pos_iou_thr=0.65,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=[
            dict(
                allowed_border=-1,
                assigner=dict(
                    center_ratio=0.2, ignore_ratio=0.5, type='RegionAssigner'),
                debug=False,
                pos_weight=-1),
            dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
        ],
        rpn_proposal=dict(
            max_per_img=300,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
rpn_weight = 0.7
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='VOC2007/ImageSets/Main0/val.txt',
        ann_subdir='Annotations1',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2007/'),
        data_root='/data/dataset/syt/data/VOCdevkit/',
        img_subdir='PNGImages',
        metainfo=dict(classes=('Target', ), palette=[
            (
                128,
                0,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'width',
                    'height',
                    'view',
                    'band_type',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VSBWILDVOCDetDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(eval_mode='area', metric='mAP', type='VOCMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'width',
            'height',
            'view',
            'band_type',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='VOC2007/ImageSets/Main0/train.txt',
        ann_subdir='Annotations1',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2007/'),
        data_root='/data/dataset/syt/data/VOCdevkit/',
        filter_cfg=dict(bbox_min_size=0, filter_empty_gt=False, min_size=0),
        img_subdir='PNGImages',
        metainfo=dict(classes=('Target', ), palette=[
            (
                128,
                0,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'width',
                    'height',
                    'view',
                    'band_type',
                ),
                type='PackDetInputs'),
        ],
        type='VSBWILDVOCDetDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'width',
            'height',
            'view',
            'band_type',
        ),
        type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='VOC2007/ImageSets/Main0/val.txt',
        ann_subdir='Annotations1',
        backend_args=None,
        data_prefix=dict(sub_data_root='VOC2007/'),
        data_root='/data/dataset/syt/data/VOCdevkit/',
        img_subdir='PNGImages',
        metainfo=dict(classes=('Target', ), palette=[
            (
                128,
                0,
                0,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'width',
                    'height',
                    'view',
                    'band_type',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='VSBWILDVOCDetDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(eval_mode='area', metric='mAP', type='VOCMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './ts/'
