_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/voc_aux.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
rpn_weight = 0.7
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='AuxFPN',
        in_channels=[256, 512, 1024, 2048],
        # in_channels=[64, 128, 256, 512], # 18, 34
        out_channels=256,
        num_outs=4,
        start_level=0,
        end_level=3,
        ),
    rpn_head=dict(
        _delete_=True,
        type='CascadeRPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    # strides=[4, 8, 16, 32, 64]),
                    strides=[4, 8]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                bridged_feature=False,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))
        ]),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8]),
        bbox_head=dict(
            bbox_coder=dict(target_stds=[0.04, 0.04, 0.08, 0.08]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.5),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rpn_proposal=dict(max_per_img=300, nms=dict(iou_threshold=0.8)),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.65, neg_iou_thr=0.65, min_pos_iou=0.65),
            sampler=dict(type='RandomSampler', num=256))),
    test_cfg=dict(
        rpn=dict(max_per_img=300, nms=dict(iou_threshold=0.8)),
        rcnn=dict(score_thr=1e-3)))

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
)




