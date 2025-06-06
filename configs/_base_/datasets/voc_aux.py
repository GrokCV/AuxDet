# dataset settings
dataset_type = "VSBWILDVOCDetDataset"
data_root = "/data/dataset/syt/data/VOCdevkit/"

METAINFO = {
    "classes": ("Target",),
    "palette": [
        (128, 0, 0),
    ],
}

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically Infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/segmentation/VOCdevkit/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#         'data/': 's3://openmmlab/datasets/segmentation/'
#     }))
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs",
     meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor",
                "width", "height", "view", "band_type"),
),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor",
                   "width", "height", "view", "band_type"),
    ),
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=METAINFO,
            ann_file='VOC2007/ImageSets/Main0/train.txt',
            img_subdir="PNGImages",
            ann_subdir="Annotations1",
            data_prefix=dict(sub_data_root='VOC2007/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
            pipeline=train_pipeline,
            backend_args=backend_args,
        ),
)

val_dataloader = dict(
    batch_size=4,#1
    num_workers=4,#2
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='VOC2007/ImageSets/Main0/val.txt',
        img_subdir="PNGImages",
        ann_subdir="Annotations1",
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL VOC2012 defaults to use 'area'.
val_evaluator = dict(type="VOCMetric", metric="mAP", eval_mode="area")
# val_evaluator = dict(type="VOCMetric", metric="mAP", eval_mode="11points")
test_evaluator = val_evaluator
