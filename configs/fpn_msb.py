# model settings
seed = 1024
num_sclies_one_img=3
cfg_3dce = dict(
    samples_per_gpu=20,
    num_slices=3,
    num_images_3dce=1,
    img_do_clip=True,
    windowing=[-1024, 3071],
    norm_spacing=0.8,
    max_size=512,  #to do
    slice_intervals=2.,
    image_path='./datasets/deeplesion/Images',
    val_avg_fp=[0.5, 1, 2, 4, 8, 16],
    val_iou_th=0.5,
)
model = dict(
    type='FasterRCNN',
    pretrained='./pretrained/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        style='pytorch',
        norm_eval=False,
        normalize=dict(
            type='BN',
            frozen=False),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        normalize=dict(type='BN'),
        activation='relu',
        msb=True,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[2, 4, 6, 8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False))
# model training and testing settings
train_cfg = dict(

    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=48,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.4,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=48,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=10)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'DeepLesion'
data_root = './datasets/deeplesion/'
img_norm_cfg = dict(
    mean=[127.5]*num_sclies_one_img, std=[127.5]*num_sclies_one_img, to_rgb=False)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        cfg_3dce=cfg_3dce,
        ann_file=data_root + 'deep_lesion_train.json',
        img_prefix=data_root + 'Images/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        cfg_3dce=cfg_3dce,
        ann_file=data_root + 'deep_lesion_val.json',
        img_prefix=data_root + 'Images/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        test_mode=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        cfg_3dce=cfg_3dce,
        ann_file=data_root + 'deep_lesion_test.json',
        img_prefix=data_root + 'Images/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
clip_grad=dict(max_norm=35, norm_type=2)
lr_scedule = dict(
    type='CosineAnnealingLR',
    T_max=12,
    T_mul=1,
    decay_rate=0.5,
    T_min=0.00001
)
# runtime settings
start_epoch = 0
end_epoch = 400
device_ids = [0]
print_interval=50
work_dir = './deeplesion/fpn_msb'
load_from = None
resume_from = None
model_info = ''
eval_val=False
