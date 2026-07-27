dataset_type = 'NuScenesOcc3DDataset'
dataset_root = 'data_origin/nuscenes/'
occ_root = 'data_origin/nuscenes/gts/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]
num_semantic_classes = len(occ_names)  # 17 (without empty)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]

# arch config
embed_dims = 256
num_layers = 5
num_query = 800
num_frames = 8
num_levels = 4
num_points = 4
num_refines = [2, 4, 8, 18, 36]
num_pt_channels = 32

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=0,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

model = dict(
    type='GUSD',
    use_grid_mask=False,
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='GUSDHead',
        num_classes=num_semantic_classes,   # 17 semantics classed (without empty)
        in_channels=embed_dims,
        num_query=num_query,
        pc_range=point_cloud_range,
        empty_label=num_semantic_classes,    # 17 = empty
        voxel_size=voxel_size,
        # Gaussian Splatting Parameters
        scale_range=[0.01, 2.4],
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=point_cloud_range[:3],
            grid_size=voxel_size[0],
        ),
        transformer=dict(
            type='GUSDTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_refines=num_refines,
            num_pt_channels=num_pt_channels,
            num_classes=num_semantic_classes,
            scales=[0.5],
            pc_range=point_cloud_range,
            use_gs_param_encoder=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5),
        loss_gs_cls=dict(
            type='MultiLoss',
            loss_cfgs=[
                dict(
                    type='OccupancyLoss',
                    weight=1.0,
                    empty_label=num_semantic_classes,          # 17 = empty
                    num_classes=num_semantic_classes,          # 17 (不含 empty)
                    balance_cls_weight=True,
                    use_focal_loss=False,
                    use_dice_loss=False,
                    use_lovasz_loss=True,
                    lovasz_ignore=num_semantic_classes,         # 17
                    lovasz_use_softmax=False,                   # aggregator 输出已是概率分布
                    use_sem_geo_scal_loss=False,
                    ignore_empty=False,
                    multi_loss_weights=dict(
                        loss_voxel_ce_weight=10.0,
                        loss_voxel_lovasz_weight=1.0),
                    manual_class_weight=[
                        1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502,
                        1.10087012, 1.26960524, 1.06258364, 1.189019,   1.06217292,
                        1.00595144, 0.85706115, 1.03923299, 0.90867526, 0.8936431,
                        0.85486129, 0.8527829
                    ]),
            ]),
        ),
    train_cfg=dict(
        pts=dict(max_gs_loss_samples=76800),
    ),
    test_cfg=dict(
        pts=dict(score_topk=76800),
    )
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = {
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadOcc3DFromFile', occ_root=occ_root), 
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=object_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='RandomTransformOcc', bda_aug_conf=bda_aug_conf),
    dict(type='DefaultFormatBundle3D', class_names=object_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2occ', 'ego2img',
        'ego2lidar', 'img_timestamp'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True, force_offline=False),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=object_names, with_label=False),
            dict(type='Collect3D', keys=['img'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp'))
        ])
]

data = dict(
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=5e-6,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)
total_epochs = 24
batch_size = 32

# load pretrained weights
load_from = 'pretrain_v3/opusv2_t_100e.pth'
revise_keys = None

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval=20, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=1, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=1)

# other flags
debug = False