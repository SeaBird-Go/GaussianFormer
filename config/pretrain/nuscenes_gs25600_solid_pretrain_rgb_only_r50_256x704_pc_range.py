'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-05-02 20:37:22
Email: haimingzhang@link.cuhk.edu.cn
Description: Use ResNet 50 as backbone and use the image input is 256x704 and the 
pc range is [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0].
'''
_base_ = [
    '../_base_/misc.py',
    '../_base_/model.py',
    # './_base_/surroundocc.py'
]

# =========== data config ==============
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ_path = "data/surroundocc/samples"
batch_size = 1

render_size = (128, 352)  # (h, w)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    # dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="PrepapreImageInputs", img_size=render_size),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="PrepapreImageInputs", img_size=render_size),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

input_shape = (704, 256)
data_aug_conf = {
    "resize_lim": (0.44, 0.44),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

dataset_type = "NuScenesDataset"

train_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train',
    load_interval=4,
    return_keys=[
        'img',
        'projection_mat',  # lidar2img actually
        'image_wh',
        'cam_positions',
        'focal_positions',
        'K', 'inv_K',
        'target_imgs',
        'lidar2cam',
    ],
)

val_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val',
    return_keys=[
        'img',
        'projection_mat',  # lidar2img actually
        'image_wh',
        'cam_positions',
        'focal_positions',
        'K', 'inv_K',
        'target_imgs',
        'lidar2cam',
    ],
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=4
)


# =========== misc config ==============
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=2e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35
# ========= model config ===============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='PhotometricLoss',
            weight=1.0,
            input_dict=dict(
                pred_rgb='render_rgb',
                gt_rgb='target_imgs')
        )
    ])

loss_input_convertion = dict(
    render_rgb="render_rgb",
    render_depth="render_depth",
)

# ========= model config ===============
embed_dims = 128
num_decoder = 4
num_single_frame_decoder = 1
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
include_color = True
load_from = 'ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960_new.pth'
semantics = True
semantic_dim = 17

model = dict(
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        start_level=1),
    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        include_color=include_color,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            include_color=include_color,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=2,
                pc_range=pc_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            include_color=include_color,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.5, 0.5, 0.5],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    ),
    head=dict(
        type='GaussianReconHead',
        apply_loss_type='random_1',
        render_size=render_size,
        depth_range=[0.1, 64.0],
    )
)
