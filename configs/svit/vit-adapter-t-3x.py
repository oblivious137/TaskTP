# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapterSViT',
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        drop_path_rate=0.1,
        layer_scale=False,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[False] * 12,
        window_size=[None] * 12),
    neck=dict(
        type='FPN',
        in_channels=[192, 192, 192, 192],
        out_channels=256,
        num_outs=5))
fp16 = None
work_dir = './work_dirs/svit/vit-adapter-t'