# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'pretrained/svit/vit-adapter-t-3x.pth'  # noqa
model = dict(
    type='GumbelTwoStageDetector',
    ratio_loss_weight=4.,
    backbone=dict(
        _delete_=True,
        type='SelectiveViTAdapter',
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
        window_size=[None] * 12,
        pretrained=None,
        select_loc=[4, 5, 6, 7, 8, 9, 10, 11, 12],
        select_model_id=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        version=0,
        keep_ratio=[0.7, 0.7, 0.7, 0.49, 0.49, 0.49, 0.343, 0.343, 0.343],
        ratio_loss=True
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 192, 192, 192],
        out_channels=256,
        num_outs=5),
    init_cfg=dict(type='Pretrained', checkpoint=pretrained))


backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='RandomChoiceResize',
                      scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      keep_ratio=True)
             ],
             [
                 dict(type='RandomChoiceResize',
                      scales=[(400, 1333), (500, 1333), (600, 1333)],
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='RandomChoiceResize',
                      scales=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='PackDetInputs', 
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]
train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline))

max_epochs = 6
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[4],
        gamma=0.1)
]

optim_wrapper = dict(
    _delete_=True, 
    optimizer=dict(
        type='AdamW',
        lr=0.00001,
        weight_decay=0.000001,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'level_embed': dict(decay_mult=0.),
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }
    )
)
fp16 = None
work_dir = './work_dirs/svit/svit-adapter-t'