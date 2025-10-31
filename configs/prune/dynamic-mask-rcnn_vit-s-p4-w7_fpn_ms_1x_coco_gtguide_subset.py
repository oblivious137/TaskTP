_base_ = [
    'mask-rcnn_vit-s-p4-w7_fpn_ms_3x_coco.py',
] 
pretrained = 'work_dirs/mask-rcnn_vit-s-p4-w7_fpn_ms_3x_coco/latest.pth'  # noqa
model = dict(
    type='DynamicMaskRCNN',
    backbone=dict(
        type='DynamicViTAdapter',
        dynamic_cfg=dict(
            prune_action='skip',
            classwise=True,
            gumbel_soft=False,
            inherit_score=False,
            straight_forward=True,
            detach_before_classifier=False,
            detach_before_mask_mlp=True,
            rule_base=False,
            num_categories=80,
        ),
        prune_loss=dict(
            type='SemanticLoss',
            semantic_weight=1.0,
            ratio_weight=2.0,
            prune_ratio_panalty=0.0,
        ),
        prune_layers=(3,6,9,),
        keep_ratios=(0.0, 0.0, 0.0),
    ),
    train_cfg=dict(
        rcnn=dict(assigner=dict(ignore_iof_thr=0.3)),
        rpn=dict(assigner=dict(ignore_iof_thr=0.3)),
    ),
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
                      scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                                 (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                                 (1333, 736), (1333, 768), (1333, 800)],
                      keep_ratio=True)
             ],
             [
                 dict(type='RandomChoiceResize',
                      scales=[(1333, 400), (1333, 500), (1333, 600)],
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='RandomChoiceResize',
                      scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                                 (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                                 (1333, 736), (1333, 768), (1333, 800)],
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='GenerateClassSubset', rule='random', classes="data/coco/annotations/metainfo.txt", subset_size=(1,8,20,40,80), ignore=True),
    dict(type='PackDetInputs', 
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'class_subset'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='GenerateClassSubset', rule='all', classes="data/coco/annotations/metainfo.txt"),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'class_subset'))
]
train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline))

max_epochs = 4
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
        milestones=[3],
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
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))

# find_unused_parameters=True