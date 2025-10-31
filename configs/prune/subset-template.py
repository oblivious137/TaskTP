test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='GenerateClassSubset', rule='coco_{subset}',
         classes="data/coco/annotations/metainfo.txt", dataset_classes="data/coco/annotations/metainfo_{subset}.txt"),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'class_subset'))
]

val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes='data/coco/annotations/metainfo_{subset}.txt',
                      palette='data/coco/annotations/palette_{subset}.txt'),
        ann_file="annotations/instances_val2017_{subset}_available.json",
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes='data/coco/annotations/metainfo_{subset}.txt',
                      palette='data/coco/annotations/palette_{subset}.txt'),
        ann_file="annotations/instances_val2017_{subset}_available.json",
        pipeline=test_pipeline))

val_evaluator=dict(
    type="CocoSubsetMetric",
    ann_file="data/coco/annotations/instances_val2017_{subset}_available.json"
)

test_evaluator=dict(
    type="CocoSubsetMetric",
    ann_file="data/coco/annotations/instances_val2017_{subset}_available.json"
)
