export MKL_THREADING_LAYER=GNU
# ./tools/dist_test.sh configs/prune/mask-rcnn_swin-t-p4-w7_fpn_1x_coco_test_animal.py work_dirs/mask-rcnn_swin-t-p4-w7_fpn_1x_coco_prune_animal/epoch_12.pth 8 
# ./tools/dist_test.sh configs/prune/mask-rcnn_swin-t-p4-w7_fpn_1x_coco_animal.py work_dirs/mask-rcnn_swin-t-p4-w7_fpn_1x_coco_animal/epoch_60.pth 8 --cfg-options test_dataloader.dataset.ann_file="annotations/instances_val2017_animal.json" test_evaluator.ann_file="data/coco/annotations/instances_val2017_animal.json"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_swin-t-p4-w7_fpn_3x_coco_prune.py work_dirs/dynamic-mask-rcnn_swin-t-p4-w7_fpn_3x_coco_prune/epoch_36.pth 8
# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 1 --cfg-options test_dataloader.dataset.type="CocoAnimalDataset" test_dataloader.dataset.ann_file="annotations/instances_val2017_animal_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_animal_available.json" test_evaluator.full_ann_file="data/coco/annotations/instances_val2017.json"
# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 8 --cfg-options test_dataloader.dataset.metainfo.classes="data/coco/annotations/metainfo_person.txt" test_dataloader.dataset.metainfo.palette="data/coco/annotations/palette_person.txt" test_dataloader.dataset.ann_file="annotations/instances_val2017_person_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_person_available.json"

# dynamic-fullset
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_6.pth
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove/epoch_6.pth

# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 8 --cfg-options model.backbone.remove_bg_by_gt=0.5 model.backbone.topk=-1

# dynamic-animal
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_6.pth --subset animal
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove/epoch_6.pth --subset animal

# dynamic-person
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_6.pth --subset person --topk 0.01 0.1 0.3 0.5 0.7 0.9 1.0
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove/epoch_6.pth --subset person
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 8 --cfg-options model.backbone.topk=1.0 test_dataloader.dataset.type="CocoPersonDataset" test_dataloader.dataset.ann_file="annotations/instances_val2017_person_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_person_available.json" test_evaluator.full_ann_file="data/coco/annotations/instances_val2017.json"

# dynamic-diningtable
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_6.pth --subset diningtable
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove/epoch_6.pth --subset diningtable
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_6.pth 8 --cfg-options model.backbone.topk=-1 # test_dataloader.dataset.metainfo.classes="data/coco/annotations/metainfo_diningtable.txt" test_dataloader.dataset.ann_file="annotations/instances_val2017_diningtable_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_diningtable_available.json"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_remove/epoch_6.pth 8 --cfg-options model.backbone.topk=-1 # test_dataloader.dataset.metainfo.classes="data/coco/annotations/metainfo_diningtable.txt" test_dataloader.dataset.ann_file="annotations/instances_val2017_diningtable_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_diningtable_available.json"

# dynamic-bottle
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable/epoch_6.pth 8 --cfg-options model.backbone.topk=0.1 test_dataloader.dataset.metainfo.classes="data/coco/annotations/metainfo_bottle.txt" test_dataloader.dataset.ann_file="annotations/instances_val2017_bottle_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_bottle_available.json"


# subset-diningtable-ft
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable/epoch_6.pth --subset diningtable
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_6.pth 8 --cfg-options model.backbone.topk=0.5
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable/epoch_1.pth 1 --cfg-options model.backbone.topk=0.5
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_diningtable/epoch_3.pth 8 --cfg-options model.backbone.topk=0.1

# subset-animal-ft
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_animal.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_animal/epoch_2.pth --subset animal
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_animal_remove.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_animal_remove/epoch_6.pth --subset animal

# subset-person-ft
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_person/epoch_1.pth --subset person --topk 0.01 0.1 0.3 0.5 0.7 0.9 1.0
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_person/epoch_4.pth --subset person --topk 0.01 0.1 0.3 0.5 0.7 0.9 1.0
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_person/epoch_1.pth 8 --cfg-options model.backbone.topk=0.01


# subset
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset/epoch_1.pth --topk 0.1
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset/epoch_3.pth --topk 0.1
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset/epoch_6.pth --topk 0.1








CUDA_VISIBLE_DEVICES=6 ./tools/dist_test.sh configs/svit/svit-adapter-t-0.5x-ftune.py pretrained/svit/svit-adapter-t-0.5x.pth 1 --cfg-options model.test_cfg.save_prefix="svit-t"


# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_lvis.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_lvis/epoch_1.pth 8


# Ours
# T
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_coco.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_coco/latest.pth 6 --cfg-options test_evaluator.outfile_prefix="work_dirs/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_coco/latest"
# ./tools/dist_test.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="full"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_car.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="car" work_dir="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/car"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_bottle.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="bottle" work_dir="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/bottle"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_animal.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="animal" work_dir="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/animal"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="person" work_dir="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/person"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="diningtable" work_dir="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/diningtable"

# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_noembed.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_noembed/epoch_6.pth --test-metrics map --visualize test --subset coco_handbag
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-s-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py work_dirs/dynamic-mask-rcnn_vit-s-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/epoch_4.pth --output work_dirs/dynamic-mask-rcnn_vit-s-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/exp2

# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-1.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-1/epoch_6.pth --test-metrics map --visualize top1
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth --test-metrics fps
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-3.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-3/epoch_6.pth --test-metrics map --visualize top3
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-4.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-4/epoch_6.pth --test-metrics map --visualize top4
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-5.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-5/epoch_6.pth --test-metrics map --visualize top5
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-10.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-10/epoch_6.pth --test-metrics map --visualize top10
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_relevant_only.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_relevant_only/epoch_6.pth --test-metrics map --subset all --visualize relevant_only
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_noembed.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_noembed/epoch_6.pth --test-metrics fps
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-20.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-20/epoch_5.pth --test-metrics fps

# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.1.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.2.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.3.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.4.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth --test-metrics fps --output work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/exp_detail
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth --test-metrics fps --subset coco_1-1,coco_1-2,coco_1-3,coco_1-4,coco_1-5 --output work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/base11.21
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.6.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth --test-metrics map --visualize 0.6thr
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.7.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.8.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-0.9.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth

# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-4layer.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-4layer/epoch_6.pth --test-metrics map --subset coco_motorcycle --visualize final

# OUrs T
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.1.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.1 --work-dir work_dirs/tiny-hyper-search/v7-0.1
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.2.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.2 --work-dir work_dirs/tiny-hyper-search/v7-0.2
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.3.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.3 --work-dir work_dirs/tiny-hyper-search/v7-0.3
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.4.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.4 --work-dir work_dirs/tiny-hyper-search/v7-0.4
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.5.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.5 --work-dir work_dirs/tiny-hyper-search/v7-0.5
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.6.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.6 --work-dir work_dirs/tiny-hyper-search/v7-0.6
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.7.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.7 --work-dir work_dirs/tiny-hyper-search/v7-0.7
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.8.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.8 --work-dir work_dirs/tiny-hyper-search/v7-0.8
# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final-0.9.py work_dirs/tiny-hyper-search/epoch_4.pth --test-metrics map,fps --output work_dirs/tiny-hyper-search/v7-0.9 --work-dir work_dirs/tiny-hyper-search/v7-0.9

# python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final.py work_dirs/tiny-hyper-search/epoch_6.pth --test-metrics map --output work_dirs/tiny-hyper-search2/v3
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ./tools/dist_test.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final.py work_dirs/tiny-hyper-search/epoch_6.pth 6 --cfg-options test_evaluator.outfile_prefix="work_dirs/tiny-hyper-search/v7/latest"


# Ours S
# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-s-p4-w7_fpn_ms_3x_coco.py work_dirs/mask-rcnn_vit-s-p4-w7_fpn_ms_3x_coco/latest.pth 8


# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_4.pth 8 --cfg-options model.test_cfg.save_prefix="full"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="full"
# CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_4.pth 6 --cfg-options model.test_cfg.save_prefix="person"
# CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_car.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_4.pth 6 --cfg-options model.test_cfg.save_prefix="car"
# CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_4.pth 6 --cfg-options model.test_cfg.save_prefix="diningtable"

# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_rule_car.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_rule/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="rule_car"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_rule_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_rule/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="rule_person"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_rule_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_rule/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="rule_diningtable"

# ./tools/dist_test.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="full"
# ./tools/dist_test.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_car.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="car"
# ./tools/dist_test.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth 8 --cfg-options model.test_cfg.save_prefix="person"
# ./tools/dist_test.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/epoch_2.pth 8 --cfg-options model.test_cfg.save_prefix="diningtable"



# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_woperson.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_4.pth 8 --cfg-options model.test_cfg.save_prefix="woperson"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_diningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/20241021_025445/epoch_6.pth 8
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_wodiningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth 8
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_woperson.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_woperson/epoch_6.pth 8
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_woperson.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_woperson/epoch_6.pth 8
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_woperson_fix.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_woperson_fix/epoch_1.pth 8

# person
# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 8 --cfg-options test_dataloader.dataset.metainfo.classes="data/coco/annotations/metainfo_person.txt" test_dataloader.dataset.metainfo.palette="data/coco/annotations/palette_person.txt" test_dataloader.dataset.ann_file="annotations/instances_val2017_person_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_person_available.json"
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide/epoch_6.pth --adaptive-topk --subset person
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth --adaptive-topk --subset person
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_person.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth 8

# wo-person
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_woperson.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth 8
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_wotable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth 8


# bottle
# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 8 --cfg-options test_dataloader.dataset.metainfo.classes="data/coco/annotations/metainfo_bottle.txt" test_dataloader.dataset.metainfo.palette="data/coco/annotations/palette_bottle.txt" test_dataloader.dataset.ann_file="annotations/instances_val2017_bottle_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_bottle_available.json"
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide/epoch_6.pth --adaptive-topk --subset bottle
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth --adaptive-topk --subset bottle

# diningtable
# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 8 --cfg-options test_dataloader.dataset.metainfo.classes="data/coco/annotations/metainfo_diningtable.txt" test_dataloader.dataset.ann_file="annotations/instances_val2017_diningtable_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_diningtable_available.json"
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide/epoch_6.pth --adaptive-topk --subset diningtable
# python tools/multi_test.py configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth --adaptive-topk --subset diningtable

# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset_wodiningtable.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/epoch_6.pth 8



# ./tools/dist_test.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_1x_lvis.py work_dirs/mask-rcnn_vit-t-p4-w7_fpn_1x_lvis/epoch_1.pth 8


# Ours Base
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python tools/multi_test.py configs/prune/mask-rcnn_vit-b-p4-w7_fpn_ms_3x_coco.py work_dirs/mask-rcnn_vit-b-p4-w7_fpn_ms_3x_coco/epoch_36.pth --test-metrics map --output work_dirs/mask-rcnn_vit-b-p4-w7_fpn_ms_3x_coco/maps
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python tools/multi_test.py configs/prune/topk/dynamic-mask-rcnn_vit-b-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py work_dirs/dynamic-mask-rcnn_vit-b-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/epoch_1.pth




# ATC
# python tools/multi_test.py configs/atc/tomeatc-adapter-b.py work_dirs/ATC/vit-b/mask-rcnn.pth --subset all
# python tools/multi_test.py configs/atc/tome-adapter-b.py work_dirs/ATC/vit-b/mask-rcnn.pth --subset all
