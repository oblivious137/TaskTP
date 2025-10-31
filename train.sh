# ./tools/dist_train.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_coco.py 8 --auto-scale-lr --cfg-options load_from="work_dirs/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_coco/latest.pth"
# ./tools/dist_train.sh configs/prune/mask-rcnn_vit-b-p4-w7_fpn_ms_3x_coco.py 8 --auto-scale-lr
# ./tools/dist_train.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset.py 8 --auto-scale-lr --cfg-options model.test_cfg.save_prefix="full" # load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/final_3.1/epoch_6.pth"
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py 8 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth" #model.test_cfg.save_prefix="full"
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py 8 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth" #model.test_cfg.save_prefix="full"
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_noembed.py 8 --auto-scale-lr --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth" #model.test_cfg.save_prefix="full"
# ./tools/dist_train.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_2.py 8 --auto-scale-lr --cfg-options model.test_cfg.save_prefix="full2" #load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/epoch_2.pth"
# ./tools/dist_train.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_3.py 8 --auto-scale-lr --cfg-options model.test_cfg.save_prefix="full3" load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_3/epoch_3.pth"
# ./tools/dist_train.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_4.py 8 --auto-scale-lr --cfg-options model.test_cfg.save_prefix="full4" #load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/epoch_2.pth"
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-4layer.py 8 --auto-scale-lr
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-1.py 8 --auto-scale-lr
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-3.py 8 --auto-scale-lr
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-5.py 8 --auto-scale-lr
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-4.py 8 --auto-scale-lr
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-10.py 8 --auto-scale-lr
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_relevant_only.py 8 --auto-scale-lr --resume

# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-20.py 8 --auto-scale-lr --resume

# T
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final.py 7 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth" #model.test_cfg.save_prefix="full"
./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final.py 4 --auto-scale-lr --cfg-options work_dir="work_dirs/tiny-hyper-search2" 


# S
# ./tools/dist_train.sh configs/prune/dynamic-mask-rcnn_vit-s-p4-w7_fpn_ms_1x_coco_gtguide_subset.py 8 --auto-scale-lr --cfg-options model.test_cfg.save_prefix="full"
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-s-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py 8 --auto-scale-lr --cfg-options model.test_cfg.save_prefix="full" load_from="work_dirs/dynamic-mask-rcnn_vit-s-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/v1/epoch_4.pth"

# B
# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-b-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk.py 8 --auto-scale-lr


# ./tools/dist_train.sh configs/svit/svit-adapter-t-0.5x-ftune.py 8 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/v1/epoch_6.pth"

# ATC
# ./tools/dist_train.sh configs/atc/tomeatc-adapter-t-0.5x-ftune.py 8 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/v1/epoch_6.pth"
# ./tools/dist_train.sh configs/atc/tomeatc-adapter-s-0.33x-ftune.py 8 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/v1/epoch_6.pth"
# ./tools/dist_train.sh configs/algm/adapter-t-0.5x-ftune.py 8 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset/v1/epoch_6.pth"


# ./tools/dist_train.sh configs/prune/mask-rcnn_vit-t-p4-w7_fpn_ms_3x_lvis.py 8 --auto-scale-lr



# ./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-b-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk_noembed.py 8 --auto-scale-lr # --cfg-options load_from="work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk/nosum2/epoch_6.pth" #model.test_cfg.save_prefix="full"
