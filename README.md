# TaskTP
This repository is the official implementation of **"Task-Oriented Token Pruning for Efficient Object Detection and Segmentation"** published in **IROS 2025**.

## Installation

To install the required dependencies, run the following command:

```bash
# modify the prefix in environment.yaml as needed
conda env create -f environment.yaml
python -m pip install -e .
```

## Preparing Data and Models

Please refer to the mmdet repository for instructions on how to prepare datasets and pre-trained models: [mmdet](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).

## Training and Evaluation

Training:

```bash
./tools/dist_train.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final.py 8 --auto-scale-lr
```

Evaluation:

```bash
./tools/dist_test.sh configs/prune/topk/dynamic-mask-rcnn_vit-t-p4-w7_fpn_ms_1x_coco_gtguide_subset_topk-final.py your_checkpoint.pth 8
```

## Acknowledgements

This code is built upon the [mmdet](https://github.com/open-mmlab/mmdetection) and [SViT](https://github.com/uzh-rpg/svit) repositories. We thank the authors for their excellent work.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
