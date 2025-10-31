import json
import copy
import argparse

code_before_meta = '''# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""
'''

METAINFO = {
    'classes':
    ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    # palette is a list of color tuples, which is used for visualization.
    'palette':
    [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
        (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
        (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
        (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
        (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
        (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
        (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
        (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
        (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
        (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
        (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
        (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
        (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
        (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
        (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
        (246, 0, 122), (191, 162, 208)]
}

subset_classes = {
    # "indoors": ('person', 'bird', 'cat', 'dog', 'backpack', 'umbrella',
    #      'handbag', 'tie', 'suitcase', 'bottle', 'wine glass', 'cup', 'fork',
    #      'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    #      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    #      'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    #      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    #      'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
}
classes = [{'supercategory': 'person', 'id': 1, 'name': 'person'},
 {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
 {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
 {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
 {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
 {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
 {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
 {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
 {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
 {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
 {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
 {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
 {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
 {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
 {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
 {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
 {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
 {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
 {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
 {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
 {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
 {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
 {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
 {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
 {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
 {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
 {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
 {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
 {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
 {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
 {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
 {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
 {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
 {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
 {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
 {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
 {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
 {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
 {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
 {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
 {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
 {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
 {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
 {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
 {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
 {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
 {'supercategory': 'food', 'id': 52, 'name': 'banana'},
 {'supercategory': 'food', 'id': 53, 'name': 'apple'},
 {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
 {'supercategory': 'food', 'id': 55, 'name': 'orange'},
 {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
 {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
 {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
 {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
 {'supercategory': 'food', 'id': 60, 'name': 'donut'},
 {'supercategory': 'food', 'id': 61, 'name': 'cake'},
 {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
 {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
 {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
 {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
 {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
 {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
 {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
 {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
 {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
 {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
 {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
 {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
 {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
 {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
 {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
 {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
 {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
 {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
 {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
 {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
 {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
 {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
 {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
 {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subset', type=str, help='target subset')
    args = parser.parse_args()

    coco_root = "data/coco"
    subset = args.subset
    if subset in set(c['supercategory'] for c in classes):
        target_classes = [c['name'] for c in classes if c['supercategory'] == subset]
    elif subset in set(c['name'] for c in classes):
        target_classes = [subset]
    else:
        assert False, f"Subset '{subset}' not found"
    if len(target_classes) == 0:
        assert subset in subset_classes, f"Subset '{subset}' not found"
        target_classes = subset_classes[subset]
    target_cids = [c['id'] for c in classes if c['name'] in target_classes]

    subset = subset.replace(' ', '')
    # create annotation files, with and without unavailable images (images don't contain target classes)
    for split in ['train', 'val']:
        ann_file = f"{coco_root}/annotations/instances_{split}2017.json"
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        ann_subset = copy.deepcopy(ann)
        ann_subset['avaliable_categories'] = [c for c in ann['categories'] if c['name'] in target_classes]
        ann_subset['annotations'] = [a for a in ann['annotations'] if a['category_id'] in target_cids]
        available_img_ids = set(a['image_id'] for a in ann_subset['annotations'])

        print(f"Subset '{subset}' in {split}2017: {len(available_img_ids)} / {len(ann['images'])}")

        # with unavailable images
        with open(f"{coco_root}/annotations/instances_{split}2017_{subset}.json", 'w') as f:
            json.dump(ann_subset, f)

        # without unavailable images
        ann_subset['images'] = [i for i in ann['images'] if i['id'] in available_img_ids]
        with open(f"{coco_root}/annotations/instances_{split}2017_{subset}_available.json", 'w') as f:
            json.dump(ann_subset, f)

    # generate dataset class file, replace METAINFO with target classes
    available_class_index = [i for i, c in enumerate(METAINFO['classes']) if c in target_classes]
    METAINFO['classes'] = target_classes
    METAINFO['palette'] = [METAINFO['palette'][i] for i in available_class_index]
    with open(f"data/coco/annotations/metainfo_{subset}.txt", 'w') as f:
        for c in METAINFO['classes']:
            f.write(f"{c}\n")
    with open(f"data/coco/annotations/palette_{subset}.txt", 'w') as f:
        for c in METAINFO['palette']:
            f.write(f"{c}\n")