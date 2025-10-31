import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import json
import numpy as np
import mmengine
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale

rule_dict = {
    "coco_1-1": ["car"],
    "coco_1-2": ["zebra"],
    "coco_1-3": ["truck"],
    "coco_1-4": ["handbag"],
    "coco_1-5": ["refrigerator"],
    "coco_10-1": ["orange", "person", "airplane", "bottle", "donut", "cell phone", "wine glass", "snowboard", "fire hydrant", "oven"],
    "coco_10-2": ["vase", "handbag", "surfboard", "suitcase", "baseball bat", "carrot", "fire hydrant", "mouse", "knife", "car"],
    "coco_10-3": ["tv", "chair", "teddy bear", "mouse", "tennis racket", "airplane", "train", "zebra", "kite", "bottle"],
    "coco_10-4": ["bed", "fork", "hot dog", "horse", "spoon", "bus", "cup", "mouse", "sink", "broccoli"],
    "coco_10-5": ["dining table", "orange", "book", "cow", "bird", "microwave", "person", "elephant", "frisbee", "sandwich"],
    "coco_40-1": ["apple", "bowl", "vase", "banana", "cup", "potted plant", "oven", "hair drier", "umbrella", "baseball bat", "clock", "microwave", "parking meter", "chair", "refrigerator", "scissors", "skis", "donut", "sink", "surfboard", "stop sign", "giraffe", "book", "toaster", "mouse", "couch", "spoon", "snowboard", "sports ball", "bus", "airplane", "remote", "cow", "boat", "cell phone", "tie", "handbag", "teddy bear", "sheep", "kite"],
    "coco_40-2": ["motorcycle", "boat", "fire hydrant", "scissors", "surfboard", "snowboard", "tennis racket", "hair drier", "orange", "tie", "giraffe", "cell phone", "chair", "bed", "spoon", "book", "bicycle", "oven", "couch", "skis", "truck", "donut", "refrigerator", "elephant", "toilet", "person", "apple", "suitcase", "potted plant", "zebra", "umbrella", "toaster", "knife", "sink", "bench", "backpack", "broccoli", "bus", "wine glass", "car"],
    "coco_40-3": ["frisbee", "potted plant", "snowboard", "microwave", "giraffe", "surfboard", "bowl", "donut", "bicycle", "orange", "bench", "carrot", "horse", "elephant", "hair drier", "skateboard", "refrigerator", "umbrella", "traffic light", "clock", "fire hydrant", "suitcase", "cell phone", "tv", "stop sign", "toaster", "car", "wine glass", "truck", "keyboard", "apple", "sports ball", "broccoli", "hot dog", "bottle", "sheep", "airplane", "vase", "baseball glove", "laptop"],
    "coco_40-4": ["dining table", "teddy bear", "tennis racket", "baseball glove", "fire hydrant", "bear", "parking meter", "banana", "handbag", "scissors", "mouse", "microwave", "stop sign", "oven", "tie", "baseball bat", "cat", "horse", "laptop", "tv", "knife", "hair drier", "toilet", "toaster", "potted plant", "chair", "sheep", "couch", "surfboard", "umbrella", "bench", "apple", "cow", "bowl", "truck", "toothbrush", "cell phone", "backpack", "bus", "bicycle"],
    "coco_40-5": ["tv", "motorcycle", "donut", "cell phone", "frisbee", "backpack", "bear", "apple", "fork", "truck", "snowboard", "horse", "dog", "bowl", "knife", "cup", "car", "refrigerator", "dining table", "sports ball", "bicycle", "bench", "handbag", "cat", "stop sign", "bus", "mouse", "keyboard", "baseball bat", "banana", "bed", "bottle", "hot dog", "book", "oven", "fire hydrant", "potted plant", "tennis racket", "pizza", "boat"],
    "coco_2-1": ['hot dog', 'bird'],
    "coco_2-2": ['bird', 'fire hydrant'],
    "coco_2-3": ['bench', 'elephant'],
    "coco_2-4": ['bench', 'horse'],
    "coco_2-5": ['suitcase', 'fire hydrant'],
    "coco_3-1": ['bird', 'baseball glove', 'motorcycle'],
    "coco_3-2": ['laptop', 'potted plant', 'baseball bat'],
    "coco_3-3": ['tie', 'tennis racket', 'fork'],
    "coco_3-4": ['chair', 'sports ball', 'banana'],
    "coco_3-5": ['toilet', 'car', 'pizza'],
    "coco_4-1": ['motorcycle', 'broccoli', 'car', 'tennis racket'],
    "coco_4-2": ['boat', 'spoon', 'bottle', 'potted plant'],
    "coco_4-3": ['tv', 'hair drier', 'sink', 'train'],
    "coco_4-4": ['bear', 'train', 'sandwich', 'sports ball'],
    "coco_4-5": ['airplane', 'giraffe', 'parking meter', 'handbag'],
    "coco_5-1": ['mouse', 'bench', 'sink', 'stop sign', 'couch'],
    "coco_5-2": ['sandwich', 'toilet', 'dog', 'baseball bat', 'fire hydrant'],
    "coco_5-3": ['knife', 'book', 'tv', 'carrot', 'remote'],
    "coco_5-4": ['hot dog', 'bird', 'fork', 'scissors', 'baseball bat'],
    "coco_5-5": ['cell phone', 'tie', 'teddy bear', 'fire hydrant', 'toothbrush'],
    "coco_6-1": ['carrot', 'sink', 'vase', 'dining table', 'toothbrush', 'laptop'],
    "coco_6-2": ['teddy bear', 'donut', 'hair drier', 'backpack', 'boat', 'clock'],
    "coco_6-3": ['tennis racket', 'cat', 'knife', 'truck', 'backpack', 'snowboard'],
    "coco_6-4": ['toothbrush', 'giraffe', 'couch', 'hair drier', 'umbrella', 'skateboard'],
    "coco_6-5": ['toilet', 'motorcycle', 'carrot', 'horse', 'fork', 'baseball glove'],
    "coco_7-1": ['fork', 'bird', 'traffic light', 'book', 'couch', 'umbrella', 'airplane'],
    "coco_7-2": ['sink', 'book', 'tv', 'elephant', 'snowboard', 'backpack', 'toothbrush'],
    "coco_7-3": ['donut', 'elephant', 'sheep', 'apple', 'banana', 'cup', 'dining table'],
    "coco_7-4": ['book', 'backpack', 'giraffe', 'mouse', 'wine glass', 'apple', 'bottle'],
    "coco_7-5": ['vase', 'couch', 'tie', 'cup', 'stop sign', 'hair drier', 'baseball glove'],
    "coco_8-1": ['zebra', 'donut', 'teddy bear', 'orange', 'scissors', 'tennis racket', 'suitcase', 'traffic light'],
    "coco_8-2": ['tie', 'book', 'baseball bat', 'potted plant', 'teddy bear', 'car', 'elephant', 'orange'],
    "coco_8-3": ['bird', 'frisbee', 'skis', 'refrigerator', 'cow', 'pizza', 'boat', 'parking meter'],
    "coco_8-4": ['scissors', 'train', 'bird', 'frisbee', 'mouse', 'tennis racket', 'laptop', 'banana'],
    "coco_8-5": ['skateboard', 'bench', 'orange', 'baseball glove', 'tv', 'wine glass', 'chair', 'bed'],
    "coco_9-1": ['book', 'bench', 'tie', 'giraffe', 'clock', 'stop sign', 'skis', 'knife', 'laptop'],
    "coco_9-2": ['baseball bat', 'skis', 'bed', 'scissors', 'boat', 'donut', 'broccoli', 'surfboard', 'traffic light'],
    "coco_9-3": ['cat', 'wine glass', 'cell phone', 'knife', 'backpack', 'zebra', 'sheep', 'cup', 'toaster'],
    "coco_9-4": ['potted plant', 'cake', 'skis', 'tennis racket', 'sink', 'sports ball', 'hair drier', 'person', 'banana'],
    "coco_9-5": ['orange', 'remote', 'traffic light', 'apple', 'boat', 'skateboard', 'toothbrush', 'cat', 'tennis racket'],
    "coco_10-1": ['donut', 'train', 'wine glass', 'clock', 'banana', 'cat', 'sandwich', 'tennis racket', 'carrot', 'fork'],
    "coco_10-2": ['hair drier', 'parking meter', 'cat', 'sandwich', 'tie', 'bench', 'bear', 'baseball glove', 'oven', 'spoon'],
    "coco_10-3": ['laptop', 'microwave', 'clock', 'sheep', 'fire hydrant', 'spoon', 'sandwich', 'train', 'wine glass', 'traffic light'],
    "coco_10-4": ['skateboard', 'potted plant', 'knife', 'laptop', 'zebra', 'sheep', 'bowl', 'person', 'orange', 'couch'],
    "coco_10-5": ['baseball glove', 'chair', 'couch', 'skis', 'hot dog', 'dining table', 'tv', 'refrigerator', 'boat', 'potted plant'],
    "coco_15-1": ['oven', 'truck', 'suitcase', 'fork', 'remote', 'bird', 'traffic light', 'broccoli', 'toaster', 'elephant', 'person', 'refrigerator', 'carrot', 'backpack', 'vase'],
    "coco_15-2": ['couch', 'tennis racket', 'chair', 'sink', 'frisbee', 'wine glass', 'motorcycle', 'baseball glove', 'bottle', 'backpack', 'refrigerator', 'orange', 'vase', 'broccoli', 'fire hydrant'],
    "coco_15-3": ['dog', 'bowl', 'backpack', 'spoon', 'remote', 'scissors', 'baseball bat', 'umbrella', 'handbag', 'carrot', 'truck', 'cup', 'sandwich', 'bottle', 'bird'],
    "coco_15-4": ['cow', 'zebra', 'scissors', 'toothbrush', 'clock', 'kite', 'dining table', 'knife', 'skateboard', 'airplane', 'bed', 'snowboard', 'boat', 'oven', 'tie'],
    "coco_15-5": ['parking meter', 'potted plant', 'orange', 'baseball glove', 'suitcase', 'motorcycle', 'airplane', 'bench', 'bottle', 'vase', 'skis', 'apple', 'truck', 'bus', 'frisbee'],
    "coco_25-1": ['apple', 'cup', 'hot dog', 'dog', 'cell phone', 'potted plant', 'mouse', 'bowl', 'hair drier', 'fork', 'scissors', 'horse', 'tennis racket', 'sheep', 'remote', 'bird', 'skateboard', 'tv', 'baseball bat', 'couch', 'skis', 'carrot', 'motorcycle', 'broccoli', 'bear'],
    "coco_25-2": ['spoon', 'frisbee', 'skis', 'couch', 'hot dog', 'snowboard', 'sink', 'bicycle', 'kite', 'refrigerator', 'banana', 'book', 'person', 'oven', 'skateboard', 'baseball glove', 'orange', 'vase', 'toothbrush', 'toilet', 'train', 'tennis racket', 'elephant', 'airplane', 'truck'],
    "coco_25-3": ['suitcase', 'cup', 'couch', 'knife', 'frisbee', 'toaster', 'cake', 'dog', 'clock', 'scissors', 'remote', 'fire hydrant', 'sheep', 'horse', 'laptop', 'vase', 'bicycle', 'car', 'skateboard', 'oven', 'broccoli', 'motorcycle', 'bear', 'cat', 'pizza'],
    "coco_25-4": ['cell phone', 'chair', 'laptop', 'sink', 'vase', 'potted plant', 'tv', 'bear', 'bus', 'fork', 'skis', 'sheep', 'cup', 'stop sign', 'horse', 'skateboard', 'bird', 'apple', 'banana', 'bed', 'dining table', 'cat', 'keyboard', 'scissors', 'broccoli'],
    "coco_25-5": ['chair', 'cell phone', 'keyboard', 'cup', 'snowboard', 'potted plant', 'clock', 'skis', 'mouse', 'hair drier', 'carrot', 'bottle', 'stop sign', 'orange', 'hot dog', 'banana', 'bowl', 'tie', 'scissors', 'parking meter', 'oven', 'sports ball', 'airplane', 'surfboard', 'baseball bat'],
    "coco_30-1": ['motorcycle', 'bicycle', 'couch', 'microwave', 'baseball bat', 'oven', 'airplane', 'vase', 'cup', 'skateboard', 'book', 'carrot', 'potted plant', 'bird', 'person', 'skis', 'tennis racket', 'spoon', 'suitcase', 'laptop', 'parking meter', 'sheep', 'surfboard', 'apple', 'cat', 'scissors', 'sandwich', 'stop sign', 'handbag', 'orange'],
    "coco_30-2": ['microwave', 'horse', 'bicycle', 'handbag', 'zebra', 'motorcycle', 'bear', 'backpack', 'dining table', 'traffic light', 'orange', 'dog', 'giraffe', 'truck', 'book', 'chair', 'tie', 'sink', 'snowboard', 'tennis racket', 'laptop', 'bottle', 'parking meter', 'remote', 'cake', 'car', 'baseball glove', 'boat', 'bus', 'scissors'],
    "coco_30-3": ['bottle', 'stop sign', 'pizza', 'car', 'spoon', 'chair', 'person', 'banana', 'skis', 'baseball bat', 'sink', 'bird', 'cake', 'fire hydrant', 'tennis racket', 'oven', 'cat', 'knife', 'dog', 'backpack', 'skateboard', 'horse', 'dining table', 'toaster', 'clock', 'hair drier', 'scissors', 'bed', 'bear', 'remote'],
    "coco_30-4": ['hot dog', 'tennis racket', 'clock', 'boat', 'car', 'toilet', 'dog', 'bird', 'toothbrush', 'laptop', 'bed', 'tv', 'spoon', 'wine glass', 'couch', 'bench', 'umbrella', 'toaster', 'remote', 'giraffe', 'oven', 'cow', 'dining table', 'orange', 'baseball glove', 'bear', 'bottle', 'refrigerator', 'suitcase', 'sheep'],
    "coco_30-5": ['baseball glove', 'surfboard', 'elephant', 'hot dog', 'bird', 'bicycle', 'wine glass', 'umbrella', 'cake', 'clock', 'vase', 'dog', 'refrigerator', 'bench', 'sandwich', 'snowboard', 'baseball bat', 'knife', 'parking meter', 'book', 'donut', 'cell phone', 'frisbee', 'pizza', 'tennis racket', 'spoon', 'banana', 'fork', 'remote', 'keyboard'],
    "coco_35-1": ['snowboard', 'keyboard', 'hair drier', 'oven', 'tie', 'skis', 'baseball glove', 'pizza', 'bird', 'fork', 'remote', 'toilet', 'sports ball', 'bowl', 'sink', 'wine glass', 'scissors', 'vase', 'giraffe', 'dog', 'dining table', 'toaster', 'couch', 'refrigerator', 'backpack', 'toothbrush', 'bench', 'zebra', 'person', 'orange', 'horse', 'skateboard', 'airplane', 'apple', 'bed'],
    "coco_35-2": ['backpack', 'carrot', 'knife', 'tennis racket', 'dining table', 'hair drier', 'toilet', 'stop sign', 'airplane', 'cell phone', 'bottle', 'traffic light', 'apple', 'car', 'bench', 'handbag', 'snowboard', 'donut', 'sandwich', 'tv', 'keyboard', 'teddy bear', 'bicycle', 'hot dog', 'orange', 'book', 'baseball glove', 'motorcycle', 'baseball bat', 'remote', 'oven', 'potted plant', 'sheep', 'bus', 'frisbee'],
    "coco_35-3": ['toothbrush', 'skis', 'apple', 'potted plant', 'backpack', 'bear', 'car', 'motorcycle', 'bed', 'remote', 'bottle', 'giraffe', 'book', 'keyboard', 'cow', 'bowl', 'orange', 'dining table', 'wine glass', 'zebra', 'bench', 'carrot', 'cake', 'vase', 'stop sign', 'umbrella', 'broccoli', 'pizza', 'cell phone', 'banana', 'spoon', 'sink', 'refrigerator', 'mouse', 'laptop'],
    "coco_35-4": ['chair', 'orange', 'traffic light', 'broccoli', 'cup', 'potted plant', 'skateboard', 'sink', 'bird', 'cake', 'tie', 'tennis racket', 'cell phone', 'remote', 'person', 'kite', 'baseball bat', 'scissors', 'sheep', 'dining table', 'wine glass', 'stop sign', 'bear', 'cow', 'toilet', 'elephant', 'hair drier', 'bottle', 'dog', 'teddy bear', 'horse', 'skis', 'toothbrush', 'handbag', 'zebra'],
    "coco_35-5": ['tv', 'orange', 'teddy bear', 'stop sign', 'mouse', 'apple', 'kite', 'car', 'skis', 'sheep', 'couch', 'elephant', 'book', 'handbag', 'person', 'hair drier', 'dog', 'horse', 'sports ball', 'spoon', 'bed', 'oven', 'backpack', 'boat', 'cake', 'refrigerator', 'cat', 'airplane', 'surfboard', 'baseball bat', 'hot dog', 'keyboard', 'frisbee', 'parking meter', 'carrot'],
    "coco_45-1": ['frisbee', 'orange', 'person', 'keyboard', 'toilet', 'truck', 'couch', 'tie', 'donut', 'pizza', 'fire hydrant', 'cup', 'bed', 'vase', 'wine glass', 'baseball glove', 'traffic light', 'broccoli', 'bird', 'apple', 'microwave', 'sink', 'fork', 'cake', 'bottle', 'hair drier', 'bench', 'backpack', 'baseball bat', 'toaster', 'book', 'airplane', 'giraffe', 'oven', 'chair', 'dog', 'parking meter', 'cat', 'suitcase', 'skis', 'sandwich', 'dining table', 'hot dog', 'sports ball', 'mouse'],
    "coco_45-2": ['potted plant', 'cake', 'bus', 'sandwich', 'hair drier', 'snowboard', 'suitcase', 'orange', 'bicycle', 'banana', 'umbrella', 'truck', 'keyboard', 'spoon', 'train', 'skateboard', 'backpack', 'teddy bear', 'tv', 'sheep', 'traffic light', 'cup', 'sink', 'knife', 'fork', 'microwave', 'sports ball', 'toaster', 'scissors', 'surfboard', 'carrot', 'apple', 'bear', 'tennis racket', 'oven', 'parking meter', 'boat', 'toothbrush', 'horse', 'cow', 'wine glass', 'person', 'cell phone', 'mouse', 'stop sign'],
    "coco_45-3": ['backpack', 'mouse', 'snowboard', 'bench', 'bus', 'baseball bat', 'cake', 'cell phone', 'dining table', 'surfboard', 'teddy bear', 'bicycle', 'hot dog', 'couch', 'laptop', 'cup', 'pizza', 'parking meter', 'skis', 'elephant', 'keyboard', 'microwave', 'bowl', 'dog', 'car', 'orange', 'skateboard', 'potted plant', 'sheep', 'tennis racket', 'clock', 'train', 'airplane', 'broccoli', 'scissors', 'spoon', 'suitcase', 'cat', 'baseball glove', 'bear', 'banana', 'tv', 'tie', 'handbag', 'horse'],
    "coco_45-4": ['bench', 'baseball glove', 'keyboard', 'apple', 'remote', 'car', 'scissors', 'chair', 'backpack', 'skis', 'giraffe', 'vase', 'bird', 'hot dog', 'bear', 'banana', 'fork', 'airplane', 'frisbee', 'sheep', 'mouse', 'person', 'elephant', 'boat', 'refrigerator', 'sandwich', 'bicycle', 'surfboard', 'hair drier', 'truck', 'potted plant', 'bowl', 'pizza', 'couch', 'spoon', 'bed', 'train', 'wine glass', 'bus', 'suitcase', 'cow', 'book', 'horse', 'sports ball', 'cell phone'],
    "coco_45-5": ['bed', 'cat', 'hot dog', 'skateboard', 'snowboard', 'cake', 'cup', 'apple', 'train', 'bench', 'giraffe', 'dining table', 'toaster', 'car', 'horse', 'tv', 'baseball bat', 'teddy bear', 'elephant', 'bowl', 'backpack', 'stop sign', 'couch', 'mouse', 'broccoli', 'traffic light', 'toilet', 'tie', 'oven', 'book', 'fire hydrant', 'airplane', 'sink', 'spoon', 'surfboard', 'wine glass', 'suitcase', 'fork', 'refrigerator', 'bicycle', 'toothbrush', 'sheep', 'clock', 'knife', 'skis'],
    "coco_50-1": ['baseball glove', 'surfboard', 'hot dog', 'bus', 'truck', 'keyboard', 'teddy bear', 'remote', 'parking meter', 'motorcycle', 'bear', 'sandwich', 'airplane', 'oven', 'toaster', 'dining table', 'elephant', 'apple', 'book', 'giraffe', 'clock', 'boat', 'cake', 'zebra', 'skis', 'spoon', 'pizza', 'snowboard', 'umbrella', 'bottle', 'scissors', 'microwave', 'bird', 'fork', 'refrigerator', 'vase', 'tennis racket', 'tie', 'wine glass', 'baseball bat', 'bed', 'potted plant', 'cup', 'mouse', 'tv', 'donut', 'sink', 'horse', 'train', 'car'],
    "coco_50-2": ['traffic light', 'suitcase', 'baseball glove', 'car', 'sheep', 'toothbrush', 'cow', 'snowboard', 'stop sign', 'apple', 'bench', 'tennis racket', 'giraffe', 'vase', 'orange', 'clock', 'elephant', 'mouse', 'pizza', 'book', 'spoon', 'sink', 'bird', 'surfboard', 'backpack', 'bed', 'sports ball', 'chair', 'zebra', 'wine glass', 'train', 'person', 'hot dog', 'potted plant', 'keyboard', 'knife', 'cake', 'cell phone', 'toilet', 'cup', 'dining table', 'frisbee', 'scissors', 'teddy bear', 'skateboard', 'parking meter', 'hair drier', 'baseball bat', 'bear', 'oven'],
    "coco_50-3": ['elephant', 'fire hydrant', 'horse', 'hair drier', 'fork', 'toilet', 'tv', 'wine glass', 'tie', 'kite', 'baseball glove', 'bottle', 'snowboard', 'umbrella', 'remote', 'microwave', 'bowl', 'banana', 'giraffe', 'book', 'baseball bat', 'car', 'motorcycle', 'hot dog', 'cake', 'toothbrush', 'dining table', 'orange', 'bench', 'pizza', 'parking meter', 'surfboard', 'cell phone', 'bird', 'bicycle', 'teddy bear', 'bed', 'suitcase', 'donut', 'keyboard', 'cow', 'clock', 'toaster', 'sink', 'traffic light', 'broccoli', 'cup', 'oven', 'truck', 'zebra'],
    "coco_50-4": ['frisbee', 'bowl', 'snowboard', 'mouse', 'couch', 'cat', 'refrigerator', 'keyboard', 'sink', 'hair drier', 'umbrella', 'parking meter', 'surfboard', 'book', 'carrot', 'toilet', 'person', 'potted plant', 'bench', 'baseball bat', 'tv', 'tennis racket', 'knife', 'hot dog', 'remote', 'traffic light', 'tie', 'spoon', 'teddy bear', 'stop sign', 'car', 'cake', 'fork', 'chair', 'motorcycle', 'dog', 'cell phone', 'bottle', 'suitcase', 'toothbrush', 'cup', 'bird', 'zebra', 'scissors', 'apple', 'bus', 'bicycle', 'pizza', 'baseball glove', 'horse'],
    "coco_50-5": ['clock', 'surfboard', 'cat', 'umbrella', 'suitcase', 'cow', 'boat', 'mouse', 'tennis racket', 'zebra', 'sheep', 'toilet', 'person', 'microwave', 'sink', 'book', 'hot dog', 'dog', 'truck', 'tie', 'toaster', 'frisbee', 'backpack', 'dining table', 'banana', 'horse', 'traffic light', 'motorcycle', 'bottle', 'parking meter', 'tv', 'apple', 'refrigerator', 'scissors', 'donut', 'chair', 'bear', 'car', 'bed', 'pizza', 'laptop', 'giraffe', 'wine glass', 'vase', 'remote', 'broccoli', 'teddy bear', 'keyboard', 'baseball bat', 'potted plant'],
    "coco_55-1": ['frisbee', 'orange', 'horse', 'car', 'book', 'suitcase', 'giraffe', 'broccoli', 'sandwich', 'banana', 'refrigerator', 'wine glass', 'knife', 'bear', 'hot dog', 'bottle', 'handbag', 'motorcycle', 'tennis racket', 'microwave', 'cup', 'parking meter', 'donut', 'bird', 'tie', 'person', 'carrot', 'potted plant', 'backpack', 'chair', 'bowl', 'couch', 'laptop', 'vase', 'traffic light', 'sheep', 'sink', 'truck', 'train', 'zebra', 'skis', 'remote', 'pizza', 'sports ball', 'boat', 'mouse', 'baseball bat', 'stop sign', 'hair drier', 'bed', 'bus', 'teddy bear', 'apple', 'cow', 'cake'],
    "coco_55-2": ['remote', 'spoon', 'bicycle', 'oven', 'sink', 'keyboard', 'tie', 'sports ball', 'surfboard', 'laptop', 'donut', 'refrigerator', 'knife', 'toothbrush', 'boat', 'mouse', 'person', 'couch', 'bed', 'parking meter', 'truck', 'bear', 'chair', 'teddy bear', 'zebra', 'orange', 'toilet', 'clock', 'vase', 'hair drier', 'tennis racket', 'kite', 'giraffe', 'broccoli', 'traffic light', 'suitcase', 'bus', 'skis', 'cake', 'bird', 'baseball glove', 'dining table', 'sandwich', 'train', 'cow', 'baseball bat', 'cat', 'apple', 'airplane', 'cell phone', 'bench', 'bowl', 'hot dog', 'stop sign', 'fork'],
    "coco_55-3": ['toothbrush', 'bus', 'backpack', 'skateboard', 'potted plant', 'laptop', 'snowboard', 'tv', 'train', 'giraffe', 'tie', 'truck', 'vase', 'cake', 'sheep', 'zebra', 'sink', 'wine glass', 'keyboard', 'toaster', 'fork', 'boat', 'person', 'banana', 'bicycle', 'bed', 'sports ball', 'pizza', 'tennis racket', 'remote', 'broccoli', 'umbrella', 'mouse', 'couch', 'refrigerator', 'microwave', 'book', 'teddy bear', 'carrot', 'cat', 'bench', 'cell phone', 'hair drier', 'motorcycle', 'car', 'bird', 'fire hydrant', 'airplane', 'handbag', 'apple', 'knife', 'orange', 'baseball glove', 'baseball bat', 'scissors'],
    "coco_55-4": ['tv', 'knife', 'bus', 'dog', 'boat', 'donut', 'elephant', 'cat', 'apple', 'tennis racket', 'kite', 'pizza', 'toothbrush', 'baseball glove', 'handbag', 'carrot', 'hot dog', 'bowl', 'toaster', 'tie', 'laptop', 'train', 'stop sign', 'sports ball', 'bed', 'skis', 'wine glass', 'fork', 'snowboard', 'microwave', 'horse', 'sink', 'toilet', 'cell phone', 'bottle', 'mouse', 'broccoli', 'hair drier', 'backpack', 'skateboard', 'refrigerator', 'giraffe', 'sheep', 'cow', 'person', 'orange', 'cup', 'banana', 'spoon', 'bench', 'parking meter', 'airplane', 'frisbee', 'oven', 'fire hydrant'],
    "coco_55-5": ['fork', 'cow', 'bed', 'baseball bat', 'bottle', 'bicycle', 'boat', 'kite', 'baseball glove', 'dining table', 'refrigerator', 'surfboard', 'train', 'truck', 'stop sign', 'sheep', 'cake', 'microwave', 'frisbee', 'toilet', 'zebra', 'airplane', 'apple', 'person', 'dog', 'hair drier', 'donut', 'toothbrush', 'cat', 'bench', 'skis', 'orange', 'potted plant', 'giraffe', 'bus', 'book', 'remote', 'teddy bear', 'banana', 'elephant', 'hot dog', 'knife', 'bear', 'couch', 'traffic light', 'wine glass', 'chair', 'backpack', 'mouse', 'cell phone', 'snowboard', 'sports ball', 'bird', 'laptop', 'vase'],
    "coco_60-1": ['zebra', 'bear', 'clock', 'broccoli', 'remote', 'vase', 'tie', 'train', 'orange', 'apple', 'toothbrush', 'sandwich', 'skis', 'cake', 'parking meter', 'microwave', 'couch', 'fork', 'bowl', 'dining table', 'potted plant', 'wine glass', 'laptop', 'boat', 'cat', 'person', 'bus', 'handbag', 'cow', 'spoon', 'hair drier', 'knife', 'frisbee', 'tennis racket', 'toilet', 'book', 'oven', 'hot dog', 'keyboard', 'teddy bear', 'sink', 'bicycle', 'surfboard', 'cell phone', 'bed', 'backpack', 'chair', 'airplane', 'mouse', 'bottle', 'scissors', 'car', 'truck', 'bench', 'umbrella', 'horse', 'donut', 'dog', 'suitcase', 'elephant'],
    "coco_60-2": ['apple', 'train', 'oven', 'carrot', 'spoon', 'cup', 'suitcase', 'clock', 'bicycle', 'fork', 'dog', 'person', 'toothbrush', 'bench', 'stop sign', 'sheep', 'dining table', 'hair drier', 'tie', 'parking meter', 'tennis racket', 'backpack', 'banana', 'surfboard', 'potted plant', 'truck', 'wine glass', 'scissors', 'sports ball', 'cat', 'cow', 'baseball glove', 'orange', 'snowboard', 'book', 'traffic light', 'skis', 'sandwich', 'skateboard', 'motorcycle', 'bed', 'couch', 'broccoli', 'airplane', 'handbag', 'giraffe', 'fire hydrant', 'mouse', 'hot dog', 'bowl', 'boat', 'zebra', 'knife', 'refrigerator', 'bottle', 'chair', 'tv', 'kite', 'bird', 'bear'],
    "coco_60-3": ['mouse', 'sandwich', 'refrigerator', 'giraffe', 'toothbrush', 'fork', 'traffic light', 'tie', 'zebra', 'tennis racket', 'bus', 'carrot', 'scissors', 'toaster', 'truck', 'cake', 'broccoli', 'pizza', 'bowl', 'skis', 'apple', 'skateboard', 'orange', 'spoon', 'baseball bat', 'oven', 'cup', 'kite', 'bird', 'person', 'stop sign', 'suitcase', 'motorcycle', 'laptop', 'dining table', 'bottle', 'boat', 'horse', 'backpack', 'cell phone', 'sports ball', 'knife', 'parking meter', 'wine glass', 'book', 'microwave', 'keyboard', 'snowboard', 'hair drier', 'cat', 'vase', 'fire hydrant', 'elephant', 'sink', 'toilet', 'sheep', 'dog', 'train', 'donut', 'frisbee'],
    "coco_60-4": ['elephant', 'bus', 'handbag', 'cake', 'motorcycle', 'spoon', 'umbrella', 'book', 'bowl', 'bench', 'kite', 'remote', 'tennis racket', 'chair', 'apple', 'toilet', 'refrigerator', 'skis', 'hot dog', 'airplane', 'cow', 'bird', 'sandwich', 'dining table', 'snowboard', 'fork', 'teddy bear', 'potted plant', 'mouse', 'vase', 'carrot', 'horse', 'sink', 'bicycle', 'backpack', 'couch', 'zebra', 'clock', 'tv', 'banana', 'cat', 'car', 'tie', 'bear', 'traffic light', 'orange', 'giraffe', 'baseball bat', 'donut', 'person', 'frisbee', 'toaster', 'parking meter', 'suitcase', 'fire hydrant', 'cell phone', 'skateboard', 'bottle', 'pizza', 'microwave'],
    "coco_60-5": ['remote', 'bird', 'baseball glove', 'book', 'parking meter', 'refrigerator', 'apple', 'person', 'dog', 'hair drier', 'knife', 'cow', 'horse', 'microwave', 'fork', 'pizza', 'bottle', 'car', 'bench', 'suitcase', 'tie', 'train', 'wine glass', 'mouse', 'oven', 'motorcycle', 'dining table', 'toothbrush', 'boat', 'toilet', 'cell phone', 'hot dog', 'baseball bat', 'couch', 'spoon', 'handbag', 'bowl', 'sandwich', 'airplane', 'skis', 'broccoli', 'bear', 'toaster', 'stop sign', 'donut', 'zebra', 'truck', 'orange', 'surfboard', 'bicycle', 'clock', 'chair', 'cake', 'scissors', 'snowboard', 'backpack', 'banana', 'kite', 'sheep', 'umbrella'],
    "coco_65-1": ['train', 'laptop', 'oven', 'cow', 'baseball glove', 'bus', 'knife', 'toothbrush', 'zebra', 'sports ball', 'tie', 'carrot', 'wine glass', 'handbag', 'refrigerator', 'elephant', 'bear', 'bowl', 'backpack', 'vase', 'sandwich', 'teddy bear', 'cup', 'mouse', 'truck', 'cell phone', 'bottle', 'potted plant', 'microwave', 'pizza', 'toaster', 'hot dog', 'donut', 'toilet', 'keyboard', 'skis', 'skateboard', 'bicycle', 'hair drier', 'traffic light', 'airplane', 'frisbee', 'baseball bat', 'bed', 'book', 'spoon', 'tv', 'dining table', 'bird', 'person', 'scissors', 'stop sign', 'snowboard', 'suitcase', 'broccoli', 'car', 'banana', 'couch', 'horse', 'fire hydrant', 'bench', 'chair', 'umbrella', 'orange', 'kite'],
    "coco_65-2": ['motorcycle', 'cake', 'toilet', 'bird', 'tennis racket', 'giraffe', 'baseball glove', 'hair drier', 'car', 'bottle', 'tv', 'elephant', 'boat', 'sink', 'backpack', 'airplane', 'dog', 'fork', 'chair', 'kite', 'frisbee', 'banana', 'hot dog', 'laptop', 'keyboard', 'surfboard', 'bench', 'book', 'remote', 'bus', 'microwave', 'spoon', 'bear', 'snowboard', 'sheep', 'clock', 'refrigerator', 'bowl', 'apple', 'cell phone', 'truck', 'toaster', 'toothbrush', 'broccoli', 'stop sign', 'person', 'dining table', 'zebra', 'cow', 'teddy bear', 'sandwich', 'vase', 'fire hydrant', 'potted plant', 'parking meter', 'handbag', 'mouse', 'wine glass', 'tie', 'skateboard', 'oven', 'sports ball', 'bed', 'orange', 'scissors'],
    "coco_65-3": ['wine glass', 'cake', 'snowboard', 'truck', 'mouse', 'toaster', 'bird', 'couch', 'clock', 'handbag', 'traffic light', 'keyboard', 'fork', 'orange', 'bench', 'book', 'tv', 'carrot', 'hot dog', 'broccoli', 'airplane', 'bus', 'apple', 'teddy bear', 'umbrella', 'parking meter', 'motorcycle', 'bicycle', 'toothbrush', 'dog', 'baseball glove', 'zebra', 'laptop', 'tie', 'dining table', 'hair drier', 'giraffe', 'stop sign', 'suitcase', 'train', 'knife', 'potted plant', 'backpack', 'donut', 'oven', 'microwave', 'sports ball', 'cat', 'sheep', 'kite', 'person', 'refrigerator', 'surfboard', 'cell phone', 'pizza', 'chair', 'tennis racket', 'cup', 'scissors', 'elephant', 'frisbee', 'skis', 'bed', 'bottle', 'horse'],
    "coco_65-4": ['chair', 'kite', 'banana', 'dining table', 'bear', 'apple', 'backpack', 'sandwich', 'hair drier', 'frisbee', 'keyboard', 'toothbrush', 'clock', 'tennis racket', 'cell phone', 'oven', 'laptop', 'bicycle', 'skis', 'umbrella', 'toilet', 'train', 'vase', 'book', 'parking meter', 'cup', 'suitcase', 'skateboard', 'bed', 'couch', 'car', 'potted plant', 'motorcycle', 'giraffe', 'donut', 'pizza', 'baseball glove', 'microwave', 'truck', 'horse', 'scissors', 'zebra', 'bus', 'snowboard', 'remote', 'airplane', 'spoon', 'baseball bat', 'refrigerator', 'fork', 'bottle', 'handbag', 'hot dog', 'sports ball', 'teddy bear', 'tie', 'bowl', 'person', 'fire hydrant', 'elephant', 'orange', 'sink', 'traffic light', 'bench', 'cake'],
    "coco_65-5": ['cup', 'remote', 'spoon', 'cat', 'car', 'fire hydrant', 'toaster', 'bicycle', 'sink', 'horse', 'couch', 'pizza', 'frisbee', 'cow', 'suitcase', 'bowl', 'bottle', 'orange', 'teddy bear', 'clock', 'kite', 'bus', 'broccoli', 'airplane', 'tie', 'oven', 'cake', 'bear', 'surfboard', 'snowboard', 'giraffe', 'zebra', 'wine glass', 'book', 'microwave', 'laptop', 'traffic light', 'tv', 'keyboard', 'hair drier', 'scissors', 'motorcycle', 'sandwich', 'handbag', 'baseball bat', 'boat', 'toilet', 'train', 'parking meter', 'bed', 'toothbrush', 'backpack', 'vase', 'potted plant', 'mouse', 'fork', 'hot dog', 'tennis racket', 'refrigerator', 'dog', 'sports ball', 'baseball glove', 'bird', 'chair', 'sheep'],
    "coco_70-1": ['scissors', 'wine glass', 'backpack', 'sandwich', 'refrigerator', 'banana', 'carrot', 'boat', 'surfboard', 'bird', 'sports ball', 'bed', 'toothbrush', 'car', 'cake', 'donut', 'fire hydrant', 'baseball bat', 'tennis racket', 'motorcycle', 'handbag', 'tv', 'pizza', 'bear', 'dog', 'broccoli', 'snowboard', 'suitcase', 'giraffe', 'tie', 'apple', 'parking meter', 'bench', 'chair', 'baseball glove', 'hot dog', 'knife', 'skis', 'teddy bear', 'toaster', 'kite', 'laptop', 'keyboard', 'bottle', 'frisbee', 'elephant', 'truck', 'airplane', 'cell phone', 'sheep', 'book', 'orange', 'dining table', 'microwave', 'toilet', 'train', 'cow', 'cat', 'sink', 'cup', 'oven', 'person', 'vase', 'fork', 'remote', 'traffic light', 'bus', 'couch', 'mouse', 'clock'],
    "coco_70-2": ['refrigerator', 'bicycle', 'cat', 'pizza', 'knife', 'fork', 'donut', 'laptop', 'chair', 'parking meter', 'bird', 'airplane', 'banana', 'hot dog', 'handbag', 'scissors', 'backpack', 'baseball glove', 'suitcase', 'couch', 'spoon', 'hair drier', 'car', 'boat', 'teddy bear', 'orange', 'oven', 'keyboard', 'person', 'dining table', 'cake', 'book', 'horse', 'bear', 'sink', 'dog', 'cell phone', 'bench', 'bowl', 'traffic light', 'wine glass', 'microwave', 'snowboard', 'motorcycle', 'remote', 'surfboard', 'apple', 'kite', 'bus', 'fire hydrant', 'truck', 'giraffe', 'toothbrush', 'cup', 'carrot', 'skateboard', 'elephant', 'sports ball', 'zebra', 'stop sign', 'bed', 'umbrella', 'tennis racket', 'baseball bat', 'potted plant', 'toilet', 'mouse', 'train', 'sheep', 'tie'],
    "coco_70-3": ['scissors', 'elephant', 'car', 'mouse', 'donut', 'bicycle', 'oven', 'refrigerator', 'boat', 'stop sign', 'potted plant', 'fire hydrant', 'cell phone', 'bottle', 'microwave', 'hot dog', 'train', 'vase', 'keyboard', 'frisbee', 'backpack', 'chair', 'bowl', 'dog', 'broccoli', 'orange', 'book', 'tennis racket', 'truck', 'person', 'suitcase', 'sports ball', 'cup', 'bed', 'hair drier', 'toaster', 'banana', 'skateboard', 'cow', 'bus', 'kite', 'horse', 'cat', 'traffic light', 'handbag', 'bird', 'parking meter', 'couch', 'dining table', 'spoon', 'zebra', 'sheep', 'airplane', 'remote', 'bear', 'snowboard', 'tie', 'knife', 'clock', 'surfboard', 'carrot', 'baseball bat', 'bench', 'sandwich', 'laptop', 'tv', 'skis', 'motorcycle', 'umbrella', 'giraffe'],
    "coco_70-4": ['sink', 'apple', 'truck', 'bicycle', 'airplane', 'boat', 'cow', 'book', 'couch', 'suitcase', 'potted plant', 'teddy bear', 'carrot', 'horse', 'giraffe', 'bear', 'hair drier', 'baseball glove', 'cell phone', 'broccoli', 'bench', 'dining table', 'refrigerator', 'surfboard', 'toaster', 'bed', 'motorcycle', 'fire hydrant', 'umbrella', 'skis', 'fork', 'microwave', 'sports ball', 'pizza', 'snowboard', 'cup', 'dog', 'chair', 'sheep', 'skateboard', 'vase', 'cake', 'tie', 'stop sign', 'traffic light', 'bottle', 'laptop', 'knife', 'wine glass', 'kite', 'hot dog', 'backpack', 'person', 'cat', 'sandwich', 'handbag', 'toothbrush', 'parking meter', 'zebra', 'banana', 'keyboard', 'mouse', 'bowl', 'scissors', 'orange', 'spoon', 'tv', 'remote', 'bus', 'baseball bat'],
    "coco_70-5": ['mouse', 'backpack', 'boat', 'stop sign', 'bear', 'frisbee', 'teddy bear', 'zebra', 'suitcase', 'toothbrush', 'broccoli', 'scissors', 'hot dog', 'surfboard', 'cow', 'bench', 'handbag', 'bottle', 'skis', 'person', 'airplane', 'tennis racket', 'wine glass', 'carrot', 'vase', 'dining table', 'oven', 'clock', 'sink', 'snowboard', 'book', 'traffic light', 'toilet', 'truck', 'keyboard', 'toaster', 'bus', 'horse', 'hair drier', 'baseball bat', 'orange', 'kite', 'cake', 'giraffe', 'remote', 'potted plant', 'cell phone', 'tv', 'banana', 'bowl', 'knife', 'tie', 'apple', 'laptop', 'spoon', 'sandwich', 'microwave', 'fire hydrant', 'couch', 'car', 'refrigerator', 'donut', 'skateboard', 'elephant', 'bird', 'train', 'chair', 'cat', 'sports ball', 'pizza'],
    "coco_75-1": ['handbag', 'traffic light', 'stop sign', 'parking meter', 'apple', 'skis', 'hair drier', 'microwave', 'sandwich', 'snowboard', 'car', 'giraffe', 'motorcycle', 'tennis racket', 'keyboard', 'pizza', 'bus', 'bear', 'airplane', 'tie', 'bench', 'elephant', 'donut', 'dog', 'cat', 'baseball glove', 'couch', 'sports ball', 'cake', 'remote', 'bicycle', 'backpack', 'dining table', 'sink', 'fire hydrant', 'knife', 'spoon', 'wine glass', 'cell phone', 'kite', 'refrigerator', 'baseball bat', 'oven', 'boat', 'zebra', 'cup', 'bed', 'banana', 'broccoli', 'truck', 'scissors', 'frisbee', 'bottle', 'toothbrush', 'carrot', 'train', 'orange', 'cow', 'person', 'bowl', 'potted plant', 'fork', 'umbrella', 'teddy bear', 'mouse', 'toaster', 'toilet', 'sheep', 'vase', 'bird', 'suitcase', 'hot dog', 'book', 'skateboard', 'clock'],
    "coco_75-2": ['person', 'baseball bat', 'motorcycle', 'skis', 'hot dog', 'clock', 'refrigerator', 'bus', 'kite', 'bench', 'couch', 'tv', 'orange', 'banana', 'tennis racket', 'surfboard', 'mouse', 'giraffe', 'hair drier', 'traffic light', 'sink', 'oven', 'snowboard', 'bed', 'horse', 'zebra', 'teddy bear', 'cake', 'suitcase', 'boat', 'sports ball', 'pizza', 'cup', 'bird', 'skateboard', 'fork', 'book', 'handbag', 'tie', 'toilet', 'sandwich', 'frisbee', 'toothbrush', 'toaster', 'sheep', 'umbrella', 'train', 'bicycle', 'cow', 'knife', 'remote', 'bowl', 'cell phone', 'bear', 'donut', 'cat', 'spoon', 'airplane', 'stop sign', 'elephant', 'broccoli', 'parking meter', 'chair', 'car', 'laptop', 'carrot', 'potted plant', 'dog', 'backpack', 'keyboard', 'scissors', 'truck', 'wine glass', 'bottle', 'baseball glove'],
    "coco_75-3": ['train', 'backpack', 'cell phone', 'vase', 'skateboard', 'dog', 'traffic light', 'dining table', 'sandwich', 'snowboard', 'spoon', 'bowl', 'apple', 'bottle', 'stop sign', 'clock', 'chair', 'broccoli', 'wine glass', 'bus', 'bear', 'knife', 'orange', 'sheep', 'giraffe', 'bird', 'suitcase', 'bed', 'truck', 'book', 'cow', 'laptop', 'person', 'sink', 'boat', 'tv', 'hot dog', 'potted plant', 'fork', 'fire hydrant', 'bicycle', 'baseball bat', 'pizza', 'hair drier', 'teddy bear', 'tie', 'sports ball', 'scissors', 'kite', 'banana', 'couch', 'tennis racket', 'airplane', 'zebra', 'carrot', 'handbag', 'toaster', 'bench', 'toilet', 'umbrella', 'refrigerator', 'car', 'cake', 'parking meter', 'cup', 'cat', 'donut', 'toothbrush', 'frisbee', 'skis', 'remote', 'mouse', 'keyboard', 'horse', 'microwave'],
    "coco_75-4": ['umbrella', 'teddy bear', 'motorcycle', 'toaster', 'apple', 'bicycle', 'zebra', 'handbag', 'boat', 'skis', 'banana', 'bear', 'toilet', 'bottle', 'dining table', 'surfboard', 'potted plant', 'horse', 'keyboard', 'backpack', 'tie', 'donut', 'hot dog', 'traffic light', 'remote', 'elephant', 'carrot', 'cell phone', 'hair drier', 'frisbee', 'bench', 'truck', 'wine glass', 'bowl', 'suitcase', 'clock', 'baseball bat', 'cup', 'sandwich', 'knife', 'dog', 'toothbrush', 'stop sign', 'sports ball', 'giraffe', 'book', 'orange', 'airplane', 'parking meter', 'spoon', 'refrigerator', 'kite', 'fork', 'tennis racket', 'pizza', 'fire hydrant', 'snowboard', 'laptop', 'bird', 'bed', 'baseball glove', 'sink', 'microwave', 'scissors', 'sheep', 'vase', 'skateboard', 'cake', 'chair', 'car', 'person', 'train', 'oven', 'cat', 'mouse'],
    "coco_75-5": ['refrigerator', 'handbag', 'bear', 'broccoli', 'bicycle', 'suitcase', 'baseball glove', 'horse', 'laptop', 'cat', 'cell phone', 'spoon', 'cup', 'cow', 'orange', 'donut', 'pizza', 'train', 'toothbrush', 'sink', 'giraffe', 'parking meter', 'dining table', 'wine glass', 'surfboard', 'fork', 'skis', 'car', 'knife', 'sandwich', 'book', 'backpack', 'carrot', 'airplane', 'bird', 'fire hydrant', 'dog', 'snowboard', 'potted plant', 'hot dog', 'hair drier', 'banana', 'traffic light', 'elephant', 'clock', 'sports ball', 'tennis racket', 'bowl', 'bench', 'remote', 'tv', 'cake', 'bottle', 'boat', 'mouse', 'person', 'microwave', 'toilet', 'motorcycle', 'scissors', 'apple', 'couch', 'toaster', 'tie', 'kite', 'skateboard', 'bed', 'stop sign', 'teddy bear', 'keyboard', 'vase', 'zebra', 'oven', 'chair', 'baseball bat'],
    # "coco_person": ["person"],
    # "coco_bicycle": ["bicycle"],
    # "coco_car": ["car"],
    # "coco_motorcycle": ["motorcycle"],
    # "coco_airplane": ["airplane"],
    # "coco_bus": ["bus"],
    # "coco_train": ["train"],
    # "coco_truck": ["truck"],
    # "coco_boat": ["boat"],
    # "coco_traffic light": ["traffic light"],
    # "coco_fire hydrant": ["fire hydrant"],
    # "coco_stop sign": ["stop sign"],
    # "coco_parking meter": ["parking meter"],
    # "coco_bench": ["bench"],
    # "coco_bird": ["bird"],
    # "coco_cat": ["cat"],
    # "coco_dog": ["dog"],
    # "coco_horse": ["horse"],
    # "coco_sheep": ["sheep"],
    # "coco_cow": ["cow"],
    # "coco_elephant": ["elephant"],
    # "coco_bear": ["bear"],
    # "coco_zebra": ["zebra"],
    # "coco_giraffe": ["giraffe"],
    # "coco_backpack": ["backpack"],
    # "coco_umbrella": ["umbrella"],
    # "coco_handbag": ["handbag"],
    # "coco_tie": ["tie"],
    # "coco_suitcase": ["suitcase"],
    # "coco_frisbee": ["frisbee"],
    # "coco_skis": ["skis"],
    # "coco_snowboard": ["snowboard"],
    # "coco_sports ball": ["sports ball"],
    # "coco_kite": ["kite"],
    # "coco_baseball bat": ["baseball bat"],
    # "coco_baseball glove": ["baseball glove"],
    # "coco_skateboard": ["skateboard"],
    # "coco_surfboard": ["surfboard"],
    # "coco_tennis racket": ["tennis racket"],
    # "coco_bottle": ["bottle"],
    # "coco_wine glass": ["wine glass"],
    # "coco_cup": ["cup"],
    # "coco_fork": ["fork"],
    # "coco_knife": ["knife"],
    # "coco_spoon": ["spoon"],
    # "coco_bowl": ["bowl"],
    # "coco_banana": ["banana"],
    # "coco_apple": ["apple"],
    # "coco_sandwich": ["sandwich"],
    # "coco_orange": ["orange"],
    # "coco_broccoli": ["broccoli"],
    # "coco_carrot": ["carrot"],
    # "coco_hot dog": ["hot dog"],
    # "coco_pizza": ["pizza"],
    # "coco_donut": ["donut"],
    # "coco_cake": ["cake"],
    # "coco_chair": ["chair"],
    # "coco_couch": ["couch"],
    # "coco_potted plant": ["potted plant"],
    # "coco_bed": ["bed"],
    # "coco_dining table": ["dining table"],
    # "coco_toilet": ["toilet"],
    # "coco_tv": ["tv"],
    # "coco_laptop": ["laptop"],
    # "coco_mouse": ["mouse"],
    # "coco_remote": ["remote"],
    # "coco_keyboard": ["keyboard"],
    # "coco_cell phone": ["cell phone"],
    # "coco_microwave": ["microwave"],
    # "coco_oven": ["oven"],
    # "coco_toaster": ["toaster"],
    # "coco_sink": ["sink"],
    # "coco_refrigerator": ["refrigerator"],
    # "coco_book": ["book"],
    # "coco_clock": ["clock"],
    # "coco_vase": ["vase"],
    # "coco_scissors": ["scissors"],
    # "coco_teddy bear": ["teddy bear"],
    # "coco_hair drier": ["hair drier"],
    # "coco_toothbrush": ["toothbrush"],
    # "coco_appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    # "coco_electronic": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    # "coco_sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
    # "coco_furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    # "coco_animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    # "coco_vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    # "coco_kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    # "coco_accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    # "coco_food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    # "coco_outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    # "coco_indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
}

@TRANSFORMS.register_module()
class GenerateClassSubset(BaseTransform):
    """Generate a subset of classes for the dataset.

    Args:
        rule (str | callable): The rule to generate the subset of classes.
            It can be a string or a callable. If it is a string, it should be
            one of ['random', 'all', 'animal', ...]. If it is a callable, it should be a
            function that takes the class indices as input and returns a list
            of class indices.
        classes (str): The file path of the full class names.
        dataset_classes (str): The file path of the valid class names in the dataset.
    """

    def __init__(self, rule: Union[str, callable], subset_size: Union[int, tuple, list] = 80, classes: Optional[str] = None,
                 ignore: Optional[bool] = True):
        assert classes is not None, "classes should be provided for GenerateClassSubset"
        categories = mmengine.list_from_file(classes)
        self.categories = categories
        self.ignore = ignore

        if rule in ['random', 'all', 'gt', 'uniform']:
            self.rule = rule
        else:
            assert rule in rule_dict, "rule should be in rule_dict or 'random', 'all', 'gt'"
            self.rule = rule
            
            class_subset = []
            for idx, cat in enumerate(self.categories):
                if (cat in rule_dict[rule]) ^ ('#wo#' in rule):
                    class_subset.append(idx)
            self.class_subset = class_subset

        self.rule = rule
        self.subset_size = subset_size

    def transform(self, results: dict) -> dict:
        if self.rule == 'uniform':
            class_mask = np.random.rand(len(self.categories)) < 0.5
            class_subset = np.arange(len(self.categories))[class_mask]
        elif self.rule == 'random':
            appeared_classes = np.unique(results['gt_bboxes_labels'])
            np.random.shuffle(appeared_classes)
            un_appeared_classes = np.setdiff1d(np.arange(len(self.categories)), appeared_classes)
            np.random.shuffle(un_appeared_classes)
            if isinstance(self.subset_size, int):
                subset_size = self.subset_size
            else:
                assert isinstance(self.subset_size, tuple)
                subset_size = random.choice(self.subset_size, 1)[0]
            
            # sample subset_size classes from all classes with at least one appeared class
            # while True:
            #     # randomly sample subset_size classes from all classes
            #     class_subset = random.choice(self.num_classes, subset_size, replace=False)
            #     # check if all sampled classes have at least one appeared class
            #     if np.sum(np.isin(class_subset, appeared_classes)) > 0:
            #         break

            # sample subset_size classes from appeared classes
            # class_subset = list()
            # src_typ=0
            # while len(class_subset) < subset_size:
            #     src_typ = src_typ ^ 1
            #     if src_typ==1:
            #         # select one from appeared classes
            #         candidates = np.setdiff1d(appeared_classes, class_subset)
            #         # prob is proportional to the number of instances in the class (reverse)
            #         if len(candidates) > 0:
            #             weights = np.array([np.sum(results['gt_bboxes_labels'] == i) for i in candidates])
            #             weights = 1 / (weights + 1e-6)  # avoid division by zero
            #             weights = weights / np.sum(weights)
            #     else:
            #         assert src_typ==0, "src_typ should be 0 or 1"
            #         # select one from un-appeared classes
            #         candidates = np.setdiff1d(un_appeared_classes, class_subset)
            #         weights = np.ones(len(candidates)) / len(candidates) if len(candidates) > 0 else np.array([])
            #     if len(candidates) == 0:
            #         continue
            #     class_subset.append(random.choice(candidates, 1, p=weights)[0])

            # sample subset_size classes from appeared and un-appeared classes
            class_subset = list()
            while len(class_subset) < subset_size:
                if np.random.rand() < 0.5:
                    # select one from appeared classes
                    candidates = np.setdiff1d(appeared_classes, class_subset)
                    # prob is proportional to the number of instances in the class (reverse)
                    if len(candidates) > 0:
                        weights = np.array([np.sum(results['gt_bboxes_labels'] == i) for i in candidates])
                        weights = 1 / (weights + 1e-6)  # avoid division by zero
                        weights = weights / np.sum(weights)
                else:
                    # select one from un-appeared classes
                    candidates = np.setdiff1d(un_appeared_classes, class_subset)
                    weights = np.ones(len(candidates)) / len(candidates) if len(candidates) > 0 else np.array([])
                if len(candidates) == 0:
                    continue
                class_subset.append(random.choice(candidates, 1, p=weights)[0])

            # class_subset = list()
            # for i in range(len(appeared_classes)):
            #     if np.random.rand() < 0.5:
            #         class_subset.append(appeared_classes[i])
            #         if len(class_subset) == subset_size:
            #             break
            # if len(class_subset) < subset_size:
            #     class_subset += list(un_appeared_classes[:subset_size - len(class_subset)])

            # class_subset = list()
            # for i in range(len(self.categories)):
            #     if np.random.rand() < 0.25:
            #         class_subset.append(i)

            # class_subset = random.choice(np.arange(len(self.categories)), subset_size, replace=False)
            # class_subset_2 = random.choice(np.arange(len(self.categories)), subset_size, replace=False)
            # class_subset = np.concatenate([class_subset, class_subset_2])
            # remove class_subset from all classes
            class_subset = np.array(class_subset)
        elif self.rule == 'all':
            class_subset = np.arange(len(self.categories))
        elif self.rule == 'gt':
            class_subset = np.unique(results['gt_bboxes_labels'])
        else:
            class_subset = np.array(self.class_subset)
        results['class_subset'] = class_subset

        # add gt_ignore_flags to results
        if 'gt_bboxes' in results and 'gt_bboxes_labels' in results and self.ignore:
            gt_cats = [self.categories[i] for i in results['gt_bboxes_labels']]
            valid_cats = [self.categories[i] for i in class_subset]
            valid_mask = np.isin(gt_cats, valid_cats)
            if 'gt_ignore_flags' in results:
                results['gt_ignore_flags'] = results['gt_ignore_flags'] | ~valid_mask
            else:
                results['gt_ignore_flags'] = ~valid_mask
        return results
        