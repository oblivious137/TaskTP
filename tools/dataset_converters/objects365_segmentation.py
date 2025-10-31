import numpy as np
import json
import argparse
import os.path as osp

parser = argparse.ArgumentParser(description='Generate COCO-style segmentation annotations for Objects365.')
parser.add_argument('--ann_file', type=str, help='Path to the COCO-style annotation file')
parser.add_argument('--output_dir', type=str, help='Directory to save the updated annotation file')

def bbox_to_polygon(bbox):
    """
    将边界框（bbox）转换为多边形表示。
    :param bbox: 边界框，格式为 [x_min, y_min, width, height]
    :return: 多边形点的列表，格式为 [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    # 矩形多边形的四个顶点（顺时针或逆时针均可）
    polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
    return [polygon]  # 返回嵌套列表以符合COCO格式

def add_mask_to_annotations(annotations):
    """
    为所有没有segmentation的实例生成矩形掩码（多边形表示），并添加到annotations中。
    :param annotations: COCO格式的标注列表
    :return: 更新后的annotations
    """
    for ann in annotations:
        if 'segmentation' not in ann or not ann['segmentation']:
            # 如果没有segmentation，使用bbox生成多边形掩码
            bbox = ann['bbox']
            polygon = bbox_to_polygon(bbox)
            ann['segmentation'] = polygon  # 更新segmentation字段
    return annotations

if __name__ == '__main__':
    args = parser.parse_args()
    ann_file = args.ann_file
    output_dir = args.output_dir
    assert ann_file.endswith('.json'), 'Annotation file must be a JSON file'
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    annotations['annotations'] = add_mask_to_annotations(annotations['annotations'])
    output_file = osp.join(output_dir, osp.basename(ann_file))
    with open(output_file, 'w') as f:
        json.dump(annotations, f)