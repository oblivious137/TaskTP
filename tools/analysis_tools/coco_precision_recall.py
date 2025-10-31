# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_pr_per_setting(coco_eval):

    iou_thrs = coco_eval.params.iouThrs
    area_labels = coco_eval.params.areaRngLbl
    img_ids = coco_eval.params.imgIds
    cat_ids = coco_eval.params.catIds

    n_iou = len(iou_thrs)
    n_area = len(area_labels)
    n_cat = len(cat_ids)
    n_img = len(img_ids)

    eval_imgs = coco_eval.evalImgs
    assert len(eval_imgs) == n_cat * n_area * n_img, \
        f"evalImgs size mismatch: expected {n_cat * n_area * n_img}, got {len(eval_imgs)}"

    # Reshape to (cat, area, img)
    eval_imgs = np.array(eval_imgs).reshape(n_cat, n_area, n_img)

    results = []

    for iou_idx, iou_thr in enumerate(iou_thrs):
        for area_idx, area_name in enumerate(area_labels):
            tp = fp = fn = 0
            for cat_idx in range(n_cat):
                for img_idx in range(n_img):
                    eval_img = eval_imgs[cat_idx, area_idx, img_idx]
                    if eval_img is None:
                        continue
                    # 对于当前 iou 阈值，取对应 match 情况
                    dt_matches = eval_img['dtMatches'][iou_idx]  # [maxDets]
                    gt_matches = eval_img['gtMatches'][iou_idx]  # [num_gt]

                    tp += np.sum(dt_matches > 0)
                    fp += np.sum(dt_matches == 0)
                    fn += len(gt_matches) - np.sum(gt_matches > 0)

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0

            results.append({
                'iou_thr': iou_thr,
                'area': area_name,
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn),
                'precision': precision,
                'recall': recall
            })

    return results



def analyze_results(res_file,
                    ann_file,
                    res_types,
                    areas=None,
                    score_thr=None):
    for res_type in res_types:
        assert res_type in ['bbox', 'segm']
    if areas:
        assert (len(areas) == 3), '3 integers should be specified as areas, \
            representing 3 area regions'

    if score_thr:
        assert score_thr >= 0, 'score_thr should be bigger than 0'

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()

    if score_thr:
        cocoDt.dataset['annotations'] = list(
            filter(lambda ann: ann['score'] >= score_thr,
                   cocoDt.dataset['annotations']))
        cocoDt.createIndex()

    for res_type in res_types:
        iou_type = res_type
        cocoEval = COCOeval(
            copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.iouThrs = [0.75, 0.5, 0.1]
        cocoEval.params.maxDets = [100]
        if areas:
            cocoEval.params.areaRng = [
                [0**2, areas[2]],
                [0**2, areas[0]],
                [areas[0], areas[1]],
                [areas[1], areas[2]],
            ]
        cocoEval.evaluate()
        
        stats = compute_pr_per_setting(cocoEval)
        for stat in stats:
            print(f"Result for {res_type} - "
                  f"IOU: {stat['iou_thr']}, Area: {stat['area']}, "
                  f"TP: {stat['TP']}, FP: {stat['FP']}, FN: {stat['FN']}, "
                  f"Precision: {stat['precision']:.4f}, Recall: {stat['recall']:.4f}")
        


def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('result', help='result file (json format) path')
    parser.add_argument(
        '--ann',
        default='data/coco/annotations/instances_val2017.json',
        help='annotation file path',
    )
    parser.add_argument(
        '--types', type=str, nargs='+', default=['bbox'], help='result types')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=None,
        help='score threshold to filter detection bboxes, only applied'
        'when users want to change it.',
    )
    parser.add_argument(
        '--areas',
        type=int,
        nargs='+',
        default=[1024, 9216, 10000000000],
        help='area regions',
    )
    args = parser.parse_args()
    analyze_results(
        res_file=args.result,
        ann_file=args.ann,
        res_types=args.types,
        areas=args.areas,
        score_thr=args.score_thr
    )


if __name__ == '__main__':
    main()
