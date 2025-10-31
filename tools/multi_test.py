import argparse
import os
import os.path as osp
import shutil
import datetime
import torch
from mmengine.config import Config, DictAction

# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco/epoch_12.pth 8 --cfg-options model.backbone.topk=1.0 test_dataloader.dataset.type="CocoPersonDataset" test_dataloader.dataset.ann_file="annotations/instances_val2017_person_available.json" test_evaluator.type="CocoSubsetMetric" test_evaluator.ann_file="data/coco/annotations/instances_val2017_person_available.json" test_evaluator.full_ann_file="data/coco/annotations/instances_val2017.json"
# ./tools/dist_test.sh configs/prune/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_person_test.py work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_subset_lora_training/epoch_12.pth 8 --cfg-options model.backbone.topk=1.0

test_subsets=[
    'coco_1-1',
    'coco_1-2',
    'coco_1-3',
    'coco_1-4',
    'coco_1-5',
    'all',
    'coco_10-1',
    'coco_10-2',
    'coco_10-3',
    'coco_10-4',
    'coco_10-5',
    'coco_40-1',
    'coco_40-2',
    'coco_40-3',
    'coco_40-4',
    'coco_40-5',
    # "coco_2-1",
    # "coco_2-2",
    # "coco_2-3",
    # "coco_2-4",
    # "coco_2-5",
    # "coco_3-1",
    # "coco_3-2",
    # "coco_3-3",
    # "coco_3-4",
    # "coco_3-5",
    # "coco_4-1",
    # "coco_4-2",
    # "coco_4-3",
    # "coco_4-4",
    # "coco_4-5",
    # "coco_5-1",
    # "coco_5-2",
    # "coco_5-3",
    # "coco_5-4",
    # "coco_5-5",
    # "coco_6-1",
    # "coco_6-2",
    # "coco_6-3",
    # "coco_6-4",
    # "coco_6-5",
    # "coco_7-1",
    # "coco_7-2",
    # "coco_7-3",
    # "coco_7-4",
    # "coco_7-5",
    # "coco_8-1",
    # "coco_8-2",
    # "coco_8-3",
    # "coco_8-4",
    # "coco_8-5",
    # "coco_9-1",
    # "coco_9-2",
    # "coco_9-3",
    # "coco_9-4",
    # "coco_9-5",
    # "coco_10-1",
    # "coco_10-2",
    # "coco_10-3",
    # "coco_10-4",
    # "coco_10-5",
    # "coco_15-1",
    # "coco_15-2",
    # "coco_15-3",
    # "coco_15-4",
    # "coco_15-5",
    # "coco_25-1",
    # "coco_25-2",
    # "coco_25-3",
    # "coco_25-4",
    # "coco_25-5",
    # "coco_30-1",
    # "coco_30-2",
    # "coco_30-3",
    # "coco_30-4",
    # "coco_30-5",
    # "coco_35-1",
    # "coco_35-2",
    # "coco_35-3",
    # "coco_35-4",
    # "coco_35-5",
    # "coco_45-1",
    # "coco_45-2",
    # "coco_45-3",
    # "coco_45-4",
    # "coco_45-5",
    # "coco_50-1",
    # "coco_50-2",
    # "coco_50-3",
    # "coco_50-4",
    # "coco_50-5",
    # "coco_55-1",
    # "coco_55-2",
    # "coco_55-3",
    # "coco_55-4",
    # "coco_55-5",
    # "coco_60-1",
    # "coco_60-2",
    # "coco_60-3",
    # "coco_60-4",
    # "coco_60-5",
    # "coco_65-1",
    # "coco_65-2",
    # "coco_65-3",
    # "coco_65-4",
    # "coco_65-5",
    # "coco_70-1",
    # "coco_70-2",
    # "coco_70-3",
    # "coco_70-4",
    # "coco_70-5",
    # "coco_75-1",
    # "coco_75-2",
    # "coco_75-3",
    # "coco_75-4",
    # "coco_75-5",
    # "coco_car",
    # "coco_dining table",
    # "coco_person",
    # "coco_umbrella",
    # "coco_potted plant",
    # "coco_horse",
    # "coco_bus",
    # "coco_motorcycle",
    # "coco_fork",
    # "coco_bed",
    # "coco_mouse",
    # "coco_food",
    # "coco_furniture",
    # "coco_outdoor",
    # "coco_sports",
    # "coco_appliance",
    # "coco_electronic",
    # "coco_animal",
    # "coco_indoor",
    # "coco_vehicle",
    # "coco_accessory",
    # "coco_kitchen",

    # "coco_couch",
    # "coco_tv",
]

subset_meta = {
    'coco': '''_base_ = ['./temp_base.py']
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='GenerateClassSubset', rule='{subset}',
         classes="data/{dataset}/annotations/metainfo.txt"),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'class_subset'))
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline))'''
}

def read_fps(logfile):
    with open(logfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'Overall fps:' in line:
            start = line.find('Overall fps:') + len('Overall fps: ')
            end = line.find(' img/s', start)
            return float(line[start:end])
    return -1.0

def read_mAP(logfile):
    with open(logfile, 'r') as f:
        lines = f.readlines()
    bAP = list()
    sAP = list()
    flag = False
    for line in lines:
        if '|' in line and 'category' not in line:
            ap = float(line.split('|')[2])
            if flag:
                sAP.append(ap)
            else:
                bAP.append(ap)
        elif 'bbox_mAP_copypaste' in line:
            flag = True
    valid_idx = [i for i in range(len(bAP)) if bAP[i] > 0 or sAP[i] > 0]
    if len(valid_idx) == 0:
        return 0.0, 0.0
    else:
        bAP = [bAP[i] for i in valid_idx]
        sAP = [sAP[i] for i in valid_idx]
    return sum(bAP)/len(bAP), sum(sAP)/len(sAP)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Test')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--test-metrics', type=str, default='mAP,fps')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--visualize', type=str, default=False)
    num_gpus = torch.cuda.device_count()

    args = parser.parse_args()

    if args.work_dir is None:
        args.work_dir = osp.join('work_dirs', osp.basename(args.config).replace('.py', ''))
    if not osp.exists(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)
    test_metrics = args.test_metrics.split(',')
    subsets = test_subsets if args.subset is None else args.subset.split(',')
    dataset = args.dataset
    if args.output is not None:
        logfile = args.output
        if not logfile.endswith('.csv'):
            logfile += '.csv'
    else:
        logfile = osp.join(args.work_dir, 'multi_test-{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logfile = open(logfile, 'w')
    logfile.write('subset')
    for metric in test_metrics:
        if metric.lower() == 'fps':
            logfile.write(',fps')
        elif metric.lower() == 'map':
            logfile.write(',bAP,sAP')
        else:
            raise NotImplementedError
    logfile.write('\n')

    base_config = Config.fromfile(args.config)

    for subset in subsets:
        logfile.write(subset)
        logfile.flush()
        config = 'temp/temp_config.py'
        base_config.dump("temp/temp_base.py")
        for metric in test_metrics:
            with open(config, 'w') as f:
                f.write(subset_meta[dataset].format(subset=subset, dataset=dataset))
            if metric.lower() == 'fps':
                cmd = f'python tools/analysis_tools/benchmark.py {config} --checkpoint {args.checkpoint} --task inference --work-dir temp --max-iter -1 --num-warmup 1000'
                os.system(cmd)
                shutil.move('temp/benchmark.log', osp.join(args.work_dir, f'{subset}_fps.log'))
                fps = read_fps(osp.join(args.work_dir, f'{subset}_fps.log'))
                logfile.write(f',{fps}')
            elif metric.lower() == 'map':
                if args.visualize:
                    prefix = subset.replace('coco_', '').replace(' ', '_')+"_"+args.visualize
                    cmd = f'./tools/dist_test.sh {config} {args.checkpoint} {num_gpus} --cfg-options work_dir="temp" model.test_cfg.save_prefix="{prefix}"'
                else:
                    cmd = f'./tools/dist_test.sh {config} {args.checkpoint} {num_gpus} --cfg-options work_dir="temp"'
                os.system(cmd)
                dir_list = list(filter(lambda x: osp.isdir(osp.join('temp', x)) and x.startswith('20'), os.listdir('temp')))
                cur_dir = sorted(dir_list)[-1]
                output_file = os.listdir(osp.join('temp', cur_dir))
                output_file = [f for f in output_file if f.endswith('.log')][0]
                output_file = osp.join('temp', cur_dir, output_file)
                shutil.move(output_file, osp.join(args.work_dir, f'{subset}_mAP.log'))
                bAP,sAP  = read_mAP(osp.join(args.work_dir, f'{subset}_mAP.log'))
                logfile.write(f',{bAP},{sAP}')
                shutil.rmtree(osp.join('temp', cur_dir))
            else:
                raise NotImplementedError
            logfile.flush()
        logfile.write('\n')

    