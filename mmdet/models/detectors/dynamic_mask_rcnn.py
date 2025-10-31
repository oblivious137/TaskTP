# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import ConfigDict
import numpy as np

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import distributed as dist
import torch.nn.functional as F


@MODELS.register_module()
class DynamicMaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
    
    def extract_feat(self, batch_inputs: Tensor, class_subsets: Tensor, forward_dict: dict) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs, class_subsets, forward_dict)
        if not self.training and self.test_cfg.get('backbone_only', False):
            return x
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        class_subsets = [data_sample.class_subset for data_sample in batch_data_samples]
        x = self.extract_feat(batch_inputs, class_subsets)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        forward_dict = dict()
        forward_dict['data_samples'] = batch_data_samples
        class_subsets = [data_sample.class_subset for data_sample in batch_data_samples]
        x = self.extract_feat(batch_inputs, class_subsets, forward_dict)

        losses = dict()

        backbone_data_samples = copy.deepcopy(batch_data_samples)
        backbone_losses = self.backbone.loss(forward_dict, backbone_data_samples)
        losses.update(backbone_losses)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        
        if self.train_cfg.keep_irrelevant_classes:
            class_subsets = None
        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples,
                                        class_subsets=class_subsets)
        # import ipdb; ipdb.set_trace()
        losses.update(roi_losses)

        # loss_rpn_cls = sum(losses['loss_rpn_cls']).detach().clone()
        # loss_rpn_bbox = sum(losses['loss_rpn_bbox']).detach().clone()
        # loss_cls = losses['loss_cls'].detach().clone()
        # loss_bbox = losses['loss_bbox'].detach().clone()
        # loss_mask = losses['loss_mask'].detach().clone()
        # acc = losses['acc'].detach().clone()

        
        # loss = loss_rpn_cls + loss_rpn_bbox + loss_cls + loss_bbox + loss_mask
        # # accumulate loss
        # dist.all_reduce(loss)
        # loss = loss / dist.get_world_size()
        # target_mean = 0.691480808080808
        # target_std = 0.03844015860121286
        # if not hasattr(self, 'loss_record'):
        #     self.loss_record = {
        #         'loss': target_mean
        #     }

        # # smmoth the loss, gamma=0.99
        # loss = 0.9999 * self.loss_record['loss'] + 0.0001 * loss.item()
        # self.loss_record['loss'] = loss
        # # generate a weight for the pruning loss, while the normal task loss distribution is close to the target distribution
        # # 1. if the loss is too high, the weight will be small, so that the pruning loss will be small
        # # 2. if the loss is too low, the weight will be large, so that the pruning loss will be large
        # # 3. if the loss is close to the target distribution, the weight will be 1
        # weight = np.power(np.e, -(loss - target_mean - target_std) / target_std)

        # acc_mean = 93.55271616161615
        # acc_std = 3.659729324696272
        # dist.all_reduce(acc)
        # acc = acc / dist.get_world_size()
        # if not hasattr(self, 'acc_record'):
        #     self.acc_record = {
        #         'acc': acc_mean
        #     }
        # acc = 0.9999 * self.acc_record['acc'] + 0.0001 * acc.item()
        # self.acc_record['acc'] = acc
        # weight = np.power(np.e, (acc - acc_mean) / acc_std)


        # losses['loss_prune'] = weight * losses['loss_prune']
        # losses['prune_weight'] = torch.tensor(weight)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        forward_dict = dict()
        forward_dict['data_samples'] = batch_data_samples
        class_subsets = [data_sample.class_subset for data_sample in batch_data_samples]
        x = self.extract_feat(batch_inputs, class_subsets, forward_dict)
        if not self.training and self.test_cfg.get('backbone_only', False):
            return x
        # activated_tokens = 0
        # all_tokens = 0
        # for layer_selected in backbone_forward_dict['selected']:
        #     for selected in layer_selected:
        #         all_tokens += selected.flatten().size(0)
        #         activated_tokens += (selected>0).sum().item()
        # print(f'Activated tokens: {activated_tokens} / {all_tokens}')

        # save score map to ./temp/date/img_idx_layer_idx.pth
        prefix = self.test_cfg.get('save_prefix', None)
        if prefix is not None:
            import os
            import torch
            import numpy as np
            # for idx, layer_selected in enumerate(forward_dict['select_mask']):
            for idx, layer_selected in enumerate(forward_dict['prune_score']):
                layer_selected = F.softmax(layer_selected, dim=-1)[:, 1]
                layer_selected = layer_selected.unsqueeze(0)
                log_scores = forward_dict['log_scores'][idx]
                for bid, data_sample in enumerate(batch_data_samples):
                    img_idx = data_sample.img_id
                    save_path = os.path.join('./temp', prefix, f'{img_idx}_{idx}.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    layer_selected_b = layer_selected[bid].reshape(1, 1, x[2].shape[2],x[2].shape[3])
                    layer_selected_b = F.interpolate(layer_selected_b, size=(data_sample.pad_shape[0], data_sample.pad_shape[1]), mode='nearest')
                    layer_selected_b = layer_selected_b[0,0,:data_sample.img_shape[0],:data_sample.img_shape[1]]
                    torch.save(layer_selected_b.cpu(), save_path)

                    save_path = os.path.join('./temp', prefix+'_semantic', f'{img_idx}_{idx}.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    score = torch.softmax(log_scores[bid], dim=-1).max(dim=-1)[0]
                    torch.save(score.cpu().reshape(x[2].shape[2],x[2].shape[3]), save_path)


        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale, class_subsets=class_subsets)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
