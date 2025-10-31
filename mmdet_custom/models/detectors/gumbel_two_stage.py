import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import cv2

from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.registry import MODELS
from mmdet.structures import SampleList

@MODELS.register_module()
class GumbelTwoStageDetector(TwoStageDetector):
    """ Compared to TwoStageDetector, GumbelTwoStageDetector only adds an additional training loss for the backbone"""

    def __init__(self, ratio_loss_weight=1., **kwargs):
        super(GumbelTwoStageDetector, self).__init__(**kwargs)
        self.ratio_loss_weight = ratio_loss_weight

    def extract_feat(self, img, need_loss=False):
        """Directly extract features from the backbone+neck."""
        out = self.backbone(img, need_loss)
        if need_loss:
            assert isinstance(out, tuple) and len(out) == 2
            x, loss = out
            if self.with_neck:
                x = self.neck(x)
            return x, loss
        else:
            if self.with_neck:
                out = self.neck(out)
            return out

    def predict(self, batch_inputs, batch_data_samples, rescale = True):
        
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        if not self.training and self.test_cfg.get('backbone_only', False):
            return x
                    
        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        prefix = self.test_cfg.get('save_prefix', None)
        if prefix is not None:
            import os
            import torch
            import numpy as np
            for idx, layer_selected in enumerate(self.backbone.forward_dict['masks']):
                for bid, data_sample in enumerate(batch_data_samples):
                    img_idx = data_sample.img_id
                    save_path = os.path.join('./temp', prefix, f'{img_idx}_{idx}.png')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    layer_selected_b = layer_selected[bid].reshape(1, 1, x[2].shape[2],x[2].shape[3]).float()
                    layer_selected_b = F.interpolate(layer_selected_b, size=(data_sample.pad_shape[0], data_sample.pad_shape[1]), mode='nearest')
                    layer_selected_b = layer_selected_b[0,0,:data_sample.img_shape[0],:data_sample.img_shape[1]]
                    mask_img = layer_selected_b.cpu().numpy() * 255
                    mask_img = mask_img.astype(np.uint8)
                    mask_img = np.clip(mask_img, 0, 255)
                    cv2.imwrite(save_path, mask_img)

        return batch_data_samples

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
        x, loss_backbone = self.extract_feat(batch_inputs, need_loss=True)

        losses = dict()
        loss_backbone['backbone.ratio_loss'] *= self.ratio_loss_weight
        losses.update(loss_backbone)

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

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses
    