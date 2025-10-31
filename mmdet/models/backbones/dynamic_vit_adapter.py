"""Vision Transformer (ViT) in PyTorch.

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""
from collections import OrderedDict
from easydict import EasyDict
import copy
import logging
import warnings
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmengine.model import BaseModule
# from mmdet.models.utils.ckpt import load_state_dict
from mmengine.logging import MMLogger
from timm.models.layers import DropPath, Mlp, to_2tuple
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE
from .vit_adapter import ViTAdapter, Injector, Extractor
from mmdet.registry import MODELS
from timm.models.layers import trunc_normal_
from itertools import chain
from mmdet.models.losses import sigmoid_focal_loss
from mmdet.models.losses.focal_loss import py_focal_loss_with_prob, py_sigmoid_focal_loss
import numpy as np

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                                   (h // 16, w // 16),
                                                   (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2

class TokenClassificationBlock(nn.Module):
    # Selective Block: select tokens according to the categories to be detected
    def __init__(self, dim, act_layer=nn.GELU, drop=0., dynamic_cfg=EasyDict()):
        '''
        Args:
            dim: int, the dimension of the input tensor
            act_layer: nn.Module, the activation layer
            drop: float, the dropout rate
            num_categories: int, the number of categories
        '''
        super().__init__()
        self.dynamic_cfg = copy.deepcopy(dynamic_cfg)
        self.prune_action = dynamic_cfg.prune_action
        self.num_categories = dynamic_cfg.num_categories
        self.selector_gumbel_soft = dynamic_cfg.get('gumbel_soft', False)
        self.inherit_score = dynamic_cfg.get('inherit_score', False)
        self.straight_forward = dynamic_cfg.get('straight_forward', True)
        self.detach_before_classifier = dynamic_cfg.get('detach_before_classifier', False)
        self.feature_mode = dynamic_cfg.get('feature_mode', 'complex')
        self.inherit_classify = dynamic_cfg.get('inherit_classify', True)
        self.thr = dynamic_cfg.get('threshold', 0.5)
        self.dim = dim
        self.hidden_dim = dim // 4
        # assert self.inherit_score
        hidden_dim = dim // 4

        if 'topk-' in self.feature_mode:
            k = int(self.feature_mode.split('-')[-1])
            if 'noembed' in self.feature_mode:
                feature_dim = k*2
            else:
                feature_dim = k*(hidden_dim+1)*2
        elif self.feature_mode == 'complex':
            feature_dim = 4+3*hidden_dim
        else:
            raise NotImplementedError

        self.mlp = Mlp(dim, hidden_dim, self.num_categories+1 if dynamic_cfg.get('classwise', True) else 2, act_layer=act_layer, drop=drop)
        # self.mlp = nn.Linear(dim, self.num_categories+1 if dynamic_cfg.get('classwise', True) else 2)
        # self.score_shift = nn.Linear(3, 2)
        # self.score_shift = Mlp(self.num_categories+1, self.num_categories+1, 2, act_layer=act_layer)
        self.category_embed = nn.Embedding(self.num_categories+1, hidden_dim)
        if not self.dynamic_cfg.get('rule_base', False):
            self.score_shift = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                act_layer(),
                nn.Linear(hidden_dim, 2)
            )
        self.pre_norm = nn.LayerNorm(dim)
        self._init_weights()
        self.register_buffer('train_iter', torch.zeros(1))
    
    def _init_weights(self):
        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         trunc_normal_(module.weight, std=.02)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        # mlp = torch.load("/home/lianghao/prune-detection/work_dirs/dynamic-mask-rcnn_vit-t-p4-w7_fpn_1x_coco_gtguide_subset/segment_mlp.pth")
        # self.mlp.load_state_dict(mlp, strict=True, assign=True)
        # self.score_shift[-1].bias.data[1] = 0.1
        # self.score_shift[-1].bias.data[0] = -0.1
        # self.score_shift[-1].bias.data[1] = 1
        # self.score_shift[-1].bias.data[0] = -1
        return

    def forward(self, x, c, H, W, categories, forward_dict):
        '''
        Args:
            x: torch.Tensor, the input tensor
            categories: [torch.Tensor], the categories to be detected
        '''
        forward_dict['start_prune'] = True
        B, L, C = x.shape
        if self.detach_before_classifier:
            x = x.detach()
        x = self.pre_norm(x)
        logits = self.mlp(x)
        if 'log_scores' not in forward_dict:
            forward_dict['log_scores'] = []
        elif self.inherit_classify:
            logits = logits + forward_dict['log_scores'][-1]
        forward_dict['log_scores'].append(logits)
        
        if self.dynamic_cfg.get('detach_before_mask_mlp', True):
            logits = logits.detach()
        probs = F.softmax(logits, dim=-1)
        if logits.isnan().sum() > 0:
            import ipdb; ipdb.set_trace()

        if self.training:
            self.train_iter += 1
            
            # filter out the useless tokens
            masks = []
            scores = []
            for b in range(B):
                mask, score = self.get_mask(x[b], probs[b], categories[b], straight_forward=self.straight_forward, forward_dict=forward_dict, bid=b)
                masks.append(mask)
                scores.append(score)
            masks = torch.stack(masks, dim=0)
            scores = torch.stack(scores, dim=0)
            
            if 'prune_score' not in forward_dict:
                forward_dict['prune_score'] = []
            forward_dict['prune_score'].append(scores)

            if self.inherit_score and 'select_mask' in forward_dict:
                masks = masks * forward_dict['select_mask'][-1]
            if 'select_mask' not in forward_dict:
                forward_dict['select_mask'] = list()
            forward_dict['select_mask'].append(masks)
        else:
            assert B == 1
            if 'select_mask' not in forward_dict:
                selected, scores = self.get_mask(x[0], probs[0], categories[0], forward_dict=forward_dict, bid=0)
                forward_dict['select_mask'] = [selected]
                forward_dict['prune_score'] = [scores]
            else:
                # x_s = x[0][forward_dict['select_mask']]
                # assert self.inherit_score
                # logits = self.mlp(x_s)
                # forward_dict['select_mask'][forward_dict['select_mask'].clone()] = self.get_mask(logits, categories[0])
                selected, scores = self.get_mask(x[0], probs[0], categories[0], forward_dict=forward_dict, bid=0)
                if self.inherit_score:
                    selected *= forward_dict['select_mask'][-1]
                forward_dict['select_mask'].append(selected)
                forward_dict['prune_score'].append(scores)
            
        return forward_dict
    
    def get_top_k(self, probs, k, preserve=False):
        '''
        Args:
            probs: torch.Tensor, the probabilities of the tokens [N, C]
        '''
        if not preserve:
            if probs.shape[1] == 0:
                return torch.zeros((probs.shape[0], k), device=probs.device)
            elif probs.shape[1] < k:
                probs = torch.sort(probs, dim=-1, descending=True)[0]
                return torch.cat([probs, torch.zeros((probs.shape[0], k-probs.shape[1]), device=probs.device)], dim=-1)
            else:
                top_k = torch.topk(probs, k, dim=-1)[0]
                return top_k
        else:
            if probs.shape[1] == 0:
                return torch.zeros_like(probs)
            elif probs.shape[1] < k:
                return probs
            else:
                top_k = torch.topk(probs, k, dim=-1)[0][:, -1]
                return probs * (probs >= top_k.unsqueeze(-1)).float()

    def get_valid_top_k_embedding(self, probs, category_mask, k):
        '''
        Args:
            probs: torch.Tensor, the input tensor [N, C]
            category_mask: torch.Tensor, the mask of the categories [C]
            k: int, the number of top-k tokens to be selected
        '''
        category_indices = torch.nonzero(category_mask).flatten()
        valid_probs = probs[:, category_indices]
        if valid_probs.shape[1] == 0:
            return torch.zeros((probs.shape[0], k*self.dim), device=probs.device)
        elif valid_probs.shape[1] < k:
            probs, prob_indices = torch.sort(valid_probs, dim=-1, descending=True)
            actual_indices = category_indices[prob_indices.flatten()].reshape(prob_indices.shape) # [N, C]
            embeddings = torch.zeros((probs.shape[0], k*self.hidden_dim), device=probs.device) # [N, k*D]
            # embeddings[:, :valid_probs.shape[1]*self.hidden_dim] = (self.category_embed(actual_indices) * probs.unsqueeze(-1)).reshape(probs.shape[0], -1)
            embeddings[:, :valid_probs.shape[1]*self.hidden_dim] = self.category_embed(actual_indices).reshape(probs.shape[0], -1)
            return embeddings, torch.cat([probs, torch.zeros((probs.shape[0], k-probs.shape[1]), device=probs.device)], dim=-1)
        else:
            top_k, top_k_indices = torch.topk(valid_probs, k, dim=-1)
            actual_indices = category_indices[top_k_indices.flatten()].reshape(top_k_indices.shape) # [N, k]
            embeddings = (self.category_embed(actual_indices)).reshape(probs.shape[0], -1)
            return embeddings, top_k


    def get_mask(self, x, probs, categories, straight_forward=False, forward_dict=None, bid=0):
        '''
        Args:
            logits: torch.Tensor, the logits of the tokens [N, C]
            categories: torch.Tensor, the categories to be detected [K]
            straight_forward: bool, whether to use straight forward
        '''
        category_mask = forward_dict['category_mask'][bid]
        B = len(forward_dict['category_mask'])
        L, C = x.shape
        
        if self.dynamic_cfg.get('rule_base', False):
            valid_probs = probs[:, category_mask]
            max_prob = probs.max(dim=-1)[0]
            max_valid_prob = valid_probs.max(dim=-1)[0]
            mask = (max_prob - max_valid_prob) < 0.4
            # valid_probs = probs[:, category_mask]
            # maxp = self.get_top_k(valid_probs, 1)
            # mask = (maxp[:, 0]<0.6) & (maxp[:, 0]>1/80/4)
            if self.training:
                return mask.float(), torch.stack([torch.zeros_like(mask, dtype=torch.float)+0.5, mask.float()], dim=-1)
            else:
                return mask, torch.stack([torch.zeros_like(mask, dtype=torch.float)+0.5, mask.float()], dim=-1)
        
        # category_mask = category_mask.float()
        # valid_probs = probs[:, category_mask]
        # invalid_probs = probs[:, ~category_mask]
        # feat1 = self.get_top_k(valid_probs, 2)
        # feat2 = self.get_top_k(invalid_probs, 1)
        # feat3 = invalid_probs.sum(dim=-1)
        # feat4 = probs * (category_mask.float()*2-1).reshape(1, -1)
        # feat4 = (category_mask.float()*2-1).reshape(1, -1).repeat(probs.shape[0], 1)
        # feat = torch.cat([x, feat1, feat2, feat3.unsqueeze(-1)], dim=-1)
        
        # feat1 = self.get_valid_top_k_embedding(probs, ~category_mask, 1)
        # feat2 = self.get_valid_top_k_embedding(probs, category_mask, 2)
        # feat3 = x
        # feat = torch.cat([feat1, feat2], dim=-1)

        if self.feature_mode == 'complex':
            feat1, feat4 = self.get_valid_top_k_embedding(probs, category_mask, 2)
            feat2, feat5 = self.get_valid_top_k_embedding(probs, ~category_mask, 1)
            feat3 = probs[:, ~category_mask].sum(dim=-1, keepdim=True)
            feat1 = (feat1.reshape(L, 2, self.hidden_dim) * feat4.reshape(L, 2, 1)).reshape(L, -1)
            feat2 = (feat2.reshape(L, 1, self.hidden_dim) * feat5.reshape(L, 1, 1)).reshape(L, -1)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=-1)
        elif 'topk-' in self.feature_mode:
            topk = int(self.feature_mode.split('-')[-1])
            feat1, feat2 = self.get_valid_top_k_embedding(probs, category_mask, topk)
            feat3, feat4 = self.get_valid_top_k_embedding(probs, ~category_mask, topk)
            if 'noembed' in self.feature_mode:
                feat2 = feat2 + feat1.sum()*0
                feat4 = feat4 + feat3.sum()*0
                feat = torch.cat([feat2, feat4], dim=-1)
            # feat5 = probs[:, ~category_mask].sum(dim=-1, keepdim=True)
            else:
                feat = torch.cat([feat1, feat2, feat3, feat4], dim=-1)
        else:
            raise NotImplementedError

        score = self.score_shift(feat)
        # score = logits

        if self.training:
            # import ipdb; ipdb.set_trace()
            if self.selector_gumbel_soft:
                score = F.log_softmax(score, dim=-1)
                select_prob = F.gumbel_softmax(score, tau=1.0, hard=False)[:, 1]
            else:
                select_prob = F.softmax(score, dim=-1)[:, 1]

            # if 'select_prob' not in forward_dict:
            #     forward_dict['select_prob'] = []
            # elif len(forward_dict['select_prob']) >= B:
            #     select_prob = select_prob * forward_dict['select_prob'][-B]
            # forward_dict['select_prob'].append(select_prob)

            if straight_forward:
                hard_label = (select_prob > self.thr).float()
                while hard_label.sum().item() / hard_label.numel() < 0.01:
                    hard_label = hard_label + (torch.rand_like(hard_label) < 0.01).float()
                mask = hard_label - select_prob.detach() + select_prob
            else:
                mask = (select_prob > self.thr).float()
        else:
            select_prob = F.softmax(score, dim=-1)[:, 1]

            # if 'select_prob' not in forward_dict:
            #     forward_dict['select_prob'] = []
            # else:
            #     select_prob = select_prob * forward_dict['select_prob'][-1]
            # forward_dict['select_prob'].append(select_prob)
            mask = (select_prob > self.thr)
        # print(mask.float().mean())
        return mask, score

# default_dynamic_cfg = EasyDict(
#     topk=None,
#     dynamic_train_topk=False,
#     selector_grad_stop=True,
#     selector_gumbel_soft=False,
#     inherit_score=True,
#     prune_action='skip',
# )


class DynamicInteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False,
                 selector_layer_mask=None, dynamic_cfg=EasyDict()):
        super().__init__()
        
        self.selector_layer_mask = selector_layer_mask
        self.num_heads = num_heads
        self.dynamic_cfg = copy.deepcopy(dynamic_cfg)
        self.contain_selector = selector_layer_mask is not None and sum(selector_layer_mask) > 0
        self.fast_impl = selector_layer_mask is None or sum(selector_layer_mask)==0 or ((sum(selector_layer_mask)==1) and selector_layer_mask[0])
        # assert self.fast_impl
        if self.contain_selector:
            self.prune_action = self.dynamic_cfg.prune_action
            assert self.prune_action in ['skip', 'remove', 'straight']
            if self.prune_action == 'remove':
                assert self.inherit_score

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None
        self.register_buffer('train_iter', torch.zeros(1))
    
    def forward(self, x, c, blocks, selectors, deform_inputs1, deform_inputs2, H, W, forward_dict):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        
        B, L, C = x.shape
        class_subsets = forward_dict['class_subsets']

        if self.fast_impl:
            if self.contain_selector:
                forward_dict = selectors[0](x, c, H, W, class_subsets, forward_dict)
            
            if self.training:
                for idx, blk in enumerate(blocks):
                    if 'start_prune' in forward_dict:
                        select_mask = forward_dict['select_mask'][-1]
                        x = self.token_reduction_forward(select_mask, blk, x, H, W, forward_dict)
                    else:
                        x = blk(x, H, W)
            else:
                assert B == 1
                if 'start_prune' in forward_dict:
                    select_mask = forward_dict['select_mask'][-1]
                    x_s = x[0][select_mask].unsqueeze(0)
                    # x_background = x[0][~select_mask].unsqueeze(0).mean(dim=1, keepdim=True)
                    # x_s = torch.cat([x_s, x_background], dim=1)
                    for blk in blocks:
                        x_s = blk(x_s, H, W)
                    x[0][select_mask] = x_s.squeeze(0)
                else:
                    for blk in blocks:
                        x = blk(x, H, W)
        else:
            sid = 0
            for idx, blk in enumerate(blocks):
                if self.selector_layer_mask is not None and self.selector_layer_mask[idx]:
                    forward_dict = selectors[sid](x, c, H, W, class_subsets, forward_dict)
                    sid += 1
                if 'start_prune' in forward_dict:
                    select_mask = forward_dict['select_mask'][-1]
                    if self.training:
                        x = self.token_reduction_forward(select_mask, blk, x, H, W, forward_dict)
                    else:
                        x_s = x[0][select_mask].unsqueeze(0)
                        x_s = blk(x_s, H, W)
                        x[0][select_mask] = x_s.squeeze(0)
                else:
                    x = blk(x, H, W)

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        if self.training:
            self.train_iter += 1
        return x, c

    def token_reduction_forward(self, selected, blk, x, H, W, forward_dict):
        B, L, C = x.shape
        assert self.training
        identity = x.clone()

        # mask columns of unselected tokens
        # attn_mask = selected.float().unsqueeze(1).expand(-1, L, -1) * 100
        # attn_mask = attn_mask + torch.eye(L, device=attn_mask.device) * 200
        # attn_mask = torch.clamp(attn_mask-100, min=-100, max=1e-6)
        # x = blk(x, H, W, attn_mask=attn_mask)
        
        # 1. attn mask
        # xs = list()
        # for b in range(B):
        #     attn_mask = selected[b].unsqueeze(0).expand(L, -1) * 100
        #     attn_mask = attn_mask + torch.eye(L, device=attn_mask.device) * 200
        #     attn_mask = attn_mask - 100
        #     attn_mask = torch.clamp(attn_mask, max=1e-6)
        #     xs.append(blk(x[b].unsqueeze(0), H, W, attn_mask=attn_mask).squeeze(0))
        # x = torch.stack(xs, dim=0)
        
        # 2. key padding mask
        key_padding_mask = ~selected.bool()
        x = blk(x, H, W, key_padding_mask=key_padding_mask)
        
        x = x * selected.unsqueeze(-1) + identity * (1-selected).unsqueeze(-1)
        return x


@MODELS.register_module()
class DynamicViTAdapter(ViTAdapter):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True,
                 prune_layers=None,
                 keep_ratios=None,
                 lora_training=False,
                 freeze_base=False,
                 prune_loss=None,
                 dynamic_cfg=EasyDict(),
                 *args,
                 **kwargs):

        super().__init__(pretrain_size=pretrain_size, num_heads=num_heads, conv_inplane=conv_inplane,
                         n_points=n_points, deform_num_heads=deform_num_heads, init_values=init_values,
                         interaction_indexes=interaction_indexes, with_cffn=with_cffn, cffn_ratio=cffn_ratio,
                         deform_ratio=deform_ratio, add_vit_feature=add_vit_feature,
                         use_extra_extractor=use_extra_extractor, *args, **kwargs)
        self.keep_ratios = keep_ratios
        self.lora_training = lora_training
        self.freeze_base = freeze_base
        self.prune_loss = prune_loss
        self.dynamic_cfg = copy.deepcopy(dynamic_cfg)
        embed_dim = self.embed_dim
        is_prune = [i in prune_layers for i in range(len(self.blocks))]
        self.interactions = nn.Sequential(*[
            DynamicInteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             selector_layer_mask=is_prune[interaction_indexes[i][0]:interaction_indexes[i][-1] + 1],
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                             dynamic_cfg=dynamic_cfg)
            for i in range(len(interaction_indexes))
        ])
        self.interactions.apply(self._init_weights)
        self.selectors = nn.ModuleList([TokenClassificationBlock(dim=embed_dim, dynamic_cfg=dynamic_cfg) for _ in range(len(prune_layers))])
        
    def train(self, mode=True):
        super().train(mode)
        if self.lora_training and mode:
            self.eval()
            for param_name, param in self.named_parameters():
                if 'loras' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for module in self.modules():
                if isinstance(module, (DynamicInteractionBlock, TokenClassificationBlock)):
                    module.training = True # set training flag for selective and dynamic interaction blocks for passing gradient
        elif self.freeze_base and mode:
            self.eval()
            for module_name, module in self.named_modules():
                if isinstance(module, (TokenClassificationBlock, )):
                    module.train()
                elif isinstance(module, DynamicInteractionBlock):
                    module.training = True
            for param_name, param in self.named_parameters():
                if 'selectors' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x, class_subsets, forward_dict=None):
        if forward_dict is None:
            forward_dict = dict()
        forward_dict['class_subsets'] = class_subsets
        
        B, C, H, W = x.shape
        category_mask = torch.zeros(x.shape[0], self.dynamic_cfg.num_categories+1, device=x.device, dtype=torch.bool)
        for b in range(B):
            for c in class_subsets[b]:
                category_mask[b][c] = 1
        forward_dict['category_mask'] = category_mask
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        remove_flag = self.dynamic_cfg.get('remove_bg_by_gt', False)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        if remove_flag is not False:
            remove_ratio = float(remove_flag)
            data_samples = forward_dict['data_samples']
            gt = torch.zeros_like(x[..., 0])
            padshape = data_samples[0].batch_input_shape
            for b in range(len(data_samples)):
                for instance in data_samples[b].gt_instances:
                    mask = torch.from_numpy(instance.masks.masks).float().to(gt.device)
                    # padshape = data_samples[b].pad_shape # h, w
                    mask = F.pad(mask, (0, padshape[1]-mask.shape[2], 0, padshape[0]-mask.shape[1])).unsqueeze(0)
                    # print(padshape, mask.shape, gt.shape)
                    
                    # interpolate mask to the same size as the selected tensor (/16)
                    mask = F.interpolate(mask, scale_factor=1/16, mode='bilinear', align_corners=False)
                    gt[b] += mask.flatten()
            gt = gt + torch.rand_like(gt)
            gt = gt >= remove_ratio
            forward_dict['gt_mask'] = gt

        # Interaction
        sid=0
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            num_selectors = 0 if layer.contain_selector is False else sum(layer.selector_layer_mask)
            # x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1], self.selector,
            #              deform_inputs1, deform_inputs2, H, W, forward_dict=forward_dict)
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1], self.selectors[sid:sid+num_selectors],
                         deform_inputs1, deform_inputs2, H, W, forward_dict=forward_dict)
            sid += num_selectors
        assert sid == len(self.selectors)
        
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
    
    def loss(self, forward_dict, data_samples):
        losses = {}
        selected = forward_dict['select_mask']
        if self.prune_loss.type == 'SemanticLoss':
            logits = forward_dict['log_scores']
            B = len(data_samples)
            loss_semantic = 0
            prune_loss = 0
            prune_panalty_loss = 0
            gt_semantic = self.get_semantic(data_samples, self.dynamic_cfg.num_categories+1, logits[0].device)
            gt_semantic = gt_semantic.permute(0, 2, 3, 1).flatten(0,2)
            gt_semantic = gt_semantic.argmax(dim=-1)
            if not hasattr(self, 'label_count'):
                self.label_count = torch.zeros(self.dynamic_cfg.num_categories+1, device=logits[0].device) + 3800
            self.label_count += gt_semantic.flatten().bincount(minlength=self.dynamic_cfg.num_categories+1)
            max_count = self.label_count.max()
            label_weight = max_count / self.label_count
            label_weight.clamp_(0.01, 10)
            semantic_instance_weight = label_weight[gt_semantic.flatten()]
            # semantic_instance_weight = list()
            # for b in range(B):
            #     label_weight = forward_dict['category_mask'][b].float()
            #     semantic_instance_weight.append(label_weight[gt_semantic[b].flatten()])
            # semantic_instance_weight = torch.stack(semantic_instance_weight, dim=0).flatten()
            gt_valid = self.get_gt(data_samples).flatten().to(logits[0].device)
            for i, logit in enumerate(logits):
                layer_selected = selected[i]
                prob = F.softmax(logit.flatten(0, 1), dim=-1).clamp(1e-3, 1-1e-3)
                # predict_correct = (~((prob>0.5) ^ (gt_semantic>0.5))).all(dim=-1).float()
                predict_correct = (prob.argmax(dim=-1) == gt_semantic).float()
                max_prob = prob.max(dim=-1)[0].detach()
                losses[f'maxp_{i}'] = max_prob.mean()
                losses[f'acc_{i}'] = predict_correct.mean()
                losses[f'keep_ratio_{i}'] = layer_selected.mean()

                semantic_weight = self.prune_loss.semantic_weight[i] if isinstance(self.prune_loss.semantic_weight, (list, tuple)) else self.prune_loss.semantic_weight
                instance_loss = py_focal_loss_with_prob(prob, gt_semantic, weight=semantic_instance_weight, gamma=2.0, alpha=0.25, reduction='mean')
                loss_semantic += semantic_weight * instance_loss

                # complex loss
                # # 1. invalid categories should be removed
                # keep_gt = gt_valid.clone().float()
                # # 2. well predicted valid categories should be removed
                # keep_gt -= predict_correct
                # keep_gt[keep_gt<0.5] = 0
                # keep_gt[keep_gt>=0.5] = 1

                # # 1. wrong predict from the valid categories should be kept
                # # keep_gt += ((predict_correct == 0) & gt_valid).float()

                # # 2. wrong predict to the valid categories should be kept
                # # predict_valids = list()
                # # for b in range(B):
                # #     category_mask = forward_dict['category_mask'][b]
                # #     predict_valid = category_mask[predict_label.reshape(B, -1)[b]].float()
                # #     predict_valids.append(predict_valid)
                # # predict_valids = torch.stack(predict_valids, dim=0).flatten()
                # # keep_gt += (predict_correct == 0).float() * predict_valids
                # # keep_gt.clamp_(0, 1)
                
                # instance_weight = torch.ones_like(keep_gt)
                # instance_weight[gt_valid & (keep_gt==0)] = max_prob[(gt_valid==1) & (keep_gt==0)]
                # # top2_prob = prob.topk(2, dim=-1)[0].reshape(B, -1, 2)
                # gt_valid = gt_valid.reshape(B, -1)
                # keep_gt = keep_gt.reshape(B, -1)
                # prob = prob.reshape(B, -1, prob.shape[-1])
                # instance_weight.reshape_as(keep_gt)
                # # for b in range(B):
                # #     category_mask = forward_dict['category_mask'][b]
                # #     # 1. for the correct predictions (pruned), if it is not in the valid categories, the prune weight should be sum of invalid probabilities
                # #     # print(prob.shape, gt_valid.shape, category_mask.shape)
                # #     # print(prob[b][~gt_valid[b]].shape)
                # #     instance_weight[b][~gt_valid[b]] = 1 - prob[b][~gt_valid[b]][:, ~category_mask].sum(dim=-1)
                # #     # 2. if it is in the valid categories, the prune weight should be the max of the valid probabilities
                # #     if category_mask.sum() > 0:
                # #         instance_weight[b][gt_valid[b]] = top2_prob[b][gt_valid[b], 0]
                # instance_weight = instance_weight.flatten()
                # keep_gt = keep_gt.flatten()
                # gt_valid = gt_valid.flatten()
                # prob = prob.reshape(-1, prob.shape[-1])
                # instance_weight.clamp_(0.01, 1)
                
                # # balance the weight of the positive and negative samples
                # num_pos = keep_gt.sum()
                # num_neg = instance_weight.sum() - num_pos
                # instance_weight[keep_gt==1] = instance_weight[keep_gt==1] * (num_neg + 1e-5) / (num_pos + 1e-5)
                # # instance_weight[keep_gt==1] *= 2
                # instance_weight = instance_weight.detach().reshape_as(layer_selected)


                # simple loss
                keep_gt = torch.ones_like(gt_valid, dtype=torch.float)
                instance_weight = torch.ones_like(keep_gt)
                if self.prune_loss.get("early_stop", True):
                    keep_gt -= predict_correct
                    instance_weight[predict_correct==1] = max_prob[predict_correct==1]

                gt_valid = gt_valid.reshape(B, -1)
                keep_gt = keep_gt.reshape(B, -1)
                prob = prob.reshape(B, -1, prob.shape[-1])
                instance_weight = instance_weight.reshape(B, -1)
                max_prob = max_prob.reshape(B, -1)

                for b in range(B):
                    category_mask = forward_dict['category_mask'][b]
                    # 1. if a token is predict from the invalid categories to the invalid categories, it can be removed
                    pred_invalid = ~category_mask[prob[b].argmax(dim=-1)]
                    mask = pred_invalid & ~gt_valid[b]
                    keep_gt[b][mask] = 0
                    # instance_weight[b][mask] = prob[b][mask][:, ~category_mask].sum(dim=-1)
                    instance_weight[b][mask] = max_prob[b][mask]
                instance_weight = instance_weight.flatten()
                keep_gt = keep_gt.flatten()
                gt_valid = gt_valid.flatten()
                prob = prob.reshape(-1, prob.shape[-1])
                keep_gt.clamp_(0, 1)
                instance_weight.clamp_(min=0)
                # instance_weight = instance_weight * semantic_instance_weight.reshape_as(instance_weight)
                num_pos = instance_weight[keep_gt==1].sum()
                num_neg = instance_weight.sum() - num_pos
                instance_weight[keep_gt==1] = instance_weight[keep_gt==1] * (num_neg + 1e-5) / (num_pos + 1e-5)
                # instance_weight[keep_gt==1] *= 2
                instance_weight = instance_weight.detach().reshape_as(layer_selected)






                # calculate the loss
                ratio_weight = self.prune_loss.ratio_weight[i] if isinstance(self.prune_loss.ratio_weight, (list, tuple)) else self.prune_loss.ratio_weight
                prune_score = forward_dict['prune_score'][i]
                pred_mask = prune_score.flatten(0, 1).argmax(dim=-1)
                fn_mask = (pred_mask==0) & (keep_gt==1)
                fp_mask = (pred_mask==1) & (keep_gt==0)
                # instance_weight[fn_mask.reshape_as(instance_weight)] *= 4
                instance_loss = F.cross_entropy(prune_score.flatten(0, 1), keep_gt.long(), reduction='none') * instance_weight.detach().flatten()
                # losses[f'prune_acc_{i}'] = (pred_mask == keep_gt.long()).float().mean()
                losses[f'prune_fn_{i}'] = fn_mask.float().sum() / ((1-pred_mask).sum()+1e-5)
                losses[f'remove_gt_{i}'] = (fn_mask & gt_valid).float().sum() / (gt_valid.sum()+1e-5)
                # import ipdb; ipdb.set_trace()
                # instance_loss = F.mse_loss(layer_selected.flatten(), keep_gt, reduction='none') * instance_weight.flatten()
                # prune_prob = F.softmax(prune_score.flatten(0, 1), dim=-1)
                # instance_loss = py_focal_loss_with_prob(prune_prob, keep_gt.long(), weight=instance_weight.flatten(), gamma=2.0, alpha=0.25, reduction='none')
                # keep_gt = torch.stack([1-keep_gt, keep_gt], dim=-1)
                # instance_loss = F.binary_cross_entropy(prune_score.flatten(0, 1), keep_gt, reduction='none') * instance_weight.flatten().unsqueeze(-1)
                prune_loss += ratio_weight * instance_loss.mean()

                prune_ratio_panalty = self.prune_loss.prune_ratio_panalty if isinstance(self.prune_loss.prune_ratio_panalty, float) else self.prune_loss.prune_ratio_panalty[i]
                keep_ratio = layer_selected.mean()
                target_keep_ratio = self.keep_ratios[i] if isinstance(self.keep_ratios, (list, tuple)) else self.keep_ratios
                prune_panalty_loss += prune_ratio_panalty * (keep_ratio-target_keep_ratio)**2
                losses[f'keep_ratio_{i}'] = keep_ratio.detach()
            
            if (loss_semantic + prune_loss + prune_panalty_loss).isnan().any():
                import ipdb; ipdb.set_trace()
            loss_semantic /= len(logits)
            prune_loss /= len(selected)
            prune_panalty_loss /= len(selected)
            if isinstance(loss_semantic, torch.Tensor):
                losses['loss_semantic'] = loss_semantic
            if isinstance(prune_loss, torch.Tensor):
                losses['loss_prune'] = prune_loss
            if isinstance(prune_panalty_loss, torch.Tensor):
                losses['loss_prune_panalty'] = prune_panalty_loss
        return losses
    
    def get_semantic(self, data_samples, num_channels, device='cuda'):
        padshape = data_samples[0].batch_input_shape
        tokenH, tokenW = padshape[0] // 16, padshape[1] // 16
        gt = torch.zeros(len(data_samples), num_channels, padshape[0], padshape[1], dtype=torch.float, device=device)
        gt[:, -1] = 1
        for b in range(len(data_samples)):
            for instance in chain(data_samples[b].ignored_instances, data_samples[b].gt_instances):
                mask = torch.from_numpy(instance.masks.masks).float().to(device)
                mask = F.pad(mask, (0, padshape[1]-mask.shape[2], 0, padshape[0]-mask.shape[1])).squeeze(0)
                label = instance.labels.item()
                if num_channels == 2:
                    label = 0
                gt[b][label][mask > 0] = 1
                gt[b][-1][mask>0] = 0
        gt = F.interpolate(gt, size=(tokenH, tokenW), mode='bilinear')
        return gt
    
    def get_gt(self, data_samples):
        padshape = data_samples[0].batch_input_shape
        tokenH, tokenW = padshape[0] // 16, padshape[1] // 16
        gt = torch.zeros(len(data_samples), tokenH * tokenW)
        for b in range(len(data_samples)):
            for instance in data_samples[b].gt_instances:
                if instance.labels.item() not in data_samples[b].class_subset:
                    continue
                mask = torch.from_numpy(instance.masks.masks).float()
                mask = F.pad(mask, (0, padshape[1]-mask.shape[2], 0, padshape[0]-mask.shape[1])).unsqueeze(0)
                mask = F.interpolate(mask, scale_factor=1/16, mode='nearest')
                gt[b] += mask.flatten()
        gt = gt > 0
        return gt
    
    def get_ignored_gt(self, data_samples):
        padshape = data_samples[0].batch_input_shape
        tokenH, tokenW = padshape[0] // 16, padshape[1] // 16
        gt = torch.zeros(len(data_samples), tokenH * tokenW)
        for b in range(len(data_samples)):
            for instance in chain(data_samples[b].ignored_instances, data_samples[b].gt_instances):
                if instance.labels.item() not in data_samples[b].class_subset:
                    mask = torch.from_numpy(instance.masks.masks).float()
                    mask = F.pad(mask, (0, padshape[1]-mask.shape[2], 0, padshape[0]-mask.shape[1])).unsqueeze(0)
                    mask = F.interpolate(mask, scale_factor=1/16, mode='nearest')
                    gt[b] += mask.flatten()
        gt = gt > 0
        return gt

def save_tensor_as_image(tensor, filename):
    import cv2
    tensor = tensor.detach().cpu().numpy()
    while tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor.transpose(1, 2, 0)
        tensor = tensor.astype(np.float32) * 255
        tensor = tensor.astype(np.uint8)
        cv2.imwrite(filename, tensor)
    else:
        tensor = tensor.astype(np.float32) * 255
        tensor = tensor.astype(np.uint8)
        cv2.imwrite(filename, tensor)