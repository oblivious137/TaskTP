# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mmdet.registry import MODELS
from .adapter_modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_
from .base.tome_atc_vit import merge_wavg

from .base.tome_atc_vit import ToMeATCVisionTransformer
from .adapter_modules import SpatialPriorModule, deform_inputs, Injector, Extractor

_logger = logging.getLogger(__name__)


class InteractionBlockToMeATC(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False, injector_deform_attn=True, extractor_deform_attn=True):
        super().__init__()
        
        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp, deform_attn=injector_deform_attn)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp, deform_attn=extractor_deform_attn)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp, deform_attn=extractor_deform_attn)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None
    
    def unmerge_features(self, x, unmerge_list):
        for un_idx in reversed(range(len(unmerge_list))):
                if unmerge_list[un_idx] is not None:
                    x = unmerge_list[un_idx](x)
        return x
    
    def merge_features(self, x, merge_list, prop_attn):
        attn_size = None
        for m_idx in range(len(merge_list)):
                if merge_list[m_idx] is not None:
                    x, attn_size = merge_wavg(merge_list[m_idx], x, attn_size)
                    if not prop_attn:
                        attn_size = None
        return x
    
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, attn_size, prop_attn, merge_list, unmerge_list):

        if self.injector.deform_attn and len(unmerge_list) > 0:
            x = self.unmerge_features(x, unmerge_list)
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        if self.injector.deform_attn and len(merge_list) > 0:
            x = self.merge_features(x, merge_list, prop_attn)
        
        for idx, blk in enumerate(blocks):
            x, attn_size, merge, unmerge = blk(x, attn_size)
            
            if not prop_attn:
                attn_size = None
            merge_list.append(merge)
            unmerge_list.append(unmerge)

        if self.extractor.deform_attn and len(unmerge_list) > 0:
            x_extractor = self.unmerge_features(x, unmerge_list)
        else:
            x_extractor = x

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x_extractor, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)

        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x_extractor, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, attn_size, merge_list, unmerge_list


@MODELS.register_module()
class ToMeATCViTAdapter(ToMeATCVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True, injector_deform_attn=True, extractor_deform_attn=True, *args, **kwargs):

        super().__init__(num_heads=num_heads, *args, **kwargs)

        # self.num_classes = 80
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlockToMeATC(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                             injector_deform_attn=injector_deform_attn, extractor_deform_attn=extractor_deform_attn)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        attn_size = None
        merge_list = []
        unmerge_list = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, attn_size, merge_list, unmerge_list = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                                    deform_inputs1, deform_inputs2, H, W, attn_size, self.prop_attn, merge_list, unmerge_list)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            for un_idx in reversed(range(len(unmerge_list))):
                if unmerge_list[un_idx] is not None:
                    x = unmerge_list[un_idx](x)
            assert x.shape == (bs, n, dim)

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
