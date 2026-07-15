# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from ..utils import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        # super().__init__(**kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim) # 统一维度
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(          # 融合上采样到1/4并且拼接到一起
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.linear_pred(_c)

        return x


class SegFormerHead2(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead2, self).__init__(input_transform=None, **kwargs)
        # super().__init__(**kwargs)
        # assert len(feature_strides) == len(self.in_channels)
        # assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=self.in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim*2)
        self.linear_c1 = MLP(input_dim=embedding_dim*2, embed_dim=embedding_dim*3)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*3,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # print(inputs[0].shape, inputs[1].shape, inputs[2].shape, inputs[3].shape)

        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        # c1, c2, c3, c4 = x

        # print(c1.shape, c2.shape, c3.shape, c4.shape)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = inputs.shape
        # print(inputs.shape)

        # _c4 = self.linear_c4(inputs).permute(0,2,1).reshape(n, -1, inputs.shape[2], inputs.shape[3])
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        #
        # _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        #
        # _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        # _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        #
        # _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        #
        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _c4 = self.linear_c4(inputs).permute(0,2,1).reshape(n, -1, inputs.shape[2], inputs.shape[3])
        _c4 = resize(_c4, size=(h * 2, w * 2),mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(_c4).permute(0,2,1).reshape(n, -1, _c4.shape[2], _c4.shape[3])
        _c3 = resize(_c3, size=(h * 3, w * 3),mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(_c3).permute(0,2,1).reshape(n, -1, _c3.shape[2], _c3.shape[3])
        _c2 = resize(_c2, size=(h * 4, w * 4),mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(_c2).permute(0,2,1).reshape(n, -1, _c2.shape[2], _c2.shape[3])

        _c = self.linear_fuse(_c1)

        # x = self.dropout(_c)
        x = self.linear_pred(_c)

        return x

