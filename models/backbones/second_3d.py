# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet3d.models.builder import BACKBONES

from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

def make_sparse_convmodule(in_channels,
                           out_channels,
                           kernel_size,
                           indice_key,
                           stride=1,
                           padding=0,
                           conv_type='SubMConv3d',
                           norm_cfg=None,
                           order=('conv', 'norm', 'act')):
    """Make sparse convolution module.

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of out channels
        kernel_size (int|tuple(int)): kernel size of convolution
        indice_key (str): the indice key used for sparse tensor
        stride (int|tuple(int)): the stride of convolution
        padding (int or list[int]): the padding number of input
        conv_type (str): sparse conv type in spconv
        norm_cfg (dict[str]): config of normalization layer
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {'conv', 'norm', 'act'} == {'conv', 'norm', 'act'}

    conv_cfg = dict(type=conv_type, indice_key=indice_key)

    layers = list()
    for layer in order:
        if layer == 'conv':
            if conv_type not in [
                    'SparseInverseConv3d', 'SparseInverseConv2d',
                    'SparseInverseConv1d'
            ]:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False))
            else:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        bias=False))
        elif layer == 'norm':
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == 'act':
            layers.append(nn.ReLU(inplace=True))

    layers = SparseSequential(*layers)
    return layers

@BACKBONES.register_module()
class SECOND_3d(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.

        use_sparse_conv: (int) the sparse conv layer,note that sparse conv can devoid
            the feature
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 sparse_conv_cnt=0,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(SECOND_3d, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        self.sparse_conv_cnt=sparse_conv_cnt

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        sparse_cnt=0
        sparse_cnt2=0
        for i, layer_num in enumerate(layer_nums):
            if layer_strides[i]<=1:
                conv_cfg = dict(type='SubMConv3d', indice_key=f'subm_second_{sparse_cnt}')
            else:
                conv_cfg = dict(type='SparseConv3d', indice_key=f'spconv_second_{sparse_cnt2}')
            block = [
                build_conv_layer(
                        conv_cfg,
                        in_filters[i],
                        out_channels[i],
                        3,
                        stride=layer_strides[i],
                        padding=1,
                        bias=False),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            if layer_strides[i]>1:
                self.sparse_conv_cnt=0
            for j in range(layer_num):
                if layer_num-j>self.sparse_conv_cnt:
                    sparse_cnt+=1
                    conv_cfg = dict(type='SubMConv3d', indice_key=f'subm_second_{sparse_cnt}')
                    block.append(
                        build_conv_layer(
                            conv_cfg,
                            out_channels[i],
                            out_channels[i],
                            3,
                            padding=1))
                    block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                    block.append(nn.ReLU(inplace=True))
                else:
                    conv_cfg = dict(type='SparseConv3d', indice_key=f'spconv_second_{sparse_cnt2}')
                    block.append(
                        build_conv_layer(
                            conv_cfg,
                            out_channels[i],
                            out_channels[i],
                            3,
                            padding=1))
                    block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                    block.append(nn.ReLU(inplace=True))
                    sparse_cnt2+=1

            block = SparseSequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)


    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): sparse tensor

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
