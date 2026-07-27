# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet3d.models.builder import NECKS
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import (SparseConvTensor, SparseSequential,
                                SparseConvTranspose3d)
else:
    from mmcv.ops import (SparseConvTensor, SparseSequential,
                          SparseConvTranspose3d)

# import spconv.pytorch as spconv
import torch.nn.functional as F

@NECKS.register_module()
class SECONDFPN_3d(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
             in_channels=[128, 128, 256],
             out_channels=[256, 256, 256],
             upsample_strides=[1, 2, 4],
             norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
             upsample_cfg=dict(type='SparseConvTranspose3d'),  # 修改为稀疏反卷积
             conv_cfg=dict(type='SubMConv3d'),  # 修改为稀疏卷积
             use_conv_for_no_stride=False,
             init_cfg=None):
        super(SECONDFPN_3d, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        sparse_cnt=0
        sparse_cnt2=0

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                # 使用稀疏反卷积进行上采样
                # upsample_layer = spconv.SparseConvTranspose3d(
                upsample_layer = SparseConvTranspose3d(
                in_channels=in_channels[i],
                out_channels=out_channel,
                kernel_size=stride,  # 可以与 stride 相同，决定上采样比例
                stride=stride,  # 上采样倍数
                indice_key=f'spconv_up_{i}',  # 确保 indice_key 不冲突
                bias=False
            )
            else:
                # 使用稀疏卷积
                conv_cfg.update({'indice_key':f'subm_secondfpn_{sparse_cnt}'})
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,  # kernel_size 对应 stride
                    stride=stride,
                    padding=1)
                sparse_cnt+=1

            deblock = SparseSequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True)
            )
            deblocks.append(deblock)

        # 存储多个卷积块
        self.deblocks = nn.ModuleList(deblocks)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x: sparse tensor

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out=[]
            for i in range(len(ups)):
                out.append(ups[i].dense())
            out = torch.cat(out, dim=1)
        else:
            out = ups[0].dense()
        return [out]
    
    
@NECKS.register_module()
class SECONDFPN_3dv2(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    want to not use sparseConvTranspose3d, cause it is said to be slow
    use F.upsample or F.interpolate, this may faster or slower

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
             in_channels=[128, 128, 256],
             out_channels=[256, 256, 256],
             upsample_strides=[1, 2, 4],
             norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
             upsample_cfg=dict(type='SparseConvTranspose3d'),  # 修改为稀疏反卷积
             conv_cfg=dict(type='SubMConv3d'),  # 修改为稀疏卷积
             use_conv_for_no_stride=False,
             init_cfg=None):
        super(SECONDFPN_3dv2, self).__init__(init_cfg=init_cfg)
        # assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        sparse_cnt=0
        sparse_cnt2=0

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                # 使用稀疏反卷积进行上采样
                # upsample_layer = spconv.SparseConvTranspose3d(
                upsample_layer = SparseConvTranspose3d(
                in_channels=in_channels[i],
                out_channels=out_channel,
                kernel_size=stride,  # 可以与 stride 相同，决定上采样比例
                stride=stride,  # 上采样倍数
                indice_key=f'spconv_up_{i}',  # 确保 indice_key 不冲突
                bias=False
            )
            else:
                # 使用稀疏卷积
                conv_cfg.update({'indice_key':f'subm_secondfpn_{sparse_cnt}'})
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,  # kernel_size 对应 stride
                    stride=stride,
                    padding=1)
                sparse_cnt+=1

            deblock = SparseSequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True)
            )
            deblocks.append(deblock)

        # 存储多个卷积块
        self.deblocks = nn.ModuleList(deblocks)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x: sparse tensor

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
    
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        out=[]
        for i in range(len(ups)):
            out.append(ups[i].dense())
        out.append(F.interpolate(x[-1].dense(),scale_factor=2,mode='trilinear',align_corners=False))
        out = torch.cat(out, dim=1)

        return [out]
    

@NECKS.register_module()
class SECONDFPN_3dv3(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    want to not use sparseConvTranspose3d, cause it is said to be slow
    use F.upsample or F.interpolate, this may faster

    built upon SECONDFPN_3dv2, but project the feature space to 256

    change: 1106 
    and return the feature map in tensor, not list
    cause if return in a list, should change the input to tensor in train/test_offline/test_online
    it will be too troublesome

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
             in_channels=[128, 128, 256],
             out_channels=[256, 256, 256],
             upsample_strides=[1, 2, 4],
             norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
             upsample_cfg=dict(type='SparseConvTranspose3d'),  # 修改为稀疏反卷积
             conv_cfg=dict(type='SubMConv3d'),  # 修改为稀疏卷积
             use_conv_for_no_stride=False,
             init_cfg=None):
        super(SECONDFPN_3dv3, self).__init__(init_cfg=init_cfg)
        # assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        ## hardcode
        self.proj_feat=nn.Sequential(
            nn.Linear(512,256),
            nn.Softplus(),
            nn.Linear(256,256)
        )

        deblocks = []
        sparse_cnt=0
        sparse_cnt2=0

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                # 使用稀疏反卷积进行上采样
                # upsample_layer = spconv.SparseConvTranspose3d(
                upsample_layer = SparseConvTranspose3d(
                in_channels=in_channels[i],
                out_channels=out_channel,
                kernel_size=stride,  # 可以与 stride 相同，决定上采样比例
                stride=stride,  # 上采样倍数
                indice_key=f'spconv_up_{i}',  # 确保 indice_key 不冲突
                bias=False
            )
            else:
                # 使用稀疏卷积
                conv_cfg.update({'indice_key':f'subm_secondfpn_{sparse_cnt}'})
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,  # kernel_size 对应 stride
                    stride=stride,
                    padding=1)
                sparse_cnt+=1

            deblock = SparseSequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True)
            )
            deblocks.append(deblock)

        # 存储多个卷积块
        self.deblocks = nn.ModuleList(deblocks)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x: sparse tensor

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
    
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        out=[]
        for i in range(len(ups)):
            out.append(ups[i].dense())
        out.append(F.interpolate(x[-1].dense(),scale_factor=2,mode='trilinear',align_corners=False))
        out = torch.cat(out, dim=1)

        # B,C,Dz,Dy,Dx -> B,Dz,Dy,Dx,C
        out=out.permute(0,2,3,4,1)
        out=self.proj_feat(out)
        out=out.permute(0,4,1,2,3) # B,C,Dz,Dy,Dx

        return out
