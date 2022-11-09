# Copyright (c) OpenMMLab. All rights reserved.
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=False, 
                 activation="leaky_relu", 
                 order=("conv", "norm", "act"),
                 act_inplace=True):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation, 
                              groups=groups, 
                              bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        if activation:
            if activation == "leaky_relu":
                self.act = nn.LeakyReLU(negative_slope=0.01, inplace=act_inplace)
            elif activation == "silu":
                self.act = nn.SiLU(inplace=act_inplace)
            elif activation == "gelu": 
                self.act = nn.GELU()
        else:
            self.act = nn.Identity()
        self.order = order

    def forward(self, x):
        for i in self.order:
            x = getattr(self, i)(x) 
        return x 


class BaseMergeCell(nn.Module):
    """The basic class for cells used in NAS-FPN and NAS-FCOS.
    BaseMergeCell takes 2 inputs. After applying convolution
    on them, they are resized to the target size. Then,
    they go through binary_op, which depends on the type of cell.
    If with_out_conv is True, the result of output will go through
    another convolution layer.
    Args:
        in_channels (int): number of input channels in out_conv layer.
        out_channels (int): number of output channels in out_conv layer.
        with_out_conv (bool): Whether to use out_conv layer
        out_conv_cfg (dict): Config dict for convolution layer, which should
            contain "groups", "kernel_size", "padding", "bias" to build
            out_conv layer.
        out_norm_cfg (dict): Config dict for normalization layer in out_conv.
        out_conv_order (tuple): The order of conv/norm/activation layers in
            out_conv.
        with_input1_conv (bool): Whether to use convolution on input1.
        with_input2_conv (bool): Whether to use convolution on input2.
        input_conv_cfg (dict): Config dict for building input1_conv layer and
            input2_conv layer, which is expected to contain the type of
            convolution.
            Default: None, which means using conv2d.
        input_norm_cfg (dict): Config dict for normalization layer in
            input1_conv and input2_conv layer. Default: None.
        upsample_mode (str): Interpolation method used to resize the output
            of input1_conv and input2_conv to target size. Currently, we
            support ['nearest', 'bilinear']. Default: 'nearest'.
    """

    def __init__(self,
                 fused_channels=256,
                 out_channels=256,
                 with_out_conv=True,
                 out_conv_cfg=dict(
                     groups=1, kernel_size=3, padding=1, bias=True),
                 out_conv_order=('act', 'conv', 'norm'),
                 with_input1_conv=False,
                 with_input2_conv=False,
                 upsample_mode='nearest'):
        super().__init__()
        assert upsample_mode in ['nearest', 'bilinear']
        self.with_out_conv = with_out_conv
        self.with_input1_conv = with_input1_conv
        self.with_input2_conv = with_input2_conv
        self.upsample_mode = upsample_mode

        if self.with_out_conv:
            self.out_conv = ConvModule(
                fused_channels,
                out_channels,
                **out_conv_cfg,
                order=out_conv_order)

        self.input1_conv = self._build_input_conv(
            out_channels) if with_input1_conv else nn.Sequential()
        self.input2_conv = self._build_input_conv(
            out_channels) if with_input2_conv else nn.Sequential()

    def _build_input_conv(self, channel):
        return ConvModule(
            channel,
            channel,
            3,
            padding=1,
            bias=True)

    @abstractmethod
    def _binary_op(self, x1, x2):
        pass

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)
        else:
            if x.shape[-2] % size[-2] != 0 or x.shape[-1] % size[-1] != 0:
                h, w = x.shape[-2:]
                target_h, target_w = size
                pad_h = math.ceil(h / target_h) * target_h - h
                pad_w = math.ceil(w / target_w) * target_w - w
                pad_l = pad_w // 2
                pad_r = pad_w - pad_l
                pad_t = pad_h // 2
                pad_b = pad_h - pad_t
                pad = (pad_l, pad_r, pad_t, pad_b)
                x = F.pad(x, pad, mode='constant', value=0.0)
            kernel_size = (x.shape[-2] // size[-2], x.shape[-1] // size[-1])
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            return x

    def forward(self, x1, x2, out_size=None):
        assert x1.shape[:2] == x2.shape[:2]
        assert out_size is None or len(out_size) == 2
        if out_size is None:  # resize to larger one
            out_size = max(x1.size()[2:], x2.size()[2:])

        x1 = self.input1_conv(x1)
        x2 = self.input2_conv(x2)

        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)

        x = self._binary_op(x1, x2)
        if self.with_out_conv:
            x = self.out_conv(x)
        return x


class SumCell(BaseMergeCell):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def _binary_op(self, x1, x2):
        return x1 + x2


class ConcatCell(BaseMergeCell):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels * 2, out_channels, **kwargs)

    def _binary_op(self, x1, x2):
        ret = torch.cat([x1, x2], dim=1)
        return ret


class GlobalPoolingCell(BaseMergeCell):

    def __init__(self, in_channels=None, out_channels=None, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class NASFPNDecoder(nn.Module):
    """NAS-FPN.

    Implementation of `NAS-FPN: Learning Scalable Feature Pyramid Architecture
    for Object Detection <https://arxiv.org/abs/1904.07392>`_

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        depth (int): Number of output scales.
        stack_times (int): The number of times the pyramid architecture will
            be stacked.
    """

    def __init__(self,
                 in_channels,
                 pyramid_channels=256,
                 segmentation_channels=128,
                 depth=5,
                 stack_times=3,
                 merge_policy="add",
                 deep_supervision=False):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))
        self.in_channels = in_channels
        self.pyramid_channels = pyramid_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.depth = depth  # num of output feature levels
        assert self.num_ins == self.depth
        self.stack_times = stack_times

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 5
        self.deep_supervision = deep_supervision

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(depth):
            l_conv = ConvModule(
                in_channels[i],
                pyramid_channels,
                1,
                activation=None)
            self.lateral_convs.append(l_conv)

        # add NAS FPN connections
        self.fpn_stages = nn.ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GlobalPoolingCell(
                in_channels=pyramid_channels,
                out_channels=pyramid_channels)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(
                in_channels=pyramid_channels,
                out_channels=pyramid_channels)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(
                in_channels=pyramid_channels,
                out_channels=pyramid_channels)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(
                in_channels=pyramid_channels,
                out_channels=pyramid_channels)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_55_5'] = SumCell(
                in_channels=pyramid_channels,
                out_channels=pyramid_channels)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_77_7'] = SumCell(
                in_channels=pyramid_channels,
                out_channels=pyramid_channels)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GlobalPoolingCell(
                in_channels=pyramid_channels,
                out_channels=pyramid_channels)
            self.fpn_stages.append(stage)

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [4, 3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)

    def forward(self, *features):
        """Forward function."""
        # build P1-P5
        features = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # This is actually P1-P5 but too lazy to change the naming scheme
        p3, p4, p5, p6, p7 = features[-5:]

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p7, p6, p5, p4, p3])]
        x = self.merge(feature_pyramid)

        if self.deep_supervision and self.training:
            return p4, p3, x

        return x