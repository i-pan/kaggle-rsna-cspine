"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
from torch import nn
from torch.nn import functional as F


def GroupNorm(num_channels):

    return nn.GroupNorm(num_groups=16, num_channels=num_channels)


class DeepLabV3PlusDecoder_3D(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        deep_supervision=False, 
        norm_layer="batch_norm",
    ):
        super().__init__()
        assert output_stride in [8, 16, 32]

        self.out_channels = out_channels
        self.output_stride = output_stride
        self.deep_supervision = deep_supervision

        if norm_layer == "batch_norm":
            norm_layer = nn.BatchNorm3d
        elif norm_layer == "group_norm":
            norm_layer = GroupNorm

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, norm_layer=norm_layer, separable=True),
            SeparableConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
        )

        if output_stride == 32:
            self.scale_factor = 8
        elif output_stride == 16:
            self.scale_factor = 4
        else:
            self.scale_factor = 2

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv3d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            norm_layer(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv3d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = F.interpolate(aspp_features, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        
        if self.deep_supervision and self.training:
            return aspp_features, high_res_features, fused_features 

        return fused_features


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_layer):
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_layer):
        super().__init__(
            SeparableConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1, norm_layer=norm_layer))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2, norm_layer=norm_layer))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3, norm_layer=norm_layer))
        modules.append(ASPPPooling(in_channels, out_channels, norm_layer=norm_layer))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv3d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)
