import torch
import torch.nn as nn

from torch import Tensor
from torchvision.models.video.resnet import BasicBlock, Bottleneck, Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D
from typing import Callable, List, Sequence, Type, Union


# TODO: upload models and load them
model_urls = {
    "r2plus1d_34_8_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth",  # noqa: E501
    "r2plus1d_34_32_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth",  # noqa: E501
    "r2plus1d_34_8_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth",  # noqa: E501
    "r2plus1d_34_32_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth",  # noqa: E501
    "r2plus1d_152_ig65m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/r2plus1d_152_ig65m_from_scratch_f106380637.pth",
    "r2plus1d_152_ig_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/r2plus1d_152_ft_kinetics_from_ig65m_f107107466.pth",
    "r2plus1d_152_sports1m_32frms": "",
    "r2plus1d_152_sports1m_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/r2plus1d_152_ft_kinetics_from_sports1m_f128957437.pth",
    "ir_csn_152_ig65m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ig65m_from_scratch_f125286141.pth",
    "ir_csn_152_ig_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ft_kinetics_from_ig65m_f126851907.pth",
    "ir_csn_152_sports1m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_Sports1M_from_scratch_f99918785.pth",
    "ir_csn_152_sports1m_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ft_kinetics_from_Sports1M_f101599884.pth",
    "ip_csn_152_ig65m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_ig65m_from_scratch_f130601052.pth",
    "ip_csn_152_ig_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_ft_kinetics_from_ig65m_f133090949.pth",
    "ip_csn_152_sports1m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_Sports1M_from_scratch_f111018543.pth",
    "ip_csn_152_sports1m_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_ft_kinetics_from_Sports1M_f111279053.pth",
}


class VideoResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
        layers: List[int],
        stem: Callable[..., nn.Module],
        num_classes: int = 400,
        zero_init_residual: bool = False,
    ) -> None:
        """Generic resnet video generator.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super().__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_builder: Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


def _generic_resnet(arch, pretrained=False, progress=False, **kwargs):
    model = VideoResNet(**kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)

    return model


class BasicStem_Pool(nn.Sequential):
    def __init__(self):
        super(BasicStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class R2Plus1dStem_Pool(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class Conv3DDepthwise(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):

        assert in_planes == out_planes
        super(Conv3DDepthwise, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            groups=in_planes,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class IPConv3DDepthwise(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        assert in_planes == out_planes
        super(IPConv3DDepthwise, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_planes),
            # nn.ReLU(inplace=True),
            Conv3DDepthwise(out_planes, out_planes, None, stride),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
            in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
