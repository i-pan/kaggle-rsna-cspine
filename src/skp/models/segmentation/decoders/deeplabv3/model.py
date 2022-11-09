from torch import nn
from typing import Optional

from ...base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from ...encoders.create import create_encoder
from .decoder import DeepLabV3PlusDecoder


class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    def __init__(
        self,
        encoder_name: str,
        encoder_params: dict = {"pretrained": True, "output_stride": 16},
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        deep_supervision: bool = False, 
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        encoder_output_stride = encoder_params.pop("output_stride", None)
        if encoder_output_stride not in [8, 16, 32]:
            raise ValueError("Encoder output stride should be 8, 16, or 32; got {}".format(encoder_output_stride))

        self.encoder = create_encoder(
            name=encoder_name,
            encoder_params=encoder_params,
            encoder_output_stride=encoder_output_stride,
            in_channels=in_channels
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            deep_supervision=deep_supervision
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            kernel_size=1,
            dropout=dropout,
            upsampling=upsampling,
        )

        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.supervisor_heads = []
            self.supervisor_heads.append(
                SegmentationHead(
                    in_channels=48,
                    out_channels=classes,
                    dropout=dropout,
                    kernel_size=3,
                    upsampling=1,
                )
            )
            self.supervisor_heads.append(
                SegmentationHead(
                    in_channels=decoder_channels,
                    out_channels=classes,
                    dropout=dropout,
                    kernel_size=3,
                    upsampling=1,
                )
            )
            self.supervisor_heads = nn.Sequential(*self.supervisor_heads)

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
