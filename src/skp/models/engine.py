import math
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchvideo.models.x3d import create_x3d_stem
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer_v2 import SwinTransformerV2
from . import backbones
from . import segmentation
from .pooling import create_pool2d_layer, create_pool3d_layer
from .sequence import Transformer, DualTransformer, DualTransformerV2
from .tools import change_initial_stride, change_num_input_channels


class Net2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 pool,
                 in_channels=3,
                 change_stride=None,
                 feature_reduction=None,
                 multisample_dropout=False,
                 load_pretrained_backbone=None,
                 freeze_backbone=False,
                 backbone_params={},
                 pool_layer_params={}):

        super().__init__()
        self.backbone, dim_feats = backbones.create_backbone(name=backbone, pretrained=pretrained, **backbone_params)
        if isinstance(pool, str):
            self.pool_layer = create_pool2d_layer(name=pool, **pool_layer_params)
        else:
            self.pool_layer = nn.Identity()
        if pool == "catavgmax": 
            dim_feats *= 2
        self.msdo = multisample_dropout
        if in_channels != 3:
            self.backbone = change_num_input_channels(self.backbone, in_channels)
        if change_stride:
            self.backbone = change_initial_stride(self.backbone, tuple(change_stride), in_channels)
        self.dropout = nn.Dropout(p=dropout)
        if isinstance(feature_reduction, int):
            # Use 1D grouped convolution to reduce # of parameters
            groups = math.gcd(dim_feats, feature_reduction)
            self.feature_reduction = nn.Conv1d(dim_feats, feature_reduction, groups=groups, kernel_size=1,
                                               stride=1, bias=False)
            dim_feats = feature_reduction
        self.classifier = nn.Linear(dim_feats, num_classes) 

        if load_pretrained_backbone:
            # Assumes that model has a `backbone` attribute
            # Note: if you want to load the entire pretrained model, this is done via the
            # builder.build_model function
            print(f"Loading pretrained backbone from {load_pretrained_backbone} ...")
            weights = torch.load(load_pretrained_backbone, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
            # Get feature_reduction, if present
            feat_reduce_weight = {re.sub(r"^feature_reduction.", "", k): v
                                  for k, v in weights.items() if "feature_reduction" in k}
            # Get backbone only
            weights = {re.sub(r'^backbone.', '', k) : v for k,v in weights.items() if 'backbone' in k}
            self.backbone.load_state_dict(weights)
            if len(feat_reduce_weight) > 0:
                print("Also loading feature reduction layer ...")
                self.feature_reduction.load_state_dict(feat_reduce_weight)

        if freeze_backbone:
            print("Freezing backbone ...")
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_features(self, x):
        features = self.backbone(x)
        features = self.pool_layer(features)
        if isinstance(self.backbone, VisionTransformer):
            features = features[:, self.backbone.num_prefix_tokens:].mean(dim=1)
        if isinstance(self.backbone, SwinTransformerV2):
            features = features.mean(dim=1)
        if hasattr(self, "feature_reduction"):
            features = self.feature_reduction(features.unsqueeze(-1)).squeeze(-1)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        if self.msdo:
            x = torch.mean(torch.stack([self.classifier(self.dropout(features)) for _ in range(5)]), dim=0)
        else:
            x = self.classifier(self.dropout(features))
        # Important nuance:
        # For binary classification, the model returns a tensor of shape (N,)
        # Otherwise, (N,C)
        return x[:, 0] if self.classifier.out_features == 1 else x


class SeqNet2D(Net2D):

    def forward(self, x):
        # x.shape = (N, C, Z, H, W)
        features = torch.stack([self.extract_features(x[:, :, _]) for _ in range(x.size(2))], dim=2)
        features = features.max(2)[0]

        if self.msdo:
            x = torch.mean(torch.stack([self.classifier(self.dropout(features)) for _ in range(5)]), dim=0)
        else:
            x = self.classifier(self.dropout(features))
        # Important nuance:
        # For binary classification, the model returns a tensor of shape (N,)
        # Otherwise, (N,C)
        return x[:, 0] if self.classifier.out_features == 1 else x


class TDCNN(nn.Module):

    def __init__(self, cnn_params, transformer_params, freeze_cnn=False, freeze_transformer=False):
        super().__init__()
        self.cnn = Net2D(**cnn_params)
        del self.cnn.dropout
        del self.cnn.classifier
        self.transformer = Transformer(**transformer_params)

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def extract_features(self, x):
        N, C, Z, H, W = x.size()
        assert N == 1, "For feature extraction, batch size must be 1"
        features = self.cnn.extract_features(x.squeeze(0).transpose(0, 1)).unsqueeze(0)
        # features.shape = (1, Z, dim_feats)
        return self.transformer.extract_features((features, torch.ones((features.size(0), features.size(1))).to(features.device)))

    def forward(self, x):
        # BCZHW
        features = torch.stack([self.cnn.extract_features(x[:, :, i]) for i in range(x.size(2))], dim=1)
        # B, seq_len, dim_feat
        return self.transformer((features, torch.ones((features.size(0), features.size(1))).to(features.device)))


class Net2DWith3DStem(Net2D):

    def __init__(self, *args, **kwargs):
        stem_out_channels = kwargs.pop("stem_out_channels", 24)
        load_pretrained_stem = kwargs.pop("load_pretrained_stem", None)
        conv_kernel_size = tuple(kwargs.pop("conv_kernel_size", (5, 3, 3)))
        conv_stride = tuple(kwargs.pop("conv_stride", (1, 2, 2)))
        in_channels = kwargs.pop("in_channels", 3)
        kwargs["in_channels"] = stem_out_channels
        super().__init__(*args, **kwargs)
        self.stem_layer = create_x3d_stem(in_channels=in_channels,
                                          out_channels=stem_out_channels,
                                          conv_kernel_size=conv_kernel_size,
                                          conv_stride=conv_stride)
        if kwargs["pretrained"]:
            from pytorchvideo.models.hub import x3d_l
            self.stem_layer.load_state_dict(x3d_l(pretrained=True).blocks[0].state_dict())

        if load_pretrained_stem:
            import re
            print(f"  Loading pretrained stem from {load_pretrained_stem} ...")
            weights = torch.load(load_pretrained_stem, map_location=lambda storage, loc: storage)['state_dict']
            stem_weights = {k.replace("model.backbone.blocks.0.", ""): v for k, v in weights.items() if "backbone.blocks.0" in k}
            self.stem_layer.load_state_dict(stem_weights)

    def forward(self, x):
        x = self.stem_layer(x)
        x = x.mean(3)
        features = self.extract_features(x)
        if self.msdo:
            x = torch.mean(torch.stack([self.classifier(self.dropout(features)) for _ in range(5)]), dim=0)
        else:
            x = self.classifier(self.dropout(features))
        # Important nuance:
        # For binary classification, the model returns a tensor of shape (N,)
        # Otherwise, (N,C)
        return x[:, 0] if self.classifier.out_features == 1 else x


class Net3D(Net2D):

    def __init__(self, *args, **kwargs):
        z_strides = kwargs.pop("z_strides", [1,1,1,1,1])
        super().__init__(*args, **kwargs)
        self.pool_layer = create_pool3d_layer(name=kwargs["pool"], **kwargs.pop("pool_layer_params", {}))


class NetSegment2D(nn.Module):
    """ For now, this class essentially servers as a wrapper for the 
    segmentation model which is mostly defined in the segmentation submodule, 
    similar to the original segmentation_models.pytorch.

    It may be worth refactoring it in the future, such that you define this as
    a general class, then select your choice of encoder and decoder. The encoder
    is pretty much the same across all the segmentation models currently 
    implemented (DeepLabV3+, FPN, Unet).
    """
    def __init__(self,
                 architecture,
                 encoder_name,
                 encoder_params,
                 decoder_params,
                 num_classes,
                 dropout,
                 in_channels,
                 load_pretrained_encoder=None,
                 freeze_encoder=False,
                 deep_supervision=False,
                 pool_layer_params={},
                 aux_head_params={}):

        super().__init__()

        self.segmentation_model = getattr(segmentation, architecture)(
                encoder_name=encoder_name,
                encoder_params=encoder_params,
                dropout=dropout,
                classes=num_classes,
                deep_supervision=deep_supervision,
                in_channels=in_channels,
                **decoder_params
            )


        if load_pretrained_encoder: 
            # Assumes that model has a `encoder` attribute
            # Note: if you want to load the entire pretrained model, this is done via the
            # builder.build_model function
            print(f"Loading pretrained encoder from {load_pretrained_encoder} ...")
            weights = torch.load(load_pretrained_encoder, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.segmentation_model', '', k) : v for k,v in weights.items()}
            # Get encoder only
            weights = {re.sub(r'^encoder.', '', k) : v for k,v in weights.items() if 'backbone' in k}
            self.segmentation_model.encoder.load_state_dict(weights)

        if freeze_encoder:
            print("Freezing encoder ...")
            for param in self.segmentation_model.encoder.parameters():
                param.requires_grad = False


    def forward(self, x):
        return self.segmentation_model(x)


class NetSegment3D(NetSegment2D): 

    pass
