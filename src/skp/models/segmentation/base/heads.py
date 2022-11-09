import torch.nn as nn
import torch.nn.functional as F

from ...pooling import create_pool2d_layer


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout=0.2, kernel_size=3, upsampling=1):
        dropout = nn.Dropout2d(p=dropout) if dropout else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(dropout, conv2d, upsampling)


class SegmentationHead_3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, kernel_size=3, upsampling=1):
        super().__init__()
        self.dropout = nn.Dropout3d(p=dropout) if dropout else nn.Identity()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = upsampling

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv3d(x)
        x = F.interpolate(x, scale_factor=self.upsampling, mode="trilinear", align_corners=False)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2):
        pool = create_pool2d_layer(pooling)
        dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        super().__init__(pool, dropout, linear)
