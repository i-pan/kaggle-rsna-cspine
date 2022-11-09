import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import SelectAdaptivePool2d

from .gem import GeM


def create_pool2d_layer(name, **kwargs):
    assert name in ["avg", "max", "fast", "avgmax", "catavgmax", "gem"]
    if name != "gem":
        pool2d_layer = SelectAdaptivePool2d(pool_type=name, flatten=True)
    elif name == "gem": 
        pool2d_layer = GeM(dim=2, **kwargs)
    return pool2d_layer