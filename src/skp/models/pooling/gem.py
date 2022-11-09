import torch
import torch.nn as nn
import torch.nn.functional as F


# From: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
def gem_1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1./p)


def gem_2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def gem_3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


_GEM_FN = {
    1: gem_1d, 2: gem_2d, 3: gem_3d
}


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.dim = dim
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        pooled = _GEM_FN[self.dim](x, p=self.p, eps=self.eps)
        return self.flatten(pooled)