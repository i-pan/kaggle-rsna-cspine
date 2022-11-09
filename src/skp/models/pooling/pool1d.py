import torch
import torch.nn as nn
import torch.nn.functional as F

from .gem import GeM


def adaptive_avgmax_pool1d(x, output_size=1):
    x_avg = F.adaptive_avg_pool1d(x, output_size)
    x_max = F.adaptive_max_pool1d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool1d(x, output_size=1):
    x_avg = F.adaptive_avg_pool1d(x, output_size)
    x_max = F.adaptive_max_pool1d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool1d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool1d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool1d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool1d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool1d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class FastAdaptiveAvgPool1d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool1d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean(2, keepdim=not self.flatten)


class AdaptiveAvgMaxPool1d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool1d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool1d(x, self.output_size)


class AdaptiveCatAvgMaxPool1d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool1d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool1d(x, self.output_size)


class SelectAdaptivePool1d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool1d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool1d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool1d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool1d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'


def create_pool1d_layer(name, **kwargs):
    assert name in ["avg", "max", "fast", "avgmax", "catavgmax", "gem"]
    if name != "gem":
        pool1d_layer = SelectAdaptivePool1d(pool_type=name, flatten=True)
    elif name == "gem": 
        pool1d_layer = GeM(dim=1, **kwargs)
    return pool1d_layer