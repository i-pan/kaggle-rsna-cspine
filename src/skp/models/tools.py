import torch.nn as nn


def change_num_input_channels(model, in_channels=1):
    """
    Assumes number of input channels in model is 3.
    """
    for i, m in enumerate(model.modules()):
      if isinstance(m, (nn.Conv2d,nn.Conv3d)) and m.in_channels == 3:
        m.in_channels = in_channels
        # First, sum across channels
        W = m.weight.sum(1, keepdim=True)
        # Then, divide by number of channels
        W = W / in_channels
        # Then, repeat by number of channels
        size = [1] * W.ndim
        size[1] = in_channels
        W = W.repeat(size)
        m.weight = nn.Parameter(W)
        break
    return model


def change_initial_stride(model, stride, in_channels):

    for i, m in enumerate(model.modules()):
      if isinstance(m, (nn.Conv2d, nn.Conv3d)) and m.in_channels == in_channels:
        m.stride = stride
        break
    return model