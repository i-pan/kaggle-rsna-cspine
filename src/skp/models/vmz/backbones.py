import torch.nn as nn

from . import csn
from . import r2plus1d


def ir_csn_152(pretrained=True, **kwargs):
    model = csn.ir_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    return model


def ir_csn_101(pretrained=True, **kwargs):
    model = ir_csn_152(pretrained=pretrained, **kwargs)
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:23]
    return model


def ir_csn_50(pretrained=True, **kwargs):
    model = ir_csn_152(pretrained=pretrained, **kwargs)
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:6]
    return model


def ip_csn_152(pretrained=True, **kwargs):
    model = csn.ip_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    return model


def ip_csn_101(pretrained=True, **kwargs):
    model = ip_csn_152(pretrained=pretrained, **kwargs)
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:23]
    return model


def ip_csn_50(pretrained=True, **kwargs):
    model = ip_csn_152(pretrained=pretrained, **kwargs)
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:6]
    return model


def r2plus1d_34(pretrained=True, **kwargs):
    model = r2plus1d.r2plus1d_34(pretraining='32_ig65m' if pretrained else '', num_classes=359)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    return model