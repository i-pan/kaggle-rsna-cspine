import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
from typing import Dict, List, Tuple

from . import lovasz_losses as L
from . import segmentation


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        return F.binary_cross_entropy_with_logits(p.float(), t.float(), pos_weight=self.pos_weight)


class MultilabelWeightedBCE(nn.BCEWithLogitsLoss):

    def __init__(self, weights, pos_weight=None):
        super().__init__(pos_weight=pos_weight)
        self.weights = torch.Tensor(weights)
        self.weights = self.weights / self.weights.sum()

    def forward(self, p, t):
        return torch.sum(self.weights.to(p.device) *
                         F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none',
                                                            pos_weight=self.pos_weight).mean(0))


class DualFeatureSequenceBCELoss(nn.Module):

    def __init__(self, weights=[0.5, 0.5], simple_bce_params={}, weighted_bce_params={}):
        super().__init__()
        if "pos_weight" in simple_bce_params:
            simple_bce_params = dict(simple_bce_params)
            simple_bce_params["pos_weight"] = torch.Tensor([simple_bce_params["pos_weight"]])
        if "pos_weight" in weighted_bce_params:
            weighted_bce_params = dict(weighted_bce_params)
            weighted_bce_params["pos_weight"] = torch.Tensor([weighted_bce_params["pos_weight"]])
        self.loss1 = nn.BCEWithLogitsLoss(**simple_bce_params)
        self.loss2 = MultilabelWeightedBCE(**weighted_bce_params)
        self.weights = torch.Tensor(weights)

    def forward(self, p, t):
        p1, p2 = p
        t1, t2 = t
        return self.weights[0] * self.loss1(p1, t1.float()) + self.weights[1] * self.loss2(p2, t2.float())


class CompetitionMetric(nn.Module):

    def forward(self, p, t):
        # p.shape = t.shape = (batch_size, 8) for 8 classes
        num_values = p.shape[0] * p.shape[1]
        weights = np.asarray([1] * num_values) / 14.
        weights[7::8] = 7 / 14.
        p = p.view(num_values, -1).squeeze(1)
        t = t.view(num_values, -1).squeeze(1)
        # print(p.shape, t.shape, weights.shape)
        # import time ; time.sleep(100)
        weights[(t == 1).cpu()] *= 2.0
        weights = torch.Tensor(weights).float().to(p.device)
        return (weights * F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")).sum()


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, p, t):
        t = t.view(-1)
        if self.weight:
            return F.cross_entropy(p.float(), t.long(), weight=self.weight.float().to(t.device))
        else:
            return F.cross_entropy(p.float(), t.long())


class OneHotCrossEntropy(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class SmoothCrossEntropy(nn.Module):
    
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return F.cross_entropy(x, target.long())


class MixBCE(nn.Module):

    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = F.binary_cross_entropy_with_logits(p.float(), t['y1'].float(), reduction='none')
        loss2 = F.binary_cross_entropy_with_logits(p.float(), t['y2'].float(), reduction='none')
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return F.binary_cross_entropy_with_logits(p.float(), t.float())


class MixCrossEntropy(nn.Module):

    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = F.cross_entropy(p.float(), t['y1'].long(), reduction='none')
        loss2 = F.cross_entropy(p.float(), t['y2'].long(), reduction='none')
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return F.cross_entropy(p.float(), t.long())


class DenseCrossEntropy(nn.Module):

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.Module):

    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        labels = F.one_hot(labels.long(), logits.size(1)).float().to(labels.device)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss


class WeightedBCE(nn.Module):
    # From Heng
    def __init__(self, pos_frac: float = 0.25, neg_frac: float = 0.75, eps: float = 1e-7):
        super(WeightedBCE, self).__init__()
        assert 0 < pos_frac < 1, f'`pos_frac` must be between 0 and 1, {pos_frac} is invalid'
        self.pos_frac = pos_frac
        self.neg_frac = neg_frac
        self.eps = eps

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        N,C,H,W = p.shape
        p = p.transpose(0,1).reshape(C, -1)
        t = t.transpose(0,1).reshape(C, -1)
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none')
        pos = (t>0.5).float()
        neg = (t<0.5).float()
        pos_weight = pos.sum(1) + self.eps
        neg_weight = neg.sum(1) + self.eps
        loss = self.pos_frac*pos*loss/pos_weight.unsqueeze(1) + self.neg_frac*neg*loss/neg_weight.unsqueeze(1)
        return loss.sum()


class MixLoss(nn.Module):

    def __init__(self,
                 loss_name,
                 loss_params):
        super().__init__()
        self.loss = eval(loss_name)(**loss_params)

    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = self.loss(p, t['y1'])
        loss2 = self.loss(p, t['y2'])
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return self.loss(p, t)


class SupervisorLoss(nn.Module):

    def __init__(self,
                 segmentation_loss, 
                 loss_params, 
                 num_losses=3, 
                 scale_factors=[0.5, 0.25], 
                 loss_weights=[1.0, 0.4, 0.2]):
        super().__init__()
        self.segmentation_loss = getattr(segmentation, segmentation_loss)(**loss_params)
        assert num_losses == (len(scale_factors) + 1) == len(loss_weights)
        self.num_losses = num_losses
        self.scale_factors = [1] + scale_factors
        self.loss_weights = torch.Tensor(loss_weights)

    def forward(self, p, t):
        if isinstance(p, tuple) and len(p) == self.num_losses:
            # Training with supervision
            loss = 0.
            for i in range(self.num_losses):
                if self.scale_factors[i] == 1:
                    loss += self.loss_weights[i] * self.segmentation_loss(p[i], t)
                else:
                    if len(t.shape) == len(p[0].shape) - 1:
                        loss += self.loss_weights[i] * self.segmentation_loss(p[i], F.interpolate(t.unsqueeze(1), scale_factor=self.scale_factors[i], mode="nearest").squeeze(1))
                    else:
                        loss += self.loss_weights[i] * self.segmentation_loss(p[i], F.interpolate(t, scale_factor=self.scale_factors[i], mode="nearest"))
            return loss
        else:
            return self.segmentation_loss(p, t)




