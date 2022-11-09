import torch
import torch.nn as nn

from .dice import DiceLoss
from .soft_bce import SoftBCEWithLogitsLoss
from .soft_ce import SoftCrossEntropyLoss


class DiceBCELoss(nn.Module):

    def __init__(self, dice_loss_params, bce_loss_params, 
                 dice_loss_weight, bce_loss_weight):
        super().__init__()
        self.dice_loss = DiceLoss(**dice_loss_params)
        self.bce_loss = SoftBCEWithLogitsLoss(**bce_loss_params)
        self.dice_loss_weight = torch.tensor(dice_loss_weight)
        self.bce_loss_weight = torch.tensor(bce_loss_weight)

    def forward(self, output, target):
        dice_loss = self.dice_loss_weight * self.dice_loss(output, target)
        bce_loss = self.bce_loss_weight * self.bce_loss(output, target)
        return dice_loss + bce_loss


class DiceCELoss(nn.Module):

    def __init__(self, dice_loss_params, ce_loss_params,
                 dice_loss_weight, ce_loss_weight):
        super().__init__()
        self.dice_loss = DiceLoss(**dice_loss_params)
        self.ce_loss = SoftCrossEntropyLoss(**ce_loss_params)
        self.dice_loss_weight = torch.tensor(dice_loss_weight)
        self.ce_loss_weight = torch.tensor(ce_loss_weight)

    def forward(self, output, target):
        dice_loss = self.dice_loss_weight * self.dice_loss(output, target)
        ce_loss = self.ce_loss_weight * self.ce_loss(output, target.long())
        return dice_loss + ce_loss