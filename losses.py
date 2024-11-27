import torch
from torch import nn

# 自定义损失函数：Binary Cross Entropy + Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(BCEDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        # 计算 Binary Cross Entropy Loss
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight)

        # 计算 Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1e-7) / (inputs.sum() + targets.sum() + 1e-7)
        dice_loss = 1 - dice

        return bce + dice_loss

class FocalBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, bce_weight=0.5, dice_weight=0.5, weight=None):
        super(FocalBCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.weight = weight

    def focal_loss(self, inputs, targets):
        # 计算 BCE Loss
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight, reduction='none')
        # 计算 p_t
        p_t = torch.sigmoid(inputs)
        p_t = p_t * targets + (1 - p_t) * (1 - targets)
        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - p_t)**self.gamma * bce
        return focal_loss.mean()

    def forward(self, inputs, targets):
        # 计算 Focal BCE Loss
        focal_bce = self.focal_loss(inputs, targets)

        # 计算 Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1e-7) / (inputs.sum() + targets.sum() + 1e-7)
        dice_loss = 1 - dice

        # 通过权重平衡 Focal BCE 和 Dice Loss
        return self.bce_weight * focal_bce + self.dice_weight * dice_loss


class BCETverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, bce_weight=0.5):
        super(BCETverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()

        tversky = (intersection + 1e-7) / (intersection + self.alpha * false_neg + self.beta * false_pos + 1e-7)
        tversky_loss = 1 - tversky

        return self.bce_weight * bce + (1 - self.bce_weight) * tversky_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 将输入和目标变成一维的向量
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class WeightedDiceLoss(nn.Module):
    def __init__(self, weight_map=None):
        super(WeightedDiceLoss, self).__init__()
        self.weight_map = weight_map

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1e-7) / (inputs.sum() + targets.sum() + 1e-7)
        loss = 1 - dice

        if self.weight_map is not None:
            # 按照加权的 Dice Loss
            weight = self.weight_map.view(-1)
            loss = (loss * weight).mean()

        return loss