from __future__ import annotations

import torch
from torch import Tensor, nn


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = torch.sum(probabilities * targets, dim=(1, 2, 3))
    union = torch.sum(probabilities, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.bce(logits, targets) + dice_loss(logits, targets)
