from __future__ import annotations

import torch
from torch import Tensor


def binary_segmentation_metrics(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    targets = targets.float()

    intersection = (predictions * targets).sum().item()
    pred_area = predictions.sum().item()
    target_area = targets.sum().item()
    union = pred_area + target_area - intersection

    precision = intersection / max(pred_area, 1.0)
    recall = intersection / max(target_area, 1.0)
    dice = (2.0 * intersection) / max(pred_area + target_area, 1.0)
    iou = intersection / max(union, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }
