#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_qc.config import ProjectPaths
from industrial_qc.runtime import resolve_device
from industrial_qc.wandb_utils import maybe_init_wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net defect segmenter.")
    parser.add_argument("--train-manifest", type=Path, default=None, help="Training manifest path.")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Validation manifest path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="unet_smoke")
    parser.add_argument("--wandb-project", default="industrial-qc-unet")
    parser.add_argument("--wandb-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from industrial_qc.data.segmentation_dataset import SegmentationDataset
    from industrial_qc.losses import BCEDiceLoss
    from industrial_qc.metrics import binary_segmentation_metrics
    from industrial_qc.models.unet import UNet

    paths = ProjectPaths.from_root(PROJECT_ROOT)
    paths.ensure()

    train_manifest = args.train_manifest or paths.processed_data_dir / "unet_ksdd2" / "train.txt"
    val_manifest = args.val_manifest or paths.processed_data_dir / "unet_ksdd2" / "val.txt"
    device = resolve_device(args.device)

    train_dataset = SegmentationDataset(train_manifest, image_size=args.image_size)
    val_dataset = SegmentationDataset(val_manifest, image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BCEDiceLoss()
    run_dir = paths.outputs_dir / "unet" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = maybe_init_wandb(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr": args.lr,
        },
        api_key=args.wandb_key,
    )

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        aggregate = {"precision": 0.0, "recall": 0.0, "dice": 0.0, "iou": 0.0}
        batches = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                val_loss += float(criterion(logits, masks).item())
                metrics = binary_segmentation_metrics(logits, masks)
                for key, value in metrics.items():
                    aggregate[key] += value
                batches += 1

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader), 1),
            "val_loss": val_loss / max(len(val_loader), 1),
        }
        if batches > 0:
            metrics.update({key: value / batches for key, value in aggregate.items()})

        if wandb_run is not None:
            wandb_run.log(metrics)

        if metrics.get("dice", -1.0) > best_dice:
            best_dice = metrics["dice"]
            torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, run_dir / "best.pt")

        print(json.dumps(metrics))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
