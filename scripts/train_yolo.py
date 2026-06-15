#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_qc.config import ProjectPaths
from industrial_qc.wandb_utils import login_wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 for defect detection.")
    parser.add_argument("--data", type=Path, default=None, help="YOLO dataset.yaml path.")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO starting weights.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-name", default="yolo_smoke")
    parser.add_argument("--wandb-project", default="industrial-qc-yolo")
    parser.add_argument("--wandb-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO, settings

    paths = ProjectPaths.from_root(PROJECT_ROOT)
    paths.ensure()
    data_path = args.data or paths.processed_data_dir / "yolo_ksdd2" / "dataset.yaml"
    run_root = paths.outputs_dir / "yolo" / "train"
    run_root.mkdir(parents=True, exist_ok=True)

    if login_wandb(args.wandb_key):
        settings.update({"wandb": True})
    else:
        settings.update({"wandb": False})

    model = YOLO(args.weights)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(run_root),
        name=args.run_name,
        exist_ok=True,
    )
    print(run_root / args.run_name)


if __name__ == "__main__":
    main()
