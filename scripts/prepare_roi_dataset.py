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
from industrial_qc.data.roi import load_segmentation_manifest, write_roi_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ROI-based segmentation manifests for U-Net training.")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--roi-padding", type=float, default=0.15)
    parser.add_argument("--roi-min-size", type=int, default=32)
    parser.add_argument("--min-area", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    paths.ensure()

    source_root = paths.processed_data_dir / "unet_ksdd2"
    output_dir = args.output_dir or paths.processed_data_dir / "unet_ksdd2_roi"

    manifests_by_split = {
        "train": load_segmentation_manifest(args.train_manifest or source_root / "train.txt"),
        "val": load_segmentation_manifest(args.val_manifest or source_root / "val.txt"),
        "test": load_segmentation_manifest(args.test_manifest or source_root / "test.txt"),
    }
    summary = write_roi_dataset(
        manifests_by_split,
        output_root=output_dir,
        padding_ratio=args.roi_padding,
        min_size=args.roi_min_size,
        min_area=args.min_area,
    )

    payload = {key: str(value) for key, value in summary.items()}
    payload["output_dir"] = str(output_dir.resolve())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
