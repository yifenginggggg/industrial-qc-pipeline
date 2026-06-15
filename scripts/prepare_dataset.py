#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_qc.config import ProjectPaths
from industrial_qc.data.convert import build_split_mapping, discover_dataset_records, prepare_processed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KolektorSDD2 for YOLO detection and U-Net segmentation.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Raw dataset root. Defaults to data/raw/KolektorSDD2.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Processed dataset root. Defaults to data/processed.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio drawn from the training split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split creation.")
    parser.add_argument("--min-area", type=int, default=4, help="Minimum defect-mask area to keep a YOLO box.")
    parser.add_argument("--smoke-run", action="store_true", help="Generate a tiny synthetic dataset before preparation.")
    return parser.parse_args()


def generate_smoke_dataset(raw_dir: Path) -> None:
    for split_name, sample_count in {"train": 6, "test": 2}.items():
        image_dir = raw_dir / split_name / "images"
        mask_dir = raw_dir / split_name / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for index in range(sample_count):
            image = np.full((256, 256, 3), 210, dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            has_defect = index % 2 == 0
            if has_defect:
                row = 40 + index * 20
                image[row : row + 6, 60:190] = np.array([180, 30, 30], dtype=np.uint8)
                mask[row : row + 6, 60:190] = 255
            Image.fromarray(image).save(image_dir / f"sample_{index:03d}.png")
            Image.fromarray(mask).save(mask_dir / f"sample_{index:03d}_mask.png")


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    paths.ensure()

    raw_dir = (args.raw_dir or paths.raw_data_dir / "KolektorSDD2").resolve()
    output_dir = (args.output_dir or paths.processed_data_dir).resolve()
    if args.smoke_run:
        raw_dir = paths.raw_data_dir / "smoke_ksdd2"
        generate_smoke_dataset(raw_dir)

    records = discover_dataset_records(raw_dir)
    split_mapping = build_split_mapping(records, val_ratio=args.val_ratio, seed=args.seed)
    summary = prepare_processed_dataset(split_mapping, output_root=output_dir, min_area=args.min_area)

    print(
        json.dumps(
            {
                "raw_dir": str(raw_dir),
                "output_dir": str(output_dir),
                "splits": {name: len(items) for name, items in split_mapping.items()},
                "dataset_yaml": str(summary["dataset_yaml"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
