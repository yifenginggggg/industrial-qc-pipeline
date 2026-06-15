from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_qc.data.convert import load_mask, mask_to_bboxes
from industrial_qc.pipeline import compute_roi_boxes


@dataclass(frozen=True)
class ROISample:
    source_image_path: Path
    source_mask_path: Path
    output_stem: str
    crop_box: tuple[int, int, int, int]
    is_defect: bool


def load_segmentation_manifest(manifest_path: str | Path) -> list[tuple[Path, Path]]:
    lines = [line.strip() for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(Path(image_path), Path(mask_path)) for image_path, mask_path in (line.split("\t") for line in lines)]


def build_roi_samples_for_record(
    image_path: Path,
    mask_path: Path,
    sample_stem: str,
    padding_ratio: float = 0.15,
    min_size: int = 32,
    min_area: int = 1,
) -> list[ROISample]:
    image = Image.open(image_path)
    width, height = image.size
    mask = load_mask(mask_path)
    boxes = mask_to_bboxes(mask, min_area=min_area)
    if not boxes:
        return [
            ROISample(
                source_image_path=image_path,
                source_mask_path=mask_path,
                output_stem=f"{sample_stem}_roi0",
                crop_box=(0, 0, width, height),
                is_defect=False,
            )
        ]

    detections = [{"xyxy": [float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)], "confidence": 1.0, "cls": 0} for min_x, min_y, max_x, max_y in boxes]
    roi_boxes = compute_roi_boxes(
        detections,
        image_size=(width, height),
        padding_ratio=padding_ratio,
        min_size=min_size,
    )
    return [
        ROISample(
            source_image_path=image_path,
            source_mask_path=mask_path,
            output_stem=f"{sample_stem}_roi{index}",
            crop_box=box,
            is_defect=True,
        )
        for index, box in enumerate(roi_boxes)
    ]


def write_roi_dataset(
    manifests_by_split: dict[str, list[tuple[Path, Path]]],
    output_root: str | Path,
    padding_ratio: float = 0.15,
    min_size: int = 32,
    min_area: int = 1,
) -> dict[str, Path]:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Path] = {}
    counts: dict[str, int] = {}

    for split_name, pairs in manifests_by_split.items():
        image_dir = output_root / "images" / split_name
        mask_dir = output_root / "masks" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        manifest_lines: list[str] = []
        split_count = 0
        for image_path, mask_path in pairs:
            samples = build_roi_samples_for_record(
                image_path=image_path,
                mask_path=mask_path,
                sample_stem=image_path.stem,
                padding_ratio=padding_ratio,
                min_size=min_size,
                min_area=min_area,
            )
            source_image = Image.open(image_path).convert("RGB")
            source_mask = Image.open(mask_path).convert("L")
            for sample in samples:
                min_x, min_y, max_x, max_y = sample.crop_box
                cropped_image = source_image.crop((min_x, min_y, max_x, max_y))
                cropped_mask = source_mask.crop((min_x, min_y, max_x, max_y))
                image_output = image_dir / f"{sample.output_stem}.png"
                mask_output = mask_dir / f"{sample.output_stem}.png"
                cropped_image.save(image_output)
                cropped_mask.save(mask_output)
                manifest_lines.append(f"{image_output}\t{mask_output}")
                split_count += 1

        manifest_path = output_root / f"{split_name}.txt"
        manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
        summary[f"{split_name}_manifest"] = manifest_path
        counts[split_name] = split_count

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "output_root": str(output_root),
                "padding_ratio": padding_ratio,
                "min_size": min_size,
                "min_area": min_area,
                "counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
