from __future__ import annotations

import base64
import io
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_HINTS = ("mask", "label", "labels", "annotation", "annotations", "ann", "gt", "seg")


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path
    split: str
    stem: str
    is_defect: bool


def mask_to_bboxes(mask: np.ndarray, min_area: int = 1) -> list[tuple[int, int, int, int]]:
    binary = mask > 0
    if not np.any(binary):
        return []

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        area = 0

        while stack:
            y, x = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                    continue
                if visited[next_y, next_x] or not binary[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))

        if area >= min_area:
            boxes.append((min_x, min_y, max_x, max_y))

    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def bbox_to_yolo(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1
    center_x = min_x + box_width / 2.0
    center_y = min_y + box_height / 2.0
    return (
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    )


def mask_to_yolo_lines(mask: np.ndarray, min_area: int = 1, class_id: int = 0) -> list[str]:
    height, width = mask.shape[:2]
    lines = []
    for box in mask_to_bboxes(mask, min_area=min_area):
        x_center, y_center, box_width, box_height = bbox_to_yolo(box, width, height)
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return lines


def discover_dataset_records(raw_dir: Path) -> list[SampleRecord]:
    raw_dir = raw_dir.expanduser().resolve()
    if _is_supervisely_project(raw_dir):
        return _discover_supervisely_records(raw_dir)

    candidates = [path for path in raw_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    mask_map: dict[str, list[Path]] = {}
    image_paths: list[Path] = []

    for path in candidates:
        relative_path = path.relative_to(raw_dir)
        if _looks_like_mask(relative_path):
            mask_map.setdefault(_canonical_key(path), []).append(path)
        else:
            image_paths.append(path)

    records: list[SampleRecord] = []
    for image_path in image_paths:
        key = _canonical_key(image_path)
        mask_path = _select_mask_candidate(image_path, mask_map.get(key, []))
        if mask_path is None:
            continue
        mask = load_mask(mask_path)
        records.append(
            SampleRecord(
                image_path=image_path,
                mask_path=mask_path,
                split=_infer_split(image_path),
                stem=_stable_stem(image_path.relative_to(raw_dir)),
                is_defect=bool(mask.any()),
            )
        )

    if not records:
        raise FileNotFoundError(
            f"No image/mask pairs were discovered under {raw_dir}. "
            "Check the dataset layout or adjust the discovery logic."
        )

    records.sort(key=lambda record: (record.split, record.stem))
    return records


def load_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix.lower() == ".json":
        return _load_supervisely_mask(mask_path)
    return normalize_mask(np.asarray(Image.open(mask_path)))


def build_split_mapping(
    records: list[SampleRecord], val_ratio: float = 0.2, seed: int = 42
) -> dict[str, list[SampleRecord]]:
    rng = random.Random(seed)
    grouped: dict[str, list[SampleRecord]] = {"train": [], "val": [], "test": []}
    train_pool = [record for record in records if record.split not in {"test", "val"}]
    grouped["test"] = [record for record in records if record.split == "test"]
    grouped["val"] = [record for record in records if record.split == "val"]

    positives = [record for record in train_pool if record.is_defect]
    negatives = [record for record in train_pool if not record.is_defect]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positive_val = _take_validation_slice(positives, val_ratio)
    negative_val = _take_validation_slice(negatives, val_ratio)
    val_ids = {record.stem for record in positive_val + negative_val}

    grouped["val"].extend(positive_val)
    grouped["val"].extend(negative_val)
    grouped["train"] = [record for record in train_pool if record.stem not in val_ids]

    for split_name in grouped:
        grouped[split_name].sort(key=lambda record: record.stem)
    return grouped


def prepare_processed_dataset(
    records_by_split: dict[str, list[SampleRecord]],
    output_root: Path,
    min_area: int = 1,
) -> dict[str, Path]:
    output_root = output_root.expanduser().resolve()
    yolo_root = output_root / "yolo_ksdd2"
    unet_root = output_root / "unet_ksdd2"
    summary: dict[str, Path] = {
        "yolo_root": yolo_root,
        "unet_root": unet_root,
        "dataset_yaml": yolo_root / "dataset.yaml",
        "train_manifest": unet_root / "train.txt",
        "val_manifest": unet_root / "val.txt",
        "test_manifest": unet_root / "test.txt",
    }

    for split_name, records in records_by_split.items():
        image_dir = yolo_root / "images" / split_name
        label_dir = yolo_root / "labels" / split_name
        split_image_dir = unet_root / "images" / split_name
        split_mask_dir = unet_root / "masks" / split_name
        for directory in (image_dir, label_dir, split_image_dir, split_mask_dir):
            directory.mkdir(parents=True, exist_ok=True)

        manifest_lines: list[str] = []
        for record in records:
            image_output = image_dir / f"{record.stem}{record.image_path.suffix.lower()}"
            yolo_label_output = label_dir / f"{record.stem}.txt"
            unet_image_output = split_image_dir / f"{record.stem}{record.image_path.suffix.lower()}"
            unet_mask_output = split_mask_dir / f"{record.stem}.png"

            shutil.copy2(record.image_path, image_output)
            shutil.copy2(record.image_path, unet_image_output)
            mask = load_mask(record.mask_path)
            Image.fromarray(mask).save(unet_mask_output)

            yolo_label_output.write_text("\n".join(mask_to_yolo_lines(mask, min_area=min_area)), encoding="utf-8")
            manifest_lines.append(f"{unet_image_output}\t{unet_mask_output}")

        (unet_root / f"{split_name}.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    summary["dataset_yaml"].write_text(
        "\n".join(
            [
                f"path: {yolo_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: defect",
            ]
        ),
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "yolo_root": str(yolo_root),
                "unet_root": str(unet_root),
                "splits": {name: len(items) for name, items in records_by_split.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _looks_like_mask(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    return any(hint in stem for hint in MASK_HINTS) or any(part in MASK_HINTS for part in lower_parts[:-1])


def _canonical_key(path: Path) -> str:
    key = path.stem.lower()
    for token in ("_mask", "-mask", "_label", "-label", "_gt", "-gt", "_seg", "-seg"):
        key = key.replace(token, "")
    return "".join(ch for ch in key if ch.isalnum())


def _infer_split(path: Path) -> str:
    lowered_parts = [part.lower() for part in path.parts]
    if "test" in lowered_parts:
        return "test"
    if "val" in lowered_parts or "valid" in lowered_parts or "validation" in lowered_parts:
        return "val"
    return "train"


def _select_mask_candidate(image_path: Path, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    image_split = _infer_split(image_path)
    candidates = sorted(
        candidates,
        key=lambda path: (_infer_split(path) != image_split, len(path.parts), path.as_posix()),
    )
    return candidates[0]


def _stable_stem(relative_path: Path) -> str:
    return "_".join(relative_path.with_suffix("").parts)


def _take_validation_slice(records: list[SampleRecord], val_ratio: float) -> list[SampleRecord]:
    if len(records) <= 1:
        return []
    val_count = max(1, int(round(len(records) * val_ratio)))
    val_count = min(val_count, len(records) - 1)
    return records[:val_count]


def _is_supervisely_project(raw_dir: Path) -> bool:
    return (raw_dir / "meta.json").exists() and any((raw_dir / split / "ann").exists() for split in ("train", "test", "val"))


def _discover_supervisely_records(raw_dir: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split_name in ("train", "val", "test"):
        ann_dir = raw_dir / split_name / "ann"
        img_dir = raw_dir / split_name / "img"
        if not ann_dir.exists() or not img_dir.exists():
            continue

        for ann_path in sorted(ann_dir.glob("*.json")):
            image_path = img_dir / ann_path.name.replace(".json", "")
            if not image_path.exists():
                continue
            mask = load_mask(ann_path)
            records.append(
                SampleRecord(
                    image_path=image_path,
                    mask_path=ann_path,
                    split=split_name,
                    stem=_stable_stem(image_path.relative_to(raw_dir)),
                    is_defect=bool(mask.any()),
                )
            )

    if not records:
        raise FileNotFoundError(f"No Supervisely image/annotation pairs found under {raw_dir}.")
    return records


def _load_supervisely_mask(annotation_path: Path) -> np.ndarray:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    height = int(annotation["size"]["height"])
    width = int(annotation["size"]["width"])
    canvas = np.zeros((height, width), dtype=np.uint8)

    for obj in annotation.get("objects", []):
        bitmap = obj.get("bitmap")
        if not bitmap:
            continue
        object_mask = _decode_supervisely_bitmap(bitmap["data"])
        origin_x, origin_y = bitmap["origin"]
        object_array = (np.asarray(object_mask) > 0).astype(np.uint8) * 255
        mask_height, mask_width = object_array.shape[:2]
        canvas[origin_y : origin_y + mask_height, origin_x : origin_x + mask_width] = np.maximum(
            canvas[origin_y : origin_y + mask_height, origin_x : origin_x + mask_width],
            object_array,
        )

    return canvas


def _decode_supervisely_bitmap(encoded_bitmap: str) -> Image.Image:
    raw_bytes = zlib.decompress(base64.b64decode(encoded_bitmap))
    return Image.open(io.BytesIO(raw_bytes))
