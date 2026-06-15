from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from scipy import ndimage

Detection = dict[str, float | int | list[float]]
Box = tuple[int, int, int, int]


def filter_detections(detections: Iterable[Detection], conf_threshold: float) -> list[Detection]:
    return [detection for detection in detections if float(detection.get("confidence", 0.0)) >= conf_threshold]


def draw_boxes(
    image: Image.Image,
    detections: Sequence[Detection],
    color: str = "#00FF66",
    annotate: bool = True,
) -> Image.Image:
    rendered = image.copy()
    drawer = ImageDraw.Draw(rendered)
    outline = ImageColor.getrgb(color)
    for detection in detections:
        min_x, min_y, max_x, max_y = detection["xyxy"]
        drawer.rectangle((min_x, min_y, max_x, max_y), outline=outline, width=2)
        if annotate and "confidence" in detection:
            drawer.text((min_x, max(min_y - 12, 0)), f"{float(detection['confidence']):.2f}", fill=outline)
    return rendered


def overlay_mask_on_image(
    image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (255, 64, 64), alpha: float = 0.4
) -> Image.Image:
    mask = (mask > 0).astype(np.uint8)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_array = np.asarray(overlay).copy()
    overlay_array[mask > 0] = (*color, int(alpha * 255))
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_binary_mask(mask: np.ndarray, output_path: str | Path) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(output_path)


def should_run_segmentation(detections: Sequence[object]) -> bool:
    return len(detections) > 0


def compute_roi_boxes(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    padding_ratio: float = 0.1,
    min_size: int = 32,
) -> list[Box]:
    boxes = [
        _expand_and_clip_box(tuple(int(round(value)) for value in detection["xyxy"]), image_size, padding_ratio, min_size)
        for detection in detections
    ]
    return _merge_boxes(boxes)


def postprocess_mask(
    mask: np.ndarray,
    min_area: int = 25,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
) -> np.ndarray:
    processed = (mask > 0).astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    if closing_iterations > 0:
        processed = ndimage.binary_closing(processed, structure=structure, iterations=closing_iterations)
    if opening_iterations > 0:
        processed = ndimage.binary_opening(processed, structure=structure, iterations=opening_iterations)

    labeled, num_components = ndimage.label(processed)
    if min_area > 1 and num_components > 0:
        areas = np.bincount(labeled.ravel())
        keep_labels = {label for label, area in enumerate(areas) if label != 0 and area >= min_area}
        processed = np.isin(labeled, list(keep_labels))

    return processed.astype(np.uint8)


def fuse_confidence(
    yolo_confidence: float,
    segmentation_confidence: float,
    area_ratio: float,
    yolo_weight: float = 0.6,
) -> float:
    clamped_weight = min(max(yolo_weight, 0.0), 1.0)
    area_hint = min(max(area_ratio, 0.0) * 20.0, 1.0)
    segmentation_score = 0.85 * min(max(segmentation_confidence, 0.0), 1.0) + 0.15 * area_hint
    fused = clamped_weight * min(max(yolo_confidence, 0.0), 1.0) + (1.0 - clamped_weight) * segmentation_score
    return float(min(max(fused, 0.0), 1.0))


def summarize_defects(
    mask: np.ndarray,
    probability_map: np.ndarray,
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    review_threshold: float = 0.45,
    yolo_weight: float = 0.6,
) -> dict[str, object]:
    binary_mask = (mask > 0).astype(np.uint8)
    labeled, num_components = ndimage.label(binary_mask)
    height, width = binary_mask.shape
    total_pixels = max(height * width, 1)
    defects: list[dict[str, object]] = []
    review_reasons: set[str] = set()

    for label_id in range(1, num_components + 1):
        component = labeled == label_id
        ys, xs = np.nonzero(component)
        if len(xs) == 0:
            continue

        min_x = int(xs.min())
        max_x = int(xs.max()) + 1
        min_y = int(ys.min())
        max_y = int(ys.max()) + 1
        bbox = (min_x, min_y, max_x, max_y)
        area_pixels = int(component.sum())
        area_ratio = area_pixels / total_pixels
        seg_confidence = float(probability_map[component].mean()) if area_pixels else 0.0
        max_seg_confidence = float(probability_map[component].max()) if area_pixels else 0.0
        overlapping_detections = [
            detection
            for detection in detections
            if _boxes_intersect(bbox, tuple(int(round(value)) for value in detection["xyxy"]))
        ]
        yolo_confidence = max((float(detection["confidence"]) for detection in overlapping_detections), default=0.0)
        fused_confidence = fuse_confidence(yolo_confidence, seg_confidence, area_ratio, yolo_weight=yolo_weight)

        defect_review_reasons: list[str] = []
        if fused_confidence < review_threshold:
            defect_review_reasons.append("low_fused_confidence")
        if overlapping_detections and seg_confidence < 0.35:
            defect_review_reasons.append("weak_segmentation_support")
        if not overlapping_detections:
            defect_review_reasons.append("no_overlapping_detection")
        if area_pixels < 25:
            defect_review_reasons.append("tiny_region")

        review_reasons.update(defect_review_reasons)
        defects.append(
            {
                "defect_id": len(defects) + 1,
                "bbox_xyxy": [min_x, min_y, max_x, max_y],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
                "area_pixels": area_pixels,
                "area_ratio": area_ratio,
                "length_pixels": max(max_x - min_x, max_y - min_y),
                "width_pixels": min(max_x - min_x, max_y - min_y),
                "yolo_confidence": yolo_confidence,
                "segmentation_confidence": seg_confidence,
                "max_segmentation_confidence": max_seg_confidence,
                "fused_confidence": fused_confidence,
                "overlapping_detection_count": len(overlapping_detections),
                "review_reasons": defect_review_reasons,
            }
        )

    if detections and not defects:
        review_reasons.add("detection_without_segmentation_support")

    defects.sort(key=lambda defect: float(defect["fused_confidence"]), reverse=True)
    for index, defect in enumerate(defects, start=1):
        defect["defect_id"] = index

    total_defect_area_pixels = int(binary_mask.sum())
    total_defect_area_ratio = total_defect_area_pixels / total_pixels
    fused_confidence = max((float(defect["fused_confidence"]) for defect in defects), default=0.0)
    review_status = "manual_review" if review_reasons else "auto_accept"

    return {
        "defect_count": len(defects),
        "has_multiple_defects": len(defects) > 1,
        "total_defect_area_pixels": total_defect_area_pixels,
        "total_defect_area_ratio": total_defect_area_ratio,
        "fused_confidence": fused_confidence,
        "review_status": review_status,
        "review_reasons": sorted(review_reasons),
        "defects": defects,
    }


def _expand_and_clip_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
    min_size: int,
) -> Box:
    image_width, image_height = image_size
    min_x, min_y, max_x, max_y = box
    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)
    pad_x = int(math.ceil(width * padding_ratio))
    pad_y = int(math.ceil(height * padding_ratio))
    extra_x = max(min_size - width, 0)
    extra_y = max(min_size - height, 0)
    min_x -= pad_x + extra_x // 2
    max_x += pad_x + extra_x - extra_x // 2
    min_y -= pad_y + extra_y // 2
    max_y += pad_y + extra_y - extra_y // 2
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image_width, max_x)
    max_y = min(image_height, max_y)
    return min_x, min_y, max_x, max_y


def _merge_boxes(boxes: Sequence[Box]) -> list[Box]:
    pending = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    changed = True
    while changed:
        changed = False
        merged: list[Box] = []
        for box in pending:
            for index, existing in enumerate(merged):
                if _boxes_intersect(box, existing):
                    merged[index] = (
                        min(box[0], existing[0]),
                        min(box[1], existing[1]),
                        max(box[2], existing[2]),
                        max(box[3], existing[3]),
                    )
                    changed = True
                    break
            else:
                merged.append(box)
        pending = merged
    return sorted(pending)


def _boxes_intersect(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return not (
        box_a[2] <= box_b[0]
        or box_b[2] <= box_a[0]
        or box_a[3] <= box_b[1]
        or box_b[3] <= box_a[1]
    )
