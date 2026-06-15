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
from industrial_qc.pipeline import (
    compute_roi_boxes,
    draw_boxes,
    filter_detections,
    postprocess_mask,
    overlay_mask_on_image,
    save_binary_mask,
    should_run_segmentation,
    summarize_defects,
)
from industrial_qc.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the YOLO11 -> U-Net inference pipeline.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--yolo-weights", type=Path, required=True, help="YOLO weights path.")
    parser.add_argument("--unet-weights", type=Path, required=True, help="U-Net checkpoint path.")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--roi-padding", type=float, default=0.15)
    parser.add_argument("--roi-min-size", type=int, default=32)
    parser.add_argument("--post-min-area", type=int, default=25)
    parser.add_argument("--opening-iterations", type=int, default=1)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--fuse-yolo-weight", type=float, default=0.6)
    parser.add_argument("--review-threshold", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _predict_probability_map(model, image: Image.Image, roi_boxes: list[tuple[int, int, int, int]], image_size: int, device: str):
    import torch

    probability_map = np.zeros((image.height, image.width), dtype=np.float32)
    resampling = getattr(Image, "Resampling", Image)
    for min_x, min_y, max_x, max_y in roi_boxes:
        crop = image.crop((min_x, min_y, max_x, max_y))
        resized = crop.resize((image_size, image_size), resampling.BILINEAR)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            roi_probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        roi_probability_image = Image.fromarray((roi_probability * 255).astype(np.uint8), mode="L")
        roi_probability_resized = roi_probability_image.resize((max_x - min_x, max_y - min_y), resampling.BILINEAR)
        roi_probability_array = np.asarray(roi_probability_resized, dtype=np.float32) / 255.0
        probability_map[min_y:max_y, min_x:max_x] = np.maximum(
            probability_map[min_y:max_y, min_x:max_x],
            roi_probability_array,
        )
    return probability_map


def _save_probability_map(probability_map: np.ndarray, output_path: Path) -> None:
    Image.fromarray((np.clip(probability_map, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(output_path)


def main() -> None:
    args = parse_args()

    import torch
    from ultralytics import YOLO

    from industrial_qc.models.unet import UNet

    paths = ProjectPaths.from_root(PROJECT_ROOT)
    paths.ensure()
    output_dir = (args.output_dir or paths.outputs_dir / "predictions" / args.image.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    yolo_model = YOLO(str(args.yolo_weights))
    yolo_result = yolo_model.predict(source=str(args.image), conf=args.conf_threshold, device=device, verbose=False)[0]
    detections = [
        {
            "xyxy": [float(value) for value in box.xyxy[0].tolist()],
            "confidence": float(box.conf.item()),
            "cls": int(box.cls.item()),
        }
        for box in yolo_result.boxes
    ]
    detections = filter_detections(detections, args.conf_threshold)

    source_image = Image.open(args.image).convert("RGB")
    detection_image = draw_boxes(source_image, detections)
    detection_path = output_dir / "detections.png"
    detection_image.save(detection_path)

    result_payload: dict[str, object] = {
        "image": str(args.image),
        "decision": "no_defect",
        "detections": detections,
        "detection_overlay": str(detection_path),
        "review_status": "auto_accept",
        "review_reasons": [],
        "fused_confidence": 0.0,
        "has_multiple_defects": False,
        "defects": [],
    }

    if should_run_segmentation(detections):
        model = UNet().to(device)
        checkpoint = torch.load(args.unet_weights, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        roi_boxes = compute_roi_boxes(
            detections,
            image_size=source_image.size,
            padding_ratio=args.roi_padding,
            min_size=args.roi_min_size,
        )
        roi_overlay = draw_boxes(
            source_image,
            [{"xyxy": list(box)} for box in roi_boxes],
            color="#FFAA00",
            annotate=False,
        )
        roi_overlay_path = output_dir / "roi_overlay.png"
        roi_overlay.save(roi_overlay_path)

        probability_map = _predict_probability_map(model, source_image, roi_boxes, args.image_size, device)
        probability_map_path = output_dir / "probability_map.png"
        _save_probability_map(probability_map, probability_map_path)

        raw_mask = (probability_map >= args.mask_threshold).astype(np.uint8)
        raw_mask_path = output_dir / "raw_mask.png"
        save_binary_mask(raw_mask, raw_mask_path)

        final_mask = postprocess_mask(
            raw_mask,
            min_area=args.post_min_area,
            opening_iterations=args.opening_iterations,
            closing_iterations=args.closing_iterations,
        )

        overlay = overlay_mask_on_image(source_image, final_mask)
        overlay_path = output_dir / "segmentation_overlay.png"
        mask_path = output_dir / "mask.png"
        overlay.save(overlay_path)
        save_binary_mask(final_mask, mask_path)
        defect_summary = summarize_defects(
            final_mask,
            probability_map,
            detections,
            image_size=source_image.size,
            review_threshold=args.review_threshold,
            yolo_weight=args.fuse_yolo_weight,
        )

        result_payload.update(
            {
                "decision": "defect" if int(defect_summary["defect_count"]) > 0 else "no_defect",
                "segmentation_trigger": "roi",
                "roi_boxes": [list(box) for box in roi_boxes],
                "roi_overlay": str(roi_overlay_path),
                "segmentation_overlay": str(overlay_path),
                "probability_map_path": str(probability_map_path),
                "raw_mask_path": str(raw_mask_path),
                "mask_path": str(mask_path),
                "review_status": defect_summary["review_status"],
                "review_reasons": defect_summary["review_reasons"],
                "fused_confidence": defect_summary["fused_confidence"],
                "has_multiple_defects": defect_summary["has_multiple_defects"],
                "mask_summary": {
                    "defect_count": defect_summary["defect_count"],
                    "total_defect_area_pixels": defect_summary["total_defect_area_pixels"],
                    "total_defect_area_ratio": defect_summary["total_defect_area_ratio"],
                },
                "defects": defect_summary["defects"],
            }
        )

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
