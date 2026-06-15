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
from industrial_qc.data.roi import load_segmentation_manifest
from industrial_qc.pipeline import compute_roi_boxes, filter_detections, postprocess_mask
from industrial_qc.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full-image and ROI segmentation strategies on a test manifest.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--full-unet-weights", type=Path, required=True)
    parser.add_argument("--roi-unet-weights", type=Path, required=True)
    parser.add_argument("--legacy-roi-unet-weights", type=Path, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--roi-padding", type=float, default=0.15)
    parser.add_argument("--roi-min-size", type=int, default=32)
    parser.add_argument("--post-min-area", type=int, default=25)
    parser.add_argument("--opening-iterations", type=int, default=1)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = int((pred & gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = int((pred | gt).sum())
    dice = (2 * inter / (pred_sum + gt_sum)) if (pred_sum + gt_sum) else 1.0
    iou = (inter / union) if union else 1.0
    return dice, iou


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def load_unet(weights_path: Path, device: str):
    import torch

    from industrial_qc.models.unet import UNet

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_full_mask(model, image: Image.Image, image_size: int, mask_threshold: float, device: str) -> np.ndarray:
    import torch

    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((image_size, image_size), resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask_img = Image.fromarray((prob * 255).astype(np.uint8), mode="L").resize(image.size, resampling.BILINEAR)
    prob_full = np.asarray(mask_img, dtype=np.float32) / 255.0
    return (prob_full >= mask_threshold).astype(np.uint8)


def predict_roi_mask(
    model,
    image: Image.Image,
    detections: list[dict[str, object]],
    image_size: int,
    mask_threshold: float,
    roi_padding: float,
    roi_min_size: int,
    device: str,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    import torch

    resampling = getattr(Image, "Resampling", Image)
    probability_map = np.zeros((image.height, image.width), dtype=np.float32)
    roi_boxes = compute_roi_boxes(detections, image_size=image.size, padding_ratio=roi_padding, min_size=roi_min_size)
    for min_x, min_y, max_x, max_y in roi_boxes:
        crop = image.crop((min_x, min_y, max_x, max_y))
        resized = crop.resize((image_size, image_size), resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        roi_img = Image.fromarray((prob * 255).astype(np.uint8), mode="L").resize((max_x - min_x, max_y - min_y), resampling.BILINEAR)
        roi_arr = np.asarray(roi_img, dtype=np.float32) / 255.0
        probability_map[min_y:max_y, min_x:max_x] = np.maximum(probability_map[min_y:max_y, min_x:max_x], roi_arr)
    return (probability_map >= mask_threshold).astype(np.uint8), roi_boxes


def build_summary(cases: list[dict[str, object]], baseline_key: str, compare_key: str) -> dict[str, object]:
    positive_cases = [case for case in cases if case["is_positive"]]
    positive_detected = [case for case in positive_cases if case["has_detection"]]
    better = [case for case in positive_cases if float(case[f"{compare_key}_dice"]) - float(case[f"{baseline_key}_dice"]) > 1e-6]
    worse = [case for case in positive_cases if float(case[f"{compare_key}_dice"]) - float(case[f"{baseline_key}_dice"]) < -1e-6]
    tied = [case for case in positive_cases if abs(float(case[f"{compare_key}_dice"]) - float(case[f"{baseline_key}_dice"])) <= 1e-6]

    return {
        "positive_all": {
            f"{baseline_key}_mean_dice": mean([float(case[f"{baseline_key}_dice"]) for case in positive_cases]),
            f"{compare_key}_mean_dice": mean([float(case[f"{compare_key}_dice"]) for case in positive_cases]),
            f"{baseline_key}_mean_iou": mean([float(case[f"{baseline_key}_iou"]) for case in positive_cases]),
            f"{compare_key}_mean_iou": mean([float(case[f"{compare_key}_iou"]) for case in positive_cases]),
        },
        "positive_detected_only": {
            f"{baseline_key}_mean_dice": mean([float(case[f"{baseline_key}_dice"]) for case in positive_detected]),
            f"{compare_key}_mean_dice": mean([float(case[f"{compare_key}_dice"]) for case in positive_detected]),
            f"{baseline_key}_mean_iou": mean([float(case[f"{baseline_key}_iou"]) for case in positive_detected]),
            f"{compare_key}_mean_iou": mean([float(case[f"{compare_key}_iou"]) for case in positive_detected]),
        },
        "counts": {
            f"{compare_key}_better": len(better),
            f"{baseline_key}_better": len(worse),
            "tie": len(tied),
        },
        "top_gains": sorted(
            (
                {
                    **case,
                    "dice_delta": float(case[f"{compare_key}_dice"]) - float(case[f"{baseline_key}_dice"]),
                    "iou_delta": float(case[f"{compare_key}_iou"]) - float(case[f"{baseline_key}_iou"]),
                }
                for case in positive_cases
            ),
            key=lambda case: case["dice_delta"],
            reverse=True,
        )[:5],
        "top_losses": sorted(
            (
                {
                    **case,
                    "dice_delta": float(case[f"{compare_key}_dice"]) - float(case[f"{baseline_key}_dice"]),
                    "iou_delta": float(case[f"{compare_key}_iou"]) - float(case[f"{baseline_key}_iou"]),
                }
                for case in positive_cases
            ),
            key=lambda case: case["dice_delta"],
        )[:5],
    }


def main() -> None:
    args = parse_args()
    import torch
    from ultralytics import YOLO

    paths = ProjectPaths.from_root(PROJECT_ROOT)
    paths.ensure()
    manifest = args.manifest or (paths.processed_data_dir / "unet_ksdd2" / "test.txt")
    output_path = args.output or (paths.outputs_dir / "roi_retrain_comparison.json")
    device = resolve_device(args.device)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    yolo_model = YOLO(str(args.yolo_weights))
    full_model = load_unet(args.full_unet_weights, device)
    roi_model = load_unet(args.roi_unet_weights, device)
    legacy_roi_model = load_unet(args.legacy_roi_unet_weights, device) if args.legacy_roi_unet_weights else None

    cases: list[dict[str, object]] = []
    for image_path, mask_path in load_segmentation_manifest(manifest):
        image = Image.open(image_path).convert("RGB")
        gt = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
        yolo_result = yolo_model.predict(source=str(image_path), conf=args.conf_threshold, device=device, verbose=False)[0]
        detections = [
            {
                "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                "confidence": float(box.conf.item()),
                "cls": int(box.cls.item()),
            }
            for box in yolo_result.boxes
        ]
        detections = filter_detections(detections, args.conf_threshold)
        has_detection = len(detections) > 0

        full_raw = predict_full_mask(full_model, image, args.image_size, args.mask_threshold, device)
        full_post = postprocess_mask(
            full_raw,
            min_area=args.post_min_area,
            opening_iterations=args.opening_iterations,
            closing_iterations=args.closing_iterations,
        )
        full_gated = full_post if has_detection else np.zeros_like(gt)

        if has_detection:
            roi_raw, roi_boxes = predict_roi_mask(
                roi_model,
                image,
                detections,
                args.image_size,
                args.mask_threshold,
                args.roi_padding,
                args.roi_min_size,
                device,
            )
            roi_post = postprocess_mask(
                roi_raw,
                min_area=args.post_min_area,
                opening_iterations=args.opening_iterations,
                closing_iterations=args.closing_iterations,
            )
            if legacy_roi_model is not None:
                legacy_raw, _ = predict_roi_mask(
                    legacy_roi_model,
                    image,
                    detections,
                    args.image_size,
                    args.mask_threshold,
                    args.roi_padding,
                    args.roi_min_size,
                    device,
                )
                legacy_post = postprocess_mask(
                    legacy_raw,
                    min_area=args.post_min_area,
                    opening_iterations=args.opening_iterations,
                    closing_iterations=args.closing_iterations,
                )
            else:
                legacy_post = None
        else:
            roi_boxes = []
            roi_post = np.zeros_like(gt)
            legacy_post = np.zeros_like(gt) if legacy_roi_model is not None else None

        full_dice, full_iou = dice_iou(full_gated, gt)
        roi_dice, roi_iou = dice_iou(roi_post, gt)

        case = {
            "image": str(image_path),
            "is_positive": bool(gt.sum() > 0),
            "has_detection": has_detection,
            "num_detections": len(detections),
            "roi_box_count": len(roi_boxes),
            "gt_pixels": int(gt.sum()),
            "full_full_dice": full_dice,
            "full_full_iou": full_iou,
            "roi_roi_dice": roi_dice,
            "roi_roi_iou": roi_iou,
        }
        if legacy_post is not None:
            legacy_dice, legacy_iou = dice_iou(legacy_post, gt)
            case["legacy_roi_dice"] = legacy_dice
            case["legacy_roi_iou"] = legacy_iou
        cases.append(case)

    payload: dict[str, object] = {
        "config": {
            "manifest": str(manifest),
            "device": device,
            "conf_threshold": args.conf_threshold,
            "image_size": args.image_size,
            "mask_threshold": args.mask_threshold,
            "roi_padding": args.roi_padding,
            "roi_min_size": args.roi_min_size,
            "post_min_area": args.post_min_area,
            "opening_iterations": args.opening_iterations,
            "closing_iterations": args.closing_iterations,
        },
        "dataset": {
            "all_test": len(cases),
            "positive_test": len([case for case in cases if case["is_positive"]]),
            "positive_detected_by_yolo": len([case for case in cases if case["is_positive"] and case["has_detection"]]),
        },
        "full_vs_roi": build_summary(cases, baseline_key="full_full", compare_key="roi_roi"),
    }
    if args.legacy_roi_unet_weights:
        payload["full_vs_legacy_roi"] = build_summary(cases, baseline_key="full_full", compare_key="legacy_roi")
        payload["legacy_roi_vs_roi"] = build_summary(cases, baseline_key="legacy_roi", compare_key="roi_roi")

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
