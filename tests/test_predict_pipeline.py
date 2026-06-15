import numpy as np

from industrial_qc.pipeline import (
    compute_roi_boxes,
    fuse_confidence,
    postprocess_mask,
    should_run_segmentation,
    summarize_defects,
)

def test_should_run_segmentation_requires_one_detection():
    assert should_run_segmentation([]) is False


def test_compute_roi_boxes_merges_overlapping_detections_and_clips_bounds():
    detections = [
        {"xyxy": [2.0, 3.0, 15.0, 20.0], "confidence": 0.7, "cls": 0},
        {"xyxy": [12.0, 10.0, 28.0, 24.0], "confidence": 0.6, "cls": 0},
        {"xyxy": [70.0, 75.0, 90.0, 95.0], "confidence": 0.8, "cls": 0},
    ]

    rois = compute_roi_boxes(detections, image_size=(100, 100), padding_ratio=0.0, min_size=10)

    assert rois == [(2, 3, 28, 24), (70, 75, 90, 95)]


def test_postprocess_mask_removes_small_regions_and_keeps_large_component():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[0, 0] = 1
    mask[11, 11] = 1

    processed = postprocess_mask(mask, min_area=8, opening_iterations=0, closing_iterations=0)

    assert processed.sum() == 36
    assert processed[0, 0] == 0
    assert processed[11, 11] == 0
    assert processed[4, 4] == 1


def test_summarize_defects_reports_multiple_regions_and_review_reason():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[6:9, 6:9] = 1
    probability_map = np.zeros((10, 10), dtype=np.float32)
    probability_map[1:3, 1:3] = 0.9
    probability_map[6:9, 6:9] = 0.35
    detections = [
        {"xyxy": [1.0, 1.0, 3.0, 3.0], "confidence": 0.85, "cls": 0},
        {"xyxy": [6.0, 6.0, 9.0, 9.0], "confidence": 0.25, "cls": 0},
    ]

    summary = summarize_defects(
        mask,
        probability_map,
        detections,
        image_size=(10, 10),
        review_threshold=0.45,
        yolo_weight=0.6,
    )

    assert summary["defect_count"] == 2
    assert summary["total_defect_area_pixels"] == 13
    assert summary["review_status"] == "manual_review"
    assert "low_fused_confidence" in summary["review_reasons"]
    assert summary["defects"][0]["fused_confidence"] > summary["defects"][1]["fused_confidence"]


def test_fuse_confidence_balances_detection_and_segmentation_evidence():
    high = fuse_confidence(yolo_confidence=0.8, segmentation_confidence=0.9, area_ratio=0.02, yolo_weight=0.6)
    low = fuse_confidence(yolo_confidence=0.2, segmentation_confidence=0.3, area_ratio=0.001, yolo_weight=0.6)

    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
