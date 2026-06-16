# Industrial QC Pipeline

Industrial surface-defect inspection pipeline built on `KolektorSDD2`, using `YOLO11` for binary defect detection and `U-Net` for pixel-level defect segmentation.

This repository implements a complete multi-stage workflow for an industrial visual inspection assignment:

1. `YOLO11` checks whether an image contains a defect.
2. If a defect is detected, the system expands the detection into one or more ROI regions.
3. `U-Net` segments the defect on ROI crops from the original image.
4. The pipeline applies mask post-processing, defect quantification, confidence fusion, and manual-review rules.
5. The final output includes structured JSON plus visual overlays for qualitative inspection.

## Features

- Two-stage inspection pipeline: `YOLO11 -> ROI -> U-Net`
- Detection-gated segmentation to avoid unnecessary segmentation on clean samples
- ROI-based segmentation inference on original image crops, not on YOLO mask images
- ROI dataset generation for retraining a segmentation model specifically for cropped defect regions
- Segmentation post-processing with morphological opening/closing and small-component filtering
- Defect quantification:
  - defect count
  - total defect area
  - area ratio
  - per-defect bounding box
  - centroid
  - length and width estimates
- Confidence fusion between YOLO confidence and segmentation confidence
- Manual review mechanism with review reasons such as low fused confidence or weak segmentation support
- Multi-defect statistics in the final JSON output
- W&B integration for YOLO and U-Net training
- ROI vs full-image evaluation script for ablation comparison

## Pipeline

```text
input image
  -> YOLO11 defect detection
  -> if no detection: output no_defect
  -> if detection:
       expand detections into ROI boxes
       crop ROI regions from original image
       run U-Net on each ROI
       merge ROI probability maps back to full resolution
       postprocess mask
       summarize defects
       export JSON + overlays + masks
```

## Repository Layout

```text
industrial_qc_pipeline/
  data/
    raw/                   # raw KolektorSDD2 or smoke dataset
    processed/             # YOLO / U-Net / ROI manifests and converted data
  outputs/                 # training outputs, checkpoints, predictions, evaluations
  requirements/
  scripts/
    prepare_dataset.py
    prepare_roi_dataset.py
    train_yolo.py
    train_unet.py
    predict_pipeline.py
    evaluate_roi_vs_full.py
    setup_env.sh
    with_dns.sh
  src/industrial_qc/
  tests/
```

## Verified Capabilities

The current codebase already includes these implemented modules:

- ROI box generation and merging: [src/industrial_qc/pipeline.py](src/industrial_qc/pipeline.py#L56)
- Segmentation post-processing: [src/industrial_qc/pipeline.py](src/industrial_qc/pipeline.py#L69)
- Confidence fusion, defect quantification, manual review, multi-defect summary: [src/industrial_qc/pipeline.py](src/industrial_qc/pipeline.py#L91)
- End-to-end inference entrypoint: [scripts/predict_pipeline.py](scripts/predict_pipeline.py#L31)
- ROI dataset preparation: [scripts/prepare_roi_dataset.py](scripts/prepare_roi_dataset.py)
- ROI vs full-image ablation evaluation: [scripts/evaluate_roi_vs_full.py](scripts/evaluate_roi_vs_full.py)

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt
```

Or use the helper:

```bash
bash scripts/setup_env.sh
```

For offline remote installs with a local wheelhouse:

```bash
bash scripts/setup_env.sh /path/to/wheelhouse
```

If the remote server has a broken local DNS stub but outbound access still works, run commands through:

```bash
bash scripts/with_dns.sh python -c "import socket; print(socket.gethostbyname('wandb.ai'))"
bash scripts/with_dns.sh bash scripts/setup_env.sh
```

## Dataset

Dataset used in this project:

- [Kolektor Surface-Defect Dataset 2](https://datasetninja.com/kolektor-surface-defect-dataset-2)

Expected raw-data location:

```text
data/raw/KolektorSDD2/
```

The preparation script converts the dataset into:

- YOLO detection format under `data/processed/yolo_ksdd2/`
- U-Net segmentation manifests under `data/processed/unet_ksdd2/`

## Prepare Data

Prepare the real dataset:

```bash
python scripts/prepare_dataset.py \
  --raw-dir data/raw/KolektorSDD2 \
  --output-dir data/processed
```

Prepare a tiny synthetic smoke dataset:

```bash
python scripts/prepare_dataset.py --smoke-run
```

## Training

### Train YOLO11

```bash
python scripts/train_yolo.py \
  --data data/processed/yolo_ksdd2/dataset.yaml \
  --weights yolo11n.pt \
  --epochs 30 \
  --batch 16 \
  --imgsz 640 \
  --workers 4 \
  --device 0 \
  --run-name ksdd2_yolo11n_e30 \
  --wandb-key "$WANDB_API_KEY"
```

Output:

- weights under `outputs/yolo/train/<run-name>/weights/`

### Train Full-Image U-Net

```bash
python scripts/train_unet.py \
  --train-manifest data/processed/unet_ksdd2/train.txt \
  --val-manifest data/processed/unet_ksdd2/val.txt \
  --epochs 30 \
  --batch-size 16 \
  --image-size 256 \
  --num-workers 4 \
  --device cuda:0 \
  --run-name ksdd2_unet_e30 \
  --wandb-key "$WANDB_API_KEY"
```

Output:

- best checkpoint under `outputs/unet/<run-name>/best.pt`

### Prepare ROI Segmentation Dataset

This step converts the full-image segmentation dataset into ROI crops derived from ground-truth defect regions. Negative samples remain as full images.

```bash
python scripts/prepare_roi_dataset.py \
  --train-manifest data/processed/unet_ksdd2/train.txt \
  --val-manifest data/processed/unet_ksdd2/val.txt \
  --test-manifest data/processed/unet_ksdd2/test.txt \
  --output-dir data/processed/unet_ksdd2_roi \
  --roi-padding 0.15 \
  --roi-min-size 32
```

### Train ROI-Retrained U-Net

```bash
python scripts/train_unet.py \
  --train-manifest data/processed/unet_ksdd2_roi/train.txt \
  --val-manifest data/processed/unet_ksdd2_roi/val.txt \
  --epochs 30 \
  --batch-size 16 \
  --image-size 256 \
  --num-workers 4 \
  --device cuda:0 \
  --run-name ksdd2_unet_roi_e30 \
  --wandb-key "$WANDB_API_KEY"
```

## Inference

Single-image pipeline entrypoint:

```bash
python scripts/predict_pipeline.py \
  --image path/to/image.png \
  --yolo-weights outputs/yolo/train/ksdd2_yolo11n_e30/weights/best.pt \
  --unet-weights outputs/unet/ksdd2_unet_roi_e30/best.pt \
  --conf-threshold 0.10 \
  --image-size 256 \
  --mask-threshold 0.5 \
  --roi-padding 0.15 \
  --roi-min-size 32 \
  --post-min-area 25 \
  --opening-iterations 1 \
  --closing-iterations 1 \
  --fuse-yolo-weight 0.6 \
  --review-threshold 0.45 \
  --device cuda:0 \
  --output-dir outputs/predictions/example_case
```

### Important Logic

- YOLO decides whether segmentation should run at all.
- If YOLO finds no defect, the pipeline directly returns `no_defect`.
- If YOLO finds one or more defects, the pipeline computes ROI boxes from YOLO detections.
- U-Net segments the original image ROI crops, then maps ROI probability maps back to full-image space.
- The pipeline does not segment a YOLO-generated mask image.

## Inference Outputs

`predict_pipeline.py` writes a folder containing:

- `detections.png`
- `roi_overlay.png`
- `probability_map.png`
- `raw_mask.png`
- `mask.png`
- `segmentation_overlay.png`
- `result.json`

Example `result.json` fields:

```json
{
  "decision": "defect",
  "review_status": "manual_review",
  "review_reasons": ["low_fused_confidence"],
  "fused_confidence": 0.62,
  "has_multiple_defects": true,
  "mask_summary": {
    "defect_count": 2,
    "total_defect_area_pixels": 1843,
    "total_defect_area_ratio": 0.021
  },
  "defects": [
    {
      "defect_id": 1,
      "bbox_xyxy": [42, 105, 89, 166],
      "centroid_xy": [65.4, 133.2],
      "area_pixels": 920,
      "length_pixels": 61,
      "width_pixels": 47,
      "yolo_confidence": 0.81,
      "segmentation_confidence": 0.73,
      "fused_confidence": 0.78,
      "review_reasons": []
    }
  ]
}
```

## ROI Ablation Evaluation

To compare:

- full-image U-Net
- ROI inference without retraining
- ROI inference with ROI retraining

run:

```bash
python scripts/evaluate_roi_vs_full.py \
  --manifest data/processed/unet_ksdd2/test.txt \
  --yolo-weights outputs/yolo/train/ksdd2_yolo11n_e30/weights/best.pt \
  --full-unet-weights outputs/unet/ksdd2_unet_e30/best.pt \
  --roi-unet-weights outputs/unet/ksdd2_unet_roi_e30/best.pt \
  --legacy-roi-unet-weights outputs/unet/ksdd2_unet_e30/best.pt \
  --conf-threshold 0.10 \
  --device cuda:1 \
  --output outputs/roi_retrain_comparison.json
```

Output:

- comparison JSON under `outputs/roi_retrain_comparison.json`

## Latest Verified Experimental Result

Latest verified ROI retraining comparison was run on `June 15, 2026` on the test set with:

- `1004` total test images
- `110` positive images
- `89` positive images detected by YOLO at confidence threshold `0.10`

### Full-image U-Net vs ROI-Retrained U-Net

All positive samples:

- full-image U-Net: `Dice = 0.5727`, `IoU = 0.4745`
- ROI-retrained U-Net: `Dice = 0.6013`, `IoU = 0.4982`

YOLO-detected positive samples only:

- full-image U-Net: `Dice = 0.7079`, `IoU = 0.5864`
- ROI-retrained U-Net: `Dice = 0.7431`, `IoU = 0.6158`

Per-case comparison:

- ROI-retrained better on `52` cases
- full-image better on `36` cases
- tie on `22` cases

### Legacy ROI Inference Without Retraining

All positive samples:

- full-image U-Net: `Dice = 0.5727`
- legacy ROI inference: `Dice = 0.5414`

This verifies an important conclusion:

- ROI cropping alone was not enough
- ROI retraining was necessary to make ROI segmentation effective

## W&B

Set `WANDB_API_KEY` or pass `--wandb-key` to the training scripts.

- YOLO training uses Ultralytics W&B logging when login succeeds.
- U-Net training logs epoch-wise `train_loss`, `val_loss`, `precision`, `recall`, `dice`, and `iou`.

## Tests

Run the local test suite:

```bash
pytest tests -q
```

Current verified local status:

- `13 passed`

## Notes

- The YOLO and U-Net train/val/test splits are aligned through the same prepared dataset workflow.
- ROI U-Net training uses ROI-cropped segmentation manifests derived from the same original split.
- The repository is designed for remote-server training and local/public code review.
