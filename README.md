# Industrial QC Pipeline

Two-stage industrial defect inspection pipeline for `KolektorSDD2` using `YOLO11` for defect detection and `U-Net` for semantic segmentation.

## Layout

```text
industrial_qc_pipeline/
  data/
    raw/
    processed/
  outputs/
  requirements/
  scripts/
  src/industrial_qc/
  tests/
```

## Quickstart

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

Prepare a tiny synthetic dataset and generate YOLO plus segmentation splits:

```bash
python scripts/prepare_dataset.py --smoke-run
```

Train the U-Net smoke run:

```bash
python scripts/train_unet.py --epochs 1 --batch-size 2 --run-name smoke_unet
```

Train YOLO once `yolo11n.pt` and processed data are available:

```bash
python scripts/train_yolo.py --epochs 1 --batch 2 --run-name smoke_yolo
```

## W&B

Set `WANDB_API_KEY` or pass `--wandb-key` to the training scripts. The YOLO training script enables Ultralytics W&B logging with `settings.update({"wandb": True})` when login succeeds.
