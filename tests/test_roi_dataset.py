import numpy as np
from PIL import Image

from industrial_qc.data.roi import build_roi_samples_for_record, write_roi_dataset


def test_build_roi_samples_for_record_splits_positive_components(tmp_path):
    image_path = tmp_path / "sample.png"
    mask_path = tmp_path / "sample_mask.png"

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 3:6] = 255
    mask[12:16, 13:18] = 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    samples = build_roi_samples_for_record(
        image_path=image_path,
        mask_path=mask_path,
        sample_stem="sample",
        padding_ratio=0.0,
        min_size=1,
    )

    assert len(samples) == 2
    assert samples[0].crop_box == (3, 2, 6, 5)
    assert samples[1].crop_box == (13, 12, 18, 16)
    assert samples[0].is_defect is True
    assert samples[1].is_defect is True


def test_build_roi_samples_for_record_keeps_negative_as_full_image(tmp_path):
    image_path = tmp_path / "negative.png"
    mask_path = tmp_path / "negative_mask.png"

    image = np.zeros((18, 12, 3), dtype=np.uint8)
    mask = np.zeros((18, 12), dtype=np.uint8)
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    samples = build_roi_samples_for_record(
        image_path=image_path,
        mask_path=mask_path,
        sample_stem="negative",
        padding_ratio=0.15,
        min_size=32,
    )

    assert len(samples) == 1
    assert samples[0].crop_box == (0, 0, 12, 18)
    assert samples[0].is_defect is False


def test_write_roi_dataset_creates_split_manifests(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    positive_image = np.zeros((10, 10, 3), dtype=np.uint8)
    positive_mask = np.zeros((10, 10), dtype=np.uint8)
    positive_mask[1:4, 2:5] = 255
    negative_image = np.zeros((10, 10, 3), dtype=np.uint8)
    negative_mask = np.zeros((10, 10), dtype=np.uint8)

    positive_image_path = image_dir / "positive.png"
    positive_mask_path = mask_dir / "positive_mask.png"
    negative_image_path = image_dir / "negative.png"
    negative_mask_path = mask_dir / "negative_mask.png"
    Image.fromarray(positive_image).save(positive_image_path)
    Image.fromarray(positive_mask).save(positive_mask_path)
    Image.fromarray(negative_image).save(negative_image_path)
    Image.fromarray(negative_mask).save(negative_mask_path)

    manifests = {
        "train": [(positive_image_path, positive_mask_path)],
        "val": [(negative_image_path, negative_mask_path)],
        "test": [],
    }

    summary = write_roi_dataset(
        manifests,
        output_root=tmp_path / "roi_dataset",
        padding_ratio=0.0,
        min_size=1,
    )

    train_lines = summary["train_manifest"].read_text(encoding="utf-8").splitlines()
    val_lines = summary["val_manifest"].read_text(encoding="utf-8").splitlines()

    assert len(train_lines) == 1
    assert len(val_lines) == 1
    train_image, train_mask = train_lines[0].split("\t")
    val_image, val_mask = val_lines[0].split("\t")
    assert Image.open(train_image).size == (3, 3)
    assert Image.open(train_mask).size == (3, 3)
    assert Image.open(val_image).size == (10, 10)
    assert Image.open(val_mask).size == (10, 10)
