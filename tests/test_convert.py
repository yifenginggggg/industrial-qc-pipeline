import numpy as np
from PIL import Image

from industrial_qc.data.convert import discover_dataset_records, mask_to_bboxes


def test_mask_to_bboxes_finds_single_component():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 3:6] = 255

    assert mask_to_bboxes(mask) == [(3, 2, 5, 4)]


def test_mask_to_bboxes_splits_two_components():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:4] = 255
    mask[6:9, 7:9] = 255

    assert mask_to_bboxes(mask) == [(1, 1, 3, 2), (7, 6, 8, 8)]


def test_discover_dataset_records_ignores_mask_hints_outside_dataset_root(tmp_path):
    root = tmp_path / "zhanglongteng_workspace" / "raw"
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:5] = 255
    Image.fromarray(image).save(image_dir / "sample_000.png")
    Image.fromarray(mask).save(mask_dir / "sample_000_mask.png")

    records = discover_dataset_records(root)

    assert len(records) == 1
    assert records[0].is_defect is True
