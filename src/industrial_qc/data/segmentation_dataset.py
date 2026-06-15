from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class SegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, manifest_path: str | Path, image_size: int = 256) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.samples = self._load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor

    @staticmethod
    def _load_manifest(manifest_path: Path) -> list[tuple[Path, Path]]:
        lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [(Path(image_path), Path(mask_path)) for image_path, mask_path in (line.split("\t") for line in lines)]
