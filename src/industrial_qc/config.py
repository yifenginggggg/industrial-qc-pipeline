from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    outputs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(root=root, data_dir=root / "data", outputs_dir=root / "outputs")

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.outputs_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
