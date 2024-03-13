from elastic_nerf.nerfacc.configs.datasets.base import (
    NGPOccDatasetConfig,
    NGPPropDatasetConfig,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from pathlib import Path
import os
from elastic_nerf.nerfacc.datasets.nerf_synthetic import (
    SubjectLoader as BlenderSyntheticLoader,
)
from nerfstudio.scripts.downloads.download_data import BlenderDownload


def download_dataset(save_dir):
    downloader = BlenderDownload(save_dir=save_dir)
    downloader.download(save_dir=save_dir)


@dataclass
class BlenderSyntheticDatasetOccConfig(NGPOccDatasetConfig):
    """Dataset/scene specific configurations for Blender Synthetic dataset."""

    data_root: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"])
        / "data/blender"
    )
    """The root directory of the dataset."""
    scene: Literal[
        "chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"
    ] = "lego"
    """Which scene to use."""

    def __post_init__(self):
        self.weight_decay = (
            1e-5 if self.scene in ["materials", "ficus", "drums"] else 1e-6
        )

    def setup(self, **kwargs) -> BlenderSyntheticLoader:
        return BlenderSyntheticLoader(**kwargs)

    def download(self):
        download_dataset(self.data_root.parent)


@dataclass
class BlenderSyntheticDatasetPropConfig(NGPPropDatasetConfig):
    """Dataset/scene specific configurations for Blender Synthetic dataset."""

    subject_loader: Type[BlenderSyntheticLoader] = field(
        default_factory=lambda: BlenderSyntheticLoader
    )
    """The subject loader."""
    data_root: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"])
        / "data/blender"
    )
    """The root directory of the dataset."""
    scene: Literal[
        "chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"
    ] = "lego"
    """Which scene to use."""

    def __post_init__(self):
        self.weight_decay = (
            1e-5 if self.scene in ["materials", "ficus", "drums"] else 1e-6
        )

    def download(self):
        download_dataset(self.data_root.parent)
