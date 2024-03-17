from elastic_nerf.nerfacc.configs.datasets.base import (
    NGPOccDatasetConfig,
    NGPPropDatasetConfig,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from pathlib import Path
import os
from elastic_nerf.nerfacc.datasets.nerf_360_v2 import SubjectLoader as MipNerf360Loader
import subprocess
import zipfile


def download_dataset(save_dir):
    dataset_url = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    zip_path = Path(save_dir.parent, "360_v2.zip")
    subprocess.run(["wget", "-O", str(zip_path), dataset_url], check=True)

    # Unzip the file to self.dataset.data_root.parent / "360_v2"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        save_dir.mkdir(parents=True, exist_ok=True)
        zip_ref.extractall(save_dir)

    # Optionally, delete the zip file after extraction
    zip_path.unlink()


@dataclass
class MipNerf360DatasetOccConfig(NGPOccDatasetConfig):
    """Dataset/scene specific configurations for Mip-NeRF 360 dataset."""

    data_root: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"]) / "data/360_v2"
    )
    """The root directory of the dataset."""
    scene: Literal[
        "garden",
        "bicycle",
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "stump",
    ] = "bicycle"
    """Which scene to use."""
    aabb_coeffs: Tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )
    """Coefficients for the AABB."""
    near_plane: float = 0.2
    """Near plane."""
    train_dataset_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"color_bkgd_aug": "random", "factor": 4}
    )
    """Keyword arguments for the train dataset."""
    test_dataset_kwargs: Dict[str, Any] = field(default_factory=lambda: {"factor": 4})
    """Keyword arguments for the test dataset."""
    grid_nlvl: int = 4
    """Number of levels of the grid."""
    render_step_size: float = 1e-3
    """Step size for rendering."""
    alpha_thre: float = 1e-2
    """Threshold for alpha."""
    cone_angle: float = 0.004
    """Cone angle."""

    def setup(self, **kwargs) -> MipNerf360Loader:
        return MipNerf360Loader(**kwargs)

    def download(self):
        download_dataset(self.data_root)


@dataclass
class MipNerf360DatasetPropConfig(NGPPropDatasetConfig):
    """Dataset/scene specific configurations for Mip-NeRF 360 dataset."""

    data_root: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"]) / "data/360_v2"
    )
    """The root directory of the dataset."""
    scene: Literal[
        "garden",
        "bicycle",
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "stump",
    ] = "bicycle"
    """Which scene to use."""
    aabb_coeffs: Tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )
    """Coefficients for the AABB."""
    unbounded: bool = True
    """Whether the scene is unbounded."""
    near_plane: float = 0.2
    """Near plane."""
    far_plane: float = 1e3
    """Far plane."""
    train_dataset_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"color_bkgd_aug": "random", "factor": 4}
    )
    """Keyword arguments for the train dataset."""
    test_dataset_kwargs: Dict[str, Any] = field(default_factory=lambda: {"factor": 4})
    """Keyword arguments for the test dataset."""
    num_samples: int = 48
    """Number of samples."""
    num_samples_per_prop: List[int] = field(default_factory=lambda: [256, 96])
    """Number of samples per proposal."""
    prop_network_resolutions: List[int] = field(default_factory=lambda: [128, 256])
    """Max resolutions of the proposal network."""
    sampling_type: Literal["lindisp"] = "lindisp"
    """Sampling type."""
    opaque_bkgd: bool = True
    """Whether to use opaque background."""

    def download(self):
        download_dataset(self.data_root)

    def setup(self, **kwargs) -> MipNerf360Loader:
        return MipNerf360Loader(**kwargs)
