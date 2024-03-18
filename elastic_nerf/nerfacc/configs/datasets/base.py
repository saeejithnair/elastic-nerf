"""
Copyright (c) 2023 Saeejith Nair, University of Waterloo.
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
import os
import subprocess
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
import tyro
from gonas.configs.base_config import InstantiateConfig, PrintableConfig
from lpips import LPIPS
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.scripts.downloads.download_data import BlenderDownload
from torchmetrics.functional import structural_similarity_index_measure
from tyro.extras._serialization import to_yaml, from_yaml
import copy
import wandb


@dataclass
class NGPBaseDatasetConfig(PrintableConfig):
    """Base dataset configuration."""

    data_root: Path
    """The root directory of the dataset."""
    scene: str
    """Which scene to use."""
    init_batch_size: int = 1024
    """Initial batch size."""
    weight_decay: float = 0.0
    """Weight decay."""
    optimizer_lr: float = 1e-2
    """Learning rate for the optimizer."""
    optimizer_eps: float = 1e-15
    """Epsilon for the optimizer."""
    scheduler_start_factor: float = 0.01
    """Start factor for the scheduler."""
    scheduler_total_iters: int = 100
    """Total iterations for the scheduler."""
    scheduler_gamma: float = 0.33
    """Gamma for the scheduler."""
    aabb_coeffs: Tuple[float, float, float, float, float, float] = (
        -1.5,
        -1.5,
        -1.5,
        1.5,
        1.5,
        1.5,
    )
    """Coefficients for the AABB."""
    near_plane: float = 0.0
    """Near plane."""
    far_plane: float = 1.0e10
    """Far plane."""
    train_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the train dataset."""
    test_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the test dataset."""
    train_split: Literal["train", "trainval"] = "train"
    """Which train split to use."""
    test_chunk_size: int = 8192
    """Chunk size for testing."""

    def download(self):
        raise NotImplementedError

    def setup(self, **kwargs):
        raise NotImplementedError


@dataclass
class NGPOccDatasetConfig(NGPBaseDatasetConfig):
    """Base dataset configuration for NGPOcc."""

    init_batch_size: int = 1024
    """Initial batch size."""
    target_sample_batch_size: int = 1 << 18
    """Target sample batch size."""
    near_plane: float = 0.0
    """Near plane."""
    far_plane: float = 1.0e10
    """Far plane."""
    grid_resolution: int = 128
    """Resolution of the grid."""
    grid_nlvl: int = 1
    """Number of levels of the grid."""
    render_step_size: float = 5e-3
    """Step size for rendering."""
    alpha_thre: float = 0.0
    """Threshold for alpha."""
    cone_angle: float = 0.0
    """Cone angle."""
    num_dynamic_batch_warmup_steps: int = 20
    """Number of warmup steps for dynamic batch size."""

    def get_dataloader(self):
        raise NotImplementedError


@dataclass
class NGPPropDatasetConfig(NGPBaseDatasetConfig):
    """Dataset/scene specific configurations."""

    init_batch_size: int = 4096
    """Initial batch size."""
    near_plane: float = 2.0
    """Near plane."""
    far_plane: float = 6.0
    """Far plane."""
    unbounded: bool = False
    """Whether the scene is unbounded."""
    num_samples: int = 64
    """Number of samples."""
    num_samples_per_prop: List[int] = field(default_factory=lambda: [128])
    """Number of samples per proposal."""
    prop_network_resolutions: List[int] = field(default_factory=lambda: [128])
    """Max resolutions of the proposal network."""
    sampling_type: Literal["uniform", "lindisp"] = "uniform"
    """Sampling type."""
    opaque_bkgd: bool = False
    """Whether to use opaque background."""


@dataclass
class VanillaNeRFDatasetConfig(NGPBaseDatasetConfig):
    """Dataset/scene specific configurations."""

    init_batch_size: int = 1024
    """Initial batch size."""
    target_sample_batch_size: int = 1 << 16
    """Target sample batch size."""
    near_plane: float = 0.0
    """The near plane of the camera."""
    far_plane: float = 1.0e10
    """The far plane of the camera."""
    grid_resolution: int = 128
    """The resolution of the grid."""
    grid_nlvl: int = 1
    """The number of levels of the grid."""
    render_step_size: float = 5e-3
    """The step size for rendering."""
    num_dynamic_batch_warmup_steps: int = 20
    """Number of warmup steps for dynamic batch size."""
