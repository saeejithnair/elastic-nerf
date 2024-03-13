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
from elastic_nerf.nerfacc.datasets.nerf_360_v2 import SubjectLoader as MipNerf360Loader
from elastic_nerf.nerfacc.datasets.nerf_synthetic import (
    SubjectLoader as BlenderSyntheticLoader,
)
from elastic_nerf.nerfacc.radiance_fields.ngp import (
    NGPRadianceField,
    NGPRadianceFieldConfig,
)
from elastic_nerf.nerfacc.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from elastic_nerf.utils import logging_utils as lu
from nerfacc.estimators.prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from elastic_nerf.nerfacc.configs.dataset import (
    NGPDatasetConfig,
    BlenderSyntheticDatasetConfig,
    MipNerf360DatasetConfig,
)

set_random_seed(42)

@dataclass
class NGPOccTrainerConfig(InstantiateConfig):
    """Configurations for training the model."""

    _target: Type = field(default_factory=lambda: NGPOccTrainer)
    """The target class to instantiate."""

    exp_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d-%H-%M-%S"))
    """The date of the experiment."""
    exp_name: str = field(
        default_factory=lambda: os.environ.get(
            "WANDB_RUN_ID", time.strftime("%Y-%m-%d-%H-%M-%S")
        )
    )
    """The name of the experiment."""
    project_name: str = "elastic-nerf"
    """The name of the project."""
    dataset_name: Literal["blender", "mipnerf360"] = "blender"
    """Which dataset to use."""
    dataset: Optional[NGPDatasetConfig] = None
    """The dataset configuration. Will be dynamically set based on the dataset_name."""
    scene: str = "lego"
    """Which scene to use."""
    model_path: Optional[Path] = None
    """The path of the pretrained model."""
    sampling_strategy: Literal[
        "uniform",
        "exp-optimal",
        "exp-optimal-reverse",
        "exp",
        "exp-reverse",
        "matroyshka",
        "matroyshka-reverse",
        "sequential",
    ] = "exp-reverse"
    """Sampling strategy for widths."""
    hidden_dim: int = 64
    """The hidden dimension of the MLP."""
    num_train_widths: int = 4
    """Number of widths to use for training."""
    num_widths_to_sample: int = 1
    """Number of widths to sample for each training step."""
    eval_elastic_widths: List[int] = field(default_factory=lambda: [64, 32, 16, 8])
    """Number of widths to use for evaluation."""
    max_steps: int = 20000
    """Maximum number of training steps."""
    num_eval_all_steps: int = 5000
    """Number of iterations after which to perform evaluation during training."""
    num_checkpoint_steps: int = 10000
    """Number of iterations after which to save a checkpoint."""
    num_log_steps: int = 500
    """Number of iterations after which to log training information."""
    radiance_field: NGPRadianceFieldConfig = field(
        default_factory=lambda: NGPRadianceFieldConfig()
    )
    """The configuration for the elastic MLP."""
    device: str = "cuda:0"
    """The device to use."""
    log_dir: Optional[Path] = None
    """The directory to store the logs."""
    wandb_dir: Optional[Path] = None
    """The directory containing wandb cache."""
    host_machine: str = os.environ["HOSTNAME"]
    """Name of the host machine"""
    enable_logging: bool = True
    """Whether to enable logging."""

    def __post_init__(self):
        if self.log_dir is None:
            self.log_dir = (
                Path(os.environ.get("RESULTS_CACHE_DIR", "./results"))
                / self.project_name
                / self.exp_name
            )

        if self.wandb_dir is None:
            self.wandb_dir = (
                Path(os.environ.get("WANDB_CACHE_DIR", "./wandb_cache"))
                / self.project_name
                / self.exp_name
            )

        if self.dataset_name == "blender":
            self.dataset = BlenderSyntheticDatasetConfig(scene=self.scene)
        elif self.dataset_name == "mipnerf360":
            self.dataset = MipNerf360DatasetConfig(scene=self.scene)
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")


