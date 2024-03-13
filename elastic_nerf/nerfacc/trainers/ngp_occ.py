from elastic_nerf.nerfacc.trainers.base import NGPBaseTrainerConfig

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
from elastic_nerf.nerfacc.trainers.base import NGPBaseTrainerConfig, NGPTrainer
from elastic_nerf.nerfacc.configs.datasets.base import (
    NGPBaseDatasetConfig,
    NGPOccDatasetConfig,
    NGPPropDatasetConfig,
)
from elastic_nerf.nerfacc.configs.datasets.blender import (
    BlenderSyntheticDatasetOccConfig,
)
from elastic_nerf.nerfacc.configs.datasets.mipnerf360 import (
    MipNerf360DatasetOccConfig,
)

set_random_seed(42)


class NGPOccTrainerConfig(NGPBaseTrainerConfig):
    """Configurations for training the model."""

    def __post_init__(self):
        super().__post_init__()

        if self.dataset_name == "blender":
            self.dataset = BlenderSyntheticDatasetOccConfig(scene=self.scene)
        elif self.dataset_name == "mipnerf360":
            self.dataset = MipNerf360DatasetOccConfig(scene=self.scene)
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

    def setup(self, **kwargs) -> "NGPOccTrainer":
        """Returns the instantiated object using the config."""
        return NGPOccTrainer(self, **kwargs)


class NGPOccTrainer(NGPTrainer):
    radiance_field: NGPRadianceField
    estimator: OccGridEstimator
    config: NGPOccTrainerConfig
    dataset: NGPOccDatasetConfig

    def __init__(self, config: NGPOccTrainerConfig):
        super().__init__(config)
        self.setup()

    def get_aabb(self, estimator):
        return estimator.aabbs[-1]

    def validate_elastic_compatibility(self):
        if not self.config.radiance_field.use_elastic:
            assert self.config.num_train_widths == 1
            assert self.config.num_widths_to_sample == 1
            assert self.config.eval_elastic_widths == [self.config.hidden_dim]

    def initialize_model(self):
        """Initialize the radiance field and optimizer."""
        aabb = torch.tensor([*self.dataset.aabb_coeffs], device=self.device)
        estimator = OccGridEstimator(
            roi_aabb=aabb,
            resolution=self.dataset.grid_resolution,
            levels=self.dataset.grid_nlvl,
        ).to(self.device)

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        radiance_field: NGPRadianceField = self.config.radiance_field.setup(
            aabb=self.get_aabb(estimator)
        ).to(self.device)
        optimizer = torch.optim.Adam(
            radiance_field.parameters(),
            lr=self.dataset.optimizer_lr,
            eps=self.dataset.optimizer_eps,
            weight_decay=self.dataset.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.dataset.scheduler_start_factor,
                    total_iters=self.dataset.scheduler_total_iters,
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        self.config.max_steps // 2,
                        self.config.max_steps * 3 // 4,
                        self.config.max_steps * 9 // 10,
                    ],
                    gamma=self.dataset.scheduler_gamma,
                ),
            ]
        )

        self.radiance_field = radiance_field
        self.estimator = estimator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler

        self.models_to_watch = {
            "radiance_field": self.radiance_field,
        }

    def get_elastic_forward_kwargs(self, elastic_width):
        if not self.config.radiance_field.use_elastic:
            return {}

        return {"active_neurons": int(elastic_width)}

    def render(self, rays, render_bkgd, **kwargs):
        return render_image_with_occgrid(
            self.radiance_field,
            self.estimator,
            rays,
            # rendering options
            near_plane=self.dataset.near_plane,
            render_step_size=self.dataset.render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=self.dataset.cone_angle,
            alpha_thre=self.dataset.alpha_thre,
            # test options
            test_chunk_size=self.dataset.test_chunk_size,
            **kwargs,
        )

    def train_granular_step(
        self, elastic_width: int, granularity_loss_weight: float
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Perform a single training step on a single width."""
        if self.dataset.target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = self.granularity_target_num_rays[elastic_width]
            num_rays = (
                num_rays
                if (
                    num_rays > 0
                    and (
                        self.get_elastic_width_sample_counts(elastic_width)
                        > self.dataset.num_dynamic_batch_warmup_steps
                    )
                )
                else (self.dataset.init_batch_size // self.config.num_widths_to_sample)
            )
            self.train_dataset.update_num_rays(num_rays)

        rays, pixels, render_bkgd = self.get_train_data()
        kwargs = self.get_elastic_forward_kwargs(elastic_width)

        rgb, _, depth, n_rendering_samples = self.render(rays, render_bkgd, **kwargs)

        if n_rendering_samples == 0:
            metrics = {
                "n_rendering_samples": int(n_rendering_samples),
                "num_rays": int(len(pixels)),
            }
            return None, metrics

        loss, mse_loss, psnr = self.compute_losses(rgb, pixels)
        loss = loss * granularity_loss_weight
        metrics = {
            "loss": float(loss),
            "mse_loss": float(mse_loss),
            "psnr": float(psnr),
            "max_depth": float(depth.max()),
            "num_rays": int(len(pixels)),
            "n_rendering_samples": n_rendering_samples,
        }

        num_rays = int(
            len(pixels)
            * (
                (
                    self.dataset.target_sample_batch_size
                    / self.config.num_widths_to_sample
                )
                / n_rendering_samples
            )
        )
        self.granularity_target_num_rays[int(elastic_width)] = num_rays

        return loss, metrics

    def train_step(
        self,
    ) -> Tuple[Dict[str, Union[float, int]], bool]:
        """Perform a single training step."""
        self.radiance_field.train()
        self.estimator.train()
        granularities_to_sample, granularity_loss_weight = self.sample_granularities()

        def occ_eval_fn(x):
            if not self.config.radiance_field.use_elastic:
                density = self.radiance_field.query_density(x)
                return density * self.dataset.render_step_size

            density_sum = None
            count = 0
            # Compute the mean density across all elastic widths.
            for elastic_width in granularities_to_sample:
                density = self.radiance_field.query_density(
                    x, active_neurons=int(elastic_width)
                )
                if density_sum is None:
                    density_sum = density
                else:
                    density_sum = torch.add(density_sum, density)
                count += 1

            mean_density = density_sum / count
            return mean_density * self.dataset.render_step_size

        self.estimator.update_every_n_steps(
            step=self.step, occ_eval_fn=occ_eval_fn, occ_thre=1e-2
        )

        loss_dict = {}
        metrics_dict = {}
        for i, elastic_width in enumerate(granularities_to_sample):
            granularity_label = f"elastic_{elastic_width}"
            torch.cuda.empty_cache()
            loss, metrics = self.train_granular_step(
                int(elastic_width), granularity_loss_weight
            )
            metrics_dict[granularity_label] = metrics

            if loss is None:
                # No points were sampled for this width.
                gradient_updated = False
                for width in granularities_to_sample:
                    self.num_updates_skipped[int(width)] += 1

                del loss_dict
                return metrics_dict, gradient_updated

            loss_dict[granularity_label] = loss

        loss = functools.reduce(torch.add, loss_dict.values())
        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()
        gradient_updated = True
        self.update_elastic_width_sample_counts(granularities_to_sample)

        return metrics_dict, gradient_updated

    @staticmethod
    def load_trainer(
        config: str,
        log_dir: Path,
        wandb_dir: Path,
        ckpt_path: Optional[Path] = None,
    ) -> "NGPOccTrainer":
        # Load model from config
        trainer_config: NGPOccTrainerConfig = from_yaml(NGPOccTrainerConfig, config)
        trainer_config.log_dir = log_dir
        trainer_config.wandb_dir = wandb_dir
        trainer_config.enable_logging = False
        if ckpt_path is not None:
            trainer_config.model_path = ckpt_path
        trainer = trainer_config.setup()

        return trainer

    def load_elastic_width(self, elastic_width: int):
        """Load the model with the specified elastic width."""
        if not hasattr(self, "full_width_radiance_field"):
            self.full_width_radiance_field = copy.deepcopy(
                self.radiance_field.to(torch.device("cpu"))
            )

        new_width_elastic_net = (
            self.full_width_radiance_field.mlp_base.elastic_mlp.get_sliced_net(
                elastic_width
            )
        )
        self.radiance_field.mlp_base.elastic_mlp = new_width_elastic_net

        self.radiance_field.to(self.device)