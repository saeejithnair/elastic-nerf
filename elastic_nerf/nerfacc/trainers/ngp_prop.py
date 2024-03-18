"""
Copyright (c) 2023 Saeejith Nair, University of Waterloo.
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import itertools
import math
import os
import subprocess
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
import tyro
from gonas.configs.base_config import InstantiateConfig, PrintableConfig
from lpips import LPIPS
from nerfacc.estimators.prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.scripts.downloads.download_data import BlenderDownload
from torchmetrics.functional import structural_similarity_index_measure

import wandb
from elastic_nerf.nerfacc.datasets.nerf_360_v2 import SubjectLoader as MipNerf360Loader
from elastic_nerf.nerfacc.datasets.nerf_synthetic import (
    SubjectLoader as BlenderSyntheticLoader,
)
from elastic_nerf.nerfacc.radiance_fields.ngp import (
    NGPDensityField,
    NGPDensityFieldConfig,
    NGPRadianceField,
    NGPRadianceFieldConfig,
)
from elastic_nerf.nerfacc.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_propnet,
)
from elastic_nerf.nerfacc.configs.datasets.base import NGPPropDatasetConfig

from elastic_nerf.nerfacc.configs.datasets.blender import (
    BlenderSyntheticDatasetPropConfig,
)
from elastic_nerf.nerfacc.configs.datasets.mipnerf360 import (
    MipNerf360DatasetPropConfig,
)
from elastic_nerf.nerfacc.trainers.base import NGPBaseTrainerConfig, NGPTrainer


@dataclass
class NGPPropTrainerConfig(NGPBaseTrainerConfig):
    """Configurations for training the model."""

    density_field: NGPDensityFieldConfig = field(
        default_factory=lambda: NGPDensityFieldConfig()
    )

    def __post_init__(self):
        super().__post_init__()

        if self.dataset_name == "blender":
            self.dataset = BlenderSyntheticDatasetPropConfig(scene=self.scene)
        elif self.dataset_name == "mipnerf360":
            self.dataset = MipNerf360DatasetPropConfig(scene=self.scene)
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

    def setup(self, **kwargs) -> "NGPPropTrainer":
        """Returns the instantiated object using the config."""
        return NGPPropTrainer(self, **kwargs)


class NGPPropTrainer(NGPTrainer):
    radiance_field: NGPRadianceField
    estimator: PropNetEstimator
    config: NGPPropTrainerConfig
    dataset: NGPPropDatasetConfig
    proposal_networks: List[NGPDensityField]
    prop_optimizer: torch.optim.Optimizer
    prop_scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(self, config: NGPPropTrainerConfig):
        super().__init__(config)
        self.setup()

    def setup_datasets(self):
        return super().setup_datasets(num_rays_scaler=self.config.num_widths_to_sample)

    def validate_elastic_compatibility(self):
        if (
            not self.config.radiance_field.use_elastic
            and not self.config.density_field.use_elastic
        ):
            assert self.config.num_train_widths == 1
            assert self.config.num_widths_to_sample == 1
            assert all(
                [width <= self.config.hidden_dim for width in self.eval_elastic_widths]
            )

    def initialize_model(self):
        """Initialize the radiance field and optimizer."""
        aabb = torch.tensor([*self.dataset.aabb_coeffs], device=self.device)
        proposal_networks = [
            self.config.density_field.setup(
                aabb=aabb,
                unbounded=self.dataset.unbounded,
                n_levels=5,
                max_resolution=resolution,
                base_mlp_width=self.config.hidden_dim,
            ).to(self.device)
            for resolution in self.dataset.prop_network_resolutions
        ]
        prop_optimizer = torch.optim.Adam(
            itertools.chain(
                *[p.parameters() for p in proposal_networks],
            ),
            lr=self.dataset.optimizer_lr,
            eps=self.dataset.optimizer_eps,
            weight_decay=self.dataset.weight_decay,
        )
        prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    prop_optimizer,
                    start_factor=self.dataset.scheduler_start_factor,
                    total_iters=self.dataset.scheduler_total_iters,
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    prop_optimizer,
                    milestones=[
                        self.config.max_steps // 2,
                        self.config.max_steps * 3 // 4,
                        self.config.max_steps * 9 // 10,
                    ],
                    gamma=self.dataset.scheduler_gamma,
                ),
            ]
        )
        estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(self.device)

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        radiance_field: NGPRadianceField = self.config.radiance_field.setup(
            aabb=aabb,
            unbounded=self.dataset.unbounded,
            base_mlp_width=self.config.hidden_dim,
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

        self.proposal_networks = proposal_networks
        self.prop_optimizer = prop_optimizer
        self.prop_scheduler = prop_scheduler
        self.radiance_field = radiance_field
        self.estimator = estimator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()

        self.models_to_watch = {
            "radiance_field": self.radiance_field,
        }
        for i, prop_net in enumerate(self.proposal_networks):
            self.models_to_watch[f"proposal_net_{i}"] = prop_net

    def get_elastic_forward_kwargs(self, elastic_width, eval=False):
        if eval and self.config.fused_eval:
            return {}

        kwargs = {}
        if self.config.radiance_field.use_elastic:
            kwargs["active_neurons_radiance"] = elastic_width
        if self.config.density_field.use_elastic:
            kwargs["active_neurons_prop"] = elastic_width
        return kwargs

    def render(
        self,
        rays,
        render_bkgd,
        radiance_field=None,
        estimator=None,
        proposal_networks=None,
        **kwargs,
    ):
        # Returns rgb, acc, depth, extras
        return render_image_with_propnet(
            self.radiance_field if radiance_field is None else radiance_field,
            self.proposal_networks if proposal_networks is None else proposal_networks,
            self.estimator if estimator is None else estimator,
            rays,
            # rendering options
            num_samples=self.dataset.num_samples,
            num_samples_per_prop=self.dataset.num_samples_per_prop,
            near_plane=self.dataset.near_plane,
            far_plane=self.dataset.far_plane,
            sampling_type=self.dataset.sampling_type,
            opaque_bkgd=self.dataset.opaque_bkgd,
            render_bkgd=render_bkgd,
            # test options
            test_chunk_size=self.dataset.test_chunk_size,
            **kwargs,
        )

    def train_granular_step(
        self,
        elastic_width: int,
        granularity_loss_weight: float,
        proposal_requires_grad: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:

        rays, pixels, render_bkgd = self.get_train_data()
        kwargs = self.get_elastic_forward_kwargs(elastic_width)
        kwargs["proposal_requires_grad"] = proposal_requires_grad

        rgb, acc, depth, extras = self.render(rays, render_bkgd, **kwargs)

        estimator_loss = self.update_estimator_every_n_steps_granular(
            extras["trans"], requires_grad=proposal_requires_grad, loss_scaler=1024
        )
        if estimator_loss is not None:
            estimator_loss = estimator_loss * granularity_loss_weight

        loss, mse_loss, psnr = self.compute_losses(rgb, pixels)
        loss = loss * granularity_loss_weight
        metrics = {
            "loss": float(loss),
            "mse_loss": float(mse_loss),
            "psnr": float(psnr),
            "max_depth": float(depth.max()),
            "num_rays": int(len(pixels)),
        }

        return loss, estimator_loss, metrics

    @torch.enable_grad()
    def update_estimator_every_n_steps_granular(
        self, trans: torch.Tensor, requires_grad: bool = False, loss_scaler: float = 1.0
    ) -> Optional[torch.Tensor]:
        if not requires_grad:
            return None

        assert len(self.estimator.prop_cache) > 0
        loss = self.estimator.compute_loss(trans, loss_scaler)
        return loss

    @torch.enable_grad()
    def update_estimator_every_n_steps(self, loss_dict: Dict[str, torch.Tensor]):
        assert self.estimator.optimizer is not None, "No optimizer is provided."
        if len(loss_dict) > 0:
            loss = functools.reduce(torch.add, loss_dict.values())
            self.estimator.optimizer.zero_grad()
            loss.backward()
            self.estimator.optimizer.step()

        if self.estimator.scheduler is not None:
            self.estimator.scheduler.step()

    def train_step(
        self,
    ) -> Tuple[Dict[str, Union[float, int]], bool]:
        """Perform a single training step."""
        self.set_mode(train=True)
        granularities_to_sample, granularity_loss_weight = self.sampling_schedule[
            self.step
        ]

        proposal_requires_grad = self.proposal_requires_grad_fn(self.step)

        loss_dict = {}
        metrics_dict = {}
        estimator_loss_dict = {}
        for i, elastic_width in enumerate(granularities_to_sample):
            if i > 0:
                torch.cuda.empty_cache()
            elastic_width = int(elastic_width)
            granularity_label = f"elastic_{elastic_width}"
            loss, estimator_loss, metrics = self.train_granular_step(
                int(elastic_width), granularity_loss_weight, proposal_requires_grad
            )
            metrics_dict[granularity_label] = metrics
            loss_dict[granularity_label] = loss
            if estimator_loss is None:
                assert not proposal_requires_grad
            else:
                estimator_loss_dict[granularity_label] = estimator_loss

        # The estimator will be updated if the estimator loss dict
        # is not empty (i.e. proposal_requires_grad is True).
        self.update_estimator_every_n_steps(estimator_loss_dict)

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
    ) -> "NGPPropTrainer":
        # Load model from config
        return NGPTrainer.load_trainer(
            config,
            log_dir,
            wandb_dir,
            config_type=NGPPropTrainerConfig,
            ckpt_path=ckpt_path,
        )

    def freeze(self):
        """Saves a deepcopy of models to be used for evaluation."""
        self.frozen = {}
        self.frozen["estimator"] = copy.deepcopy(self.estimator)
        self.frozen["radiance_field"] = copy.deepcopy(self.radiance_field)
        self.frozen["proposal_networks"] = copy.deepcopy(self.proposal_networks)
