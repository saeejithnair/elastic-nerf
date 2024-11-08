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
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP
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
from mup import set_base_shapes
import mup
from elastic_nerf.nerfacc.radiance_fields.optimizers import ElasticMuAdam


@dataclass
class NGPPropTrainerConfig(NGPBaseTrainerConfig):
    """Configurations for training the model."""

    radiance_field: NGPRadianceFieldConfig = field(
        default_factory=lambda: NGPRadianceFieldConfig()
    )
    """The configuration for the elastic MLP."""
    density_field: NGPDensityFieldConfig = field(
        default_factory=lambda: NGPDensityFieldConfig()
    )
    """The configuration for the density field."""

    def __post_init__(self):
        super().__post_init__()

        if self.dataset_name == "blender":
            self.dataset = BlenderSyntheticDatasetPropConfig(scene=self.scene)
        elif self.dataset_name == "mipnerf360":
            self.dataset = MipNerf360DatasetPropConfig(scene=self.scene)
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        
        if self.optimizer_lr is not None:
            self.dataset.optimizer_lr = self.optimizer_lr

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
        if not self.use_elastic():
            assert self.config.num_train_widths == 1
            assert self.config.num_widths_to_sample == 1
            assert all(
                [width <= self.config.hidden_dim for width in self.eval_elastic_widths]
            )

    def use_elastic(self):
        return (
            self.config.radiance_field.use_elastic
            or self.config.density_field.use_elastic
        )

    def init_mup_proposal_nets(self, proposal_networks, aabb):
        base_width = 8
        initialized_proposal_networks = []
        for i, proposal_network in enumerate(proposal_networks):
            if self.config.use_mup:
                resolution = self.dataset.prop_network_resolutions[i]
                base_model = self.config.density_field.setup(
                    aabb=aabb,
                    unbounded=self.dataset.unbounded,
                    n_levels=5,
                    max_resolution=resolution,
                    base_mlp_width=base_width,
                ).to(self.device)
                delta_model = self.config.density_field.setup(
                    aabb=aabb,
                    unbounded=self.dataset.unbounded,
                    n_levels=5,
                    max_resolution=resolution,
                    base_mlp_width=base_width + 1,
                ).to(self.device)
                set_base_shapes(proposal_network, base_model, delta=delta_model)

            proposal_network = self.initialize_params(proposal_network)

            initialized_proposal_networks.append(proposal_network)

        return initialized_proposal_networks

    def initialize_params(self, model):
        for name, param in model.named_parameters():
            if self.config.use_mup and "weight" in name:
                if "output_layer" in name:
                    nonlinearity = "linear"
                    prefactor = 1.0
                else:
                    nonlinearity = "relu"
                    prefactor = np.sqrt(2.0)
                torch.nn.init.normal_(param, mean=0, std=1)
                # Apply spectral normalization to the param.
                fanin, fanout = mup.init._calculate_fan_in_and_fan_out(param)
                with torch.no_grad():
                    param.data = (
                        prefactor
                        * np.sqrt(fanout / fanin)
                        * param.data
                        / torch.linalg.matrix_norm(param, ord=2)
                    )
            elif "weight" in name:
                if "output_layer" in name:
                    nonlinearity = "linear"
                else:
                    nonlinearity = "relu"
                torch.nn.init.kaiming_normal_(param, nonlinearity=nonlinearity)
            else:
                print(f"Skipping initialization for {name}")
                continue
        return model

    def init_mup_radiance_field(self, radiance_field, aabb):
        if self.config.use_mup:
            base_width = 8
            base_model = self.config.radiance_field.setup(
                aabb=aabb,
                unbounded=self.dataset.unbounded,
                base_mlp_width=base_width,
                head_mlp_width=base_width,
            ).to(self.device)
            delta_model = self.config.radiance_field.setup(
                aabb=aabb,
                unbounded=self.dataset.unbounded,
                base_mlp_width=base_width + 1,
                head_mlp_width=base_width + 1,
            ).to(self.device)
            set_base_shapes(radiance_field, base_model, delta=delta_model)

        radiance_field = self.initialize_params(radiance_field)

        return radiance_field

    def initialize_model(self):
        """Initialize the radiance field and optimizer."""
        aabb = torch.tensor([*self.dataset.aabb_coeffs], device=self.device)
        proposal_networks = [
            self.config.density_field.setup(
                aabb=aabb.clone(),
                unbounded=self.dataset.unbounded,
                n_levels=5,
                max_resolution=resolution,
                base_mlp_width=self.config.hidden_dim,
            ).to(self.device)
            for resolution in self.dataset.prop_network_resolutions
        ]
        proposal_networks = self.init_mup_proposal_nets(proposal_networks, aabb.clone())
        optimizer_type = ElasticMuAdam if self.config.use_mup else torch.optim.Adam
        print(f"Using optimizer: {optimizer_type}")
        prop_optimizer = optimizer_type(
            itertools.chain(
                *[p.parameters() for p in proposal_networks],
            ),
            lr=self.compute_lr(self.dataset.optimizer_lr),
            eps=self.dataset.optimizer_eps,
            # weight_decay=self.dataset.weight_decay,
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
        if self.config.use_elastic_lr:
            raise NotImplementedError("Elastic LR not supported for proposal networks.")

        estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(self.device)

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        radiance_field: NGPRadianceField = self.config.radiance_field.setup(
            aabb=aabb.clone(),
            unbounded=self.dataset.unbounded,
            base_mlp_width=self.config.hidden_dim,
            head_mlp_width=self.config.hidden_dim,
        ).to(self.device)
        radiance_field = self.init_mup_radiance_field(radiance_field, aabb)
        # radiance_field_param_groups = radiance_field.get_param_groups()
        radiance_field_param_groups = radiance_field.parameters()
        optimizer = optimizer_type(
            radiance_field_param_groups,
            lr=self.compute_lr(self.dataset.optimizer_lr),
            eps=self.dataset.optimizer_eps,
            # weight_decay=self.dataset.weight_decay,
        )
        schedulers = [
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
        if self.config.use_elastic_lr:
            elastic_lr_schedules = []
            for param_group in radiance_field_param_groups:
                if "elastic_lr" in param_group and param_group["elastic_lr"] is True:
                    elastic_lr_schedules.append(self.get_elastic_lr())
                else:
                    elastic_lr_schedules.append(lambda step: 1.0)
            schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    elastic_lr_schedules,
                )
            )

        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers)
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
            self.models_to_watch[f"proposal_net_{i}"] = self.proposal_networks[i]

    def get_elastic_forward_kwargs(self, elastic_width, eval=False):
        if eval and self.config.fused_eval:
            return {}

        kwargs = {}
        if self.config.radiance_field.use_elastic:
            kwargs["active_neurons_radiance"] = elastic_width
        if self.config.density_field.use_elastic:
            kwargs["active_neurons_prop"] = elastic_width
        return kwargs

    def get_elastic_lr(self):
        # base_64 = base_8 * (1/fanin) = base_8 * (1/fanin_64)
        # width_64_lr = base_8_lr * (fanin_8/fanin_64)
        # width_8_lr = base_64_lr * (fanin_64/fanin_8)
        # elastic_width_lr = base_64_lr * (fanin_64 / elastic_width)
        def get_lr_multiplier(step):
            granularities_to_sample, granularity_loss_weights = self.sampling_schedule[
                step
            ]
            assert len(granularities_to_sample) == 1
            elastic_width = int(granularities_to_sample[0])
            # TODO: Don't hardcode, get base width dynamically
            return 8 / elastic_width

        return get_lr_multiplier

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
        train_data_idx: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:

        rays, pixels, render_bkgd = self.get_train_data(train_data_idx)
        kwargs = self.get_elastic_forward_kwargs(elastic_width)
        kwargs["proposal_requires_grad"] = proposal_requires_grad

        rgb, acc, depth, extras = self.render(rays, render_bkgd, **kwargs)

        estimator_loss = self.update_estimator_every_n_steps_granular(
            extras["trans"], requires_grad=proposal_requires_grad, loss_scaler=1024
        )
        if estimator_loss is not None:
            estimator_loss = estimator_loss * granularity_loss_weight

        loss, mse_loss, psnr = self.compute_losses(rgb, pixels)
        elastic_loss = 0
        metrics = {}
        if self.config.use_elastic_loss:
            for model in self.models_to_watch.values():
                for n, m in model.named_modules():
                    if isinstance(m, ElasticMLP):
                        spectral_loss = m.get_spectral_loss(elastic_width)
                        metrics[f"elastic_loss/{n}"] = float(spectral_loss)
                        elastic_loss += spectral_loss
            metrics.update(
                {"elastic_loss": float(elastic_loss), "render_loss": float(loss)}
            )
            loss += elastic_loss / 100
        loss = loss * granularity_loss_weight
        metrics.update(
            {
                "loss": float(loss),
                "mse_loss": float(mse_loss),
                "psnr": float(psnr),
                "max_depth": float(depth.max()),
                "num_rays": int(len(pixels)),
            }
        )

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
        granularities_to_sample, granularity_loss_weights = self.sampling_schedule[
            self.step
        ]

        # Do an update on the first step so that the estimator optimizer gets stepped
        # once before the estimator LR scheduler starts getting stepped (this is for PyTorch 1.1.0 and onward)
        proposal_requires_grad = self.proposal_requires_grad_fn(self.step) or (
            self.step == 0
        )
        loss_dict = {}
        metrics_dict = {}
        estimator_loss_dict = {}
        for i, elastic_width in enumerate(granularities_to_sample):
            if i > 0:
                torch.cuda.empty_cache()
            elastic_width = int(elastic_width)
            granularity_label = f"elastic_{elastic_width}"
            granularity_loss_weight = float(granularity_loss_weights[i])
            train_data_idx = self.get_train_data_idx(self.step, i)
            loss, estimator_loss, metrics = self.train_granular_step(
                int(elastic_width),
                granularity_loss_weight,
                proposal_requires_grad,
                train_data_idx=train_data_idx,
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
        print(f"Froze modules at step {self.step}")
