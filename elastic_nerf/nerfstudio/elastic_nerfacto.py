# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import elastic_nerf.utils.plotting_utils as pu
import numpy as np
import torch
import torch.nn as nn
from elastic_nerf.modules.elastic_nerfacto_field import ElasticNerfactoField
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@dataclass
class ElasticNerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ElasticNerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    train_granularities: int = 4
    """Number of granularities to use for training the elastic model."""
    eval_granularities: int = 4
    """Number of granularities to test model elasticity on."""
    num_granularities_to_sample: int = 4
    """Number of elastic scales to sample during the forward pass."""
    granularities_sample_prob: Literal[
        "uniform",
        "exp",
        "exp-reverse",
        "exp-optimal",
        "exp-optimal-reverse",
        "matroyshka",
        "matroyshka-reverse",
    ] = "uniform"
    elasticity_method: Literal["elastic_loss", "kinder"] = "kinder"
    """Which method to use for elasticity."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    num_hidden_layers: int = 1
    """Number of hidden layers in the base mlp."""
    use_granular_norm: bool = False
    """Whether to use granular norm after elastic MLP hidden layers."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 128,
                "use_linear": False,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 256,
                "use_linear": False,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""


class ElasticNerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: ElasticNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.train_granularities = torch.tensor(
            [2**i for i in range(self.config.train_granularities)]
        )
        self.eval_granularities = torch.tensor(
            [2**i for i in range(self.config.eval_granularities)]
        )

        # The sampling weights determine the probability of a granularity
        # being selected for a forward pass.
        self.granularity_sampling_weights: torch.Tensor = (
            self.get_granularity_sampling_weights(len(self.train_granularities))
        )

        # To save time during the frequent eval metrics dict computations,
        # we only use the granularities associated with that forward pass.
        # However, this field gets set to True when we evaluate all images.
        self.evaluate_all_granularities = False

        # Keep track of how many samples we've seen for each granularity.
        # The keys for this should be the eval granularities so that in our
        # logs, we can see for certain that the non train granularities have
        # 0 samples.
        self.granularity_sample_counts = {
            int(granularity): 0 for granularity in self.eval_granularities
        }

        assert self.config.hidden_dim % self.train_granularities[-1] == 0

        # Fields
        self.field = ElasticNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
            use_granular_norm=self.config.use_granular_norm,
            num_granularities=self.config.train_granularities,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = (
            None  # None is for piecewise as default (see ProposalNetworkSampler)
        )
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        field_params = {
            "params": list(self.field.parameters()),
        }

        if self.config.elasticity_method == "kinder":
            field_params.update(
                {
                    "weight_decay": list(self.field.elastic_mlp.parameters()),
                }
            )
        param_groups["fields"] = [field_params]
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs_granular(
        self,
        ray_bundle: RayBundle,
        ray_samples: RaySamples,
        weights_list: List,
        ray_samples_list: List,
        active_neurons,
    ):
        field_outputs = self.field.forward(
            ray_samples,
            compute_normals=self.config.predict_normals,
            active_neurons=active_neurons,
        )
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(
                field_outputs, ray_samples
            )

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(
            weights=weights, ray_samples=ray_samples
        )
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(
                normals=field_outputs[FieldHeadNames.NORMALS], weights=weights
            )
            pred_normals = self.renderer_normals(
                field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights
            )
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS],
                ray_bundle.directions,
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        elastic_outputs = {}

        # If training in kinder mode, use the whole model (granularity=1x) for the
        # forward pass. Otherwise, randomly sample one granularity for the forward pass.
        granularities = self.train_granularities
        if self.training:
            if self.config.elasticity_method == "kinder":
                # Assuming the largest granularity is the first in the list
                granularities = [self.train_granularities[0]]
            else:
                # Randomly sample a specified number of unique granularities
                num_granularities_to_sample = min(
                    len(self.train_granularities),
                    self.config.num_granularities_to_sample,
                )
                granularity_indices = torch.multinomial(
                    self.granularity_sampling_weights,
                    num_granularities_to_sample,
                    replacement=False,
                )
                granularities = self.train_granularities[granularity_indices]
        elif self.evaluate_all_granularities:
            granularities = self.eval_granularities

        for granularity in granularities:
            granularity = int(granularity)
            assert self.config.hidden_dim % granularity == 0
            active_neurons = self.config.hidden_dim // granularity
            granularity_label = self.get_granularity_label(granularity)
            elastic_outputs[granularity_label] = self.get_outputs_granular(
                ray_bundle, ray_samples, weights_list, ray_samples_list, active_neurons
            )
            self.granularity_sample_counts[granularity] += 1

        return elastic_outputs

    def get_granularity_sampling_weights(self, num_granularities: int) -> torch.Tensor:
        """Generates normalized weights for sampling granularities."""
        if self.config.granularities_sample_prob == "exp-optimal":
            weights = torch.tensor(
                [math.exp(0.1 * i) for i in range(num_granularities)]
            )
        elif self.config.granularities_sample_prob == "exp-optimal-reverse":
            weights = torch.tensor(
                [math.exp(0.1 * i) for i in range(num_granularities)]
            ).flip(
                0,
            )
        elif self.config.granularities_sample_prob == "exp":
            weights = torch.tensor([math.exp(i) for i in range(num_granularities)])
        elif self.config.granularities_sample_prob == "exp-reverse":
            weights = torch.tensor(
                [math.exp(i) for i in range(num_granularities)]
            ).flip(
                0,
            )
        elif self.config.granularities_sample_prob == "matroyshka":
            weights = torch.tensor([math.sqrt(2**i) for i in range(num_granularities)])
        elif self.config.granularities_sample_prob == "matroyshka-reverse":
            weights = torch.tensor(
                [math.sqrt(2**i) for i in range(num_granularities)]
            ).flip(
                0,
            )
        elif self.config.granularities_sample_prob == "uniform":
            weights = torch.ones(num_granularities)
        else:
            raise NotImplementedError

        return weights / weights.sum()

    def compute_weight_norms_elastic_layer(self, elastic_module: nn.Module):
        """Compute the weight norms for the elastic layer."""
        weight_norms = {}
        for name, parameter in elastic_module.named_parameters():
            if "weight" in name:
                weight_norms[name] = torch.norm(parameter).item()
        return weight_norms

    def get_metrics_dict_granular(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return metrics_dict

    def get_granularity_label(self, granularity: int) -> str:
        return f"elastic_{granularity}"

    def get_metrics_dict(
        self, elastic_outputs: Dict[str, Dict], batch
    ) -> Dict[str, Dict]:
        elastic_metrics_dict: Dict[str, Dict] = {}
        for granularity_label, elastic_output in elastic_outputs.items():
            elastic_metrics_dict[granularity_label] = self.get_metrics_dict_granular(
                elastic_output, batch
            )

        return elastic_metrics_dict

    def get_elastic_params_and_heatmaps(self) -> Tuple[Dict, Dict]:
        elastic_weight_norms_dict: Dict[str, Dict] = {}
        for granularity in self.eval_granularities:
            granularity = int(granularity)
            granularity_label = self.get_granularity_label(granularity)
            elastic_weight_norms_dict[granularity_label] = (
                self.field.elastic_mlp.compute_active_weight_norm(
                    int(self.config.hidden_dim // granularity)
                )
            )

        elastic_weight_heatmaps = pu.generate_mlp_heatmap(
            self.field.elastic_mlp, model_name="field.elastic_mlp"
        )

        return elastic_weight_norms_dict, elastic_weight_heatmaps

    def get_loss_dict_granular(
        self, outputs, image, metrics_dict: Optional[Dict] = None
    ):
        loss_dict = {}
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict["interlevel_loss"] = (
                self.config.interlevel_loss_mult
                * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = (
                self.config.distortion_loss_mult * metrics_dict["distortion"]
            )
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = (
                    self.config.orientation_loss_mult
                    * torch.mean(outputs["rendered_orientation_loss"])
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = (
                    self.config.pred_normal_loss_mult
                    * torch.mean(outputs["rendered_pred_normal_loss"])
                )
        return loss_dict

    def get_loss_dict(
        self,
        elastic_outputs,
        batch,
        elastic_metrics_dict: Dict[str, Dict],
    ):
        elastic_loss_dict = {}
        image = batch["image"].to(self.device)
        num_granularities_sampled = len(elastic_outputs)
        granularity_loss_weight = 1 / num_granularities_sampled
        for granularity_label in elastic_outputs.keys():
            loss_dict = self.get_loss_dict_granular(
                elastic_outputs[granularity_label],
                image,
                elastic_metrics_dict[granularity_label],
            )
            for loss_key in loss_dict:
                metric_key = f"{granularity_label}/{loss_key}"
                # Multiply each loss by the granularity weight
                elastic_loss_dict[metric_key] = (
                    loss_dict[loss_key] * granularity_loss_weight
                )
        return elastic_loss_dict

    def get_image_metrics_and_images_granular(
        self, outputs: Dict[str, torch.Tensor], gt_rgb: torch.Tensor
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        predicted_rgb = outputs[
            "rgb"
        ]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def get_image_metrics_and_images(
        self, elastic_outputs: Dict[str, Dict], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)

        elastic_metrics_dict = {}
        elastic_images_dict = {}
        for granularity_label in elastic_outputs.keys():
            metrics_dict, images_dict = self.get_image_metrics_and_images_granular(
                elastic_outputs[granularity_label], gt_rgb
            )
            elastic_metrics_dict[granularity_label] = metrics_dict
            elastic_images_dict[granularity_label] = images_dict

        return elastic_metrics_dict, elastic_images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, evaluate_all_granularities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """

        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        output_dicts = {}
        # outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
                start_idx, end_idx
            )
            self.evaluate_all_granularities = evaluate_all_granularities
            elastic_outputs = self.forward(ray_bundle=ray_bundle)
            self.evaluate_all_granularities = False

            for granularity_label in elastic_outputs:
                if granularity_label not in output_dicts:
                    output_dicts[granularity_label] = {}
                outputs = elastic_outputs[granularity_label]
                for output_name, output in outputs.items():  # type: ignore
                    if output_name not in output_dicts[granularity_label]:
                        output_dicts[granularity_label][output_name] = []
                    if not torch.is_tensor(output):
                        # TODO: handle lists of tensors as well
                        continue
                    output_dicts[granularity_label][output_name].append(output)
        elastic_outputs = {}
        for granularity_label in output_dicts:
            elastic_outputs[granularity_label] = {}
            for output_name, outputs_list in output_dicts[granularity_label].items():
                elastic_outputs[granularity_label][output_name] = torch.cat(
                    outputs_list
                ).view(image_height, image_width, -1)

        return elastic_outputs
