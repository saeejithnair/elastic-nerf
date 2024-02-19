"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from dataclasses import dataclass, field, fields
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gonas.configs.base_config import FlexibleInstantiateConfig, InstantiateConfig
from nerfstudio import data
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.cuda.amp.autocast_mode import autocast

from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP, ElasticMLPConfig

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    #  ord: Union[float, int] = float("inf"),
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x


@dataclass
class NGPRadianceFieldConfig(InstantiateConfig):
    """Instance-NGP Radiance Field Config"""

    _target: Type = field(default_factory=lambda: NGPRadianceField)

    use_elastic: bool = False
    """Whether to use an elastic MLP."""
    base: ElasticMLPConfig = field(
        default_factory=lambda: ElasticMLPConfig(output_activation=None)
    )
    """Configuration if using an elastic MLP."""

    @staticmethod
    def from_dict(data: dict) -> "NGPRadianceFieldConfig":
        cls = NGPRadianceFieldConfig()
        """Create a config from a dictionary."""
        for field_info in fields(cls):
            field_name = field_info.name
            if field_name.startswith("_"):  # Skipping private or protected fields
                continue
            if field_name in data:
                field_value = data[field_name]
                field_type = field_info.type

                if hasattr(field_type, "from_dict") and isinstance(field_value, dict):
                    # If the field type has a 'from_dict' method and the corresponding value is a dict,
                    # call 'from_dict' recursively
                    setattr(cls, field_name, field_type.from_dict(field_value))
                elif isinstance(field_value, dict) and not hasattr(
                    field_type, "from_dict"
                ):
                    # If the field value is a dict and the field type does not have a 'from_dict' method,
                    # assume it is a type where the keys are the field names and the values are the field values
                    setattr(cls, field_name, field_type(**field_value))
                else:
                    # For simple types, just assign the value directly
                    setattr(cls, field_name, field_value)
        return cls


class ElasticMLPWithInputEncoding(torch.nn.Module):
    """Elastic MLP with input encoding"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoding_config: dict,
        elastic_mlp: ElasticMLPConfig,
        net_depth: int = 1,
        net_width: int = 64,
    ) -> None:
        super().__init__()
        self.encoding = tcnn.Encoding(
            n_input_dims=input_dim, encoding_config=encoding_config
        )
        self.elastic_mlp: ElasticMLP = elastic_mlp.setup(
            input_dim=self.encoding.n_output_dims,
            output_dim=output_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=None,
            output_enabled=True,
        )

    def forward(
        self, x: torch.Tensor, active_neurons: Optional[int] = None
    ) -> torch.Tensor:
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons

        with autocast():
            x = self.encoding(x)
            return self.elastic_mlp(x, **kwargs)


class NGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        config: NGPRadianceFieldConfig,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.config = config
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }
        if self.config.use_elastic:
            self.mlp_base = ElasticMLPWithInputEncoding(
                input_dim=num_dim,
                output_dim=1 + self.geo_feat_dim,
                encoding_config=encoding_config,
                elastic_mlp=config.base,
            )
        else:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=num_dim,
                n_output_dims=1 + self.geo_feat_dim,
                encoding_config=encoding_config,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (self.direction_encoding.n_output_dims if self.use_viewdirs else 0)
                    + self.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def query_density(
        self, x, return_feat: bool = False, active_neurons: Optional[int] = None
    ):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons

        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim), **kwargs)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation) * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = self.mlp_head(h).reshape(list(embedding.shape[:-1]) + [3]).to(embedding)
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
        active_neurons: Optional[int] = None,
    ):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons

        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(
                positions, return_feat=True, **kwargs
            )
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore


@dataclass
class NGPDensityFieldConfig(InstantiateConfig):
    """Instance-NGP Radiance Field Config"""

    _target: Type = field(default_factory=lambda: NGPDensityField)

    use_elastic: bool = False
    """Whether to use an elastic MLP."""
    base: ElasticMLPConfig = field(
        default_factory=lambda: ElasticMLPConfig(output_activation=None)
    )
    """Configuration if using an elastic MLP."""


class NGPDensityField(torch.nn.Module):
    """Instance-NGP Density Field used for resampling"""

    def __init__(
        self,
        config: NGPDensityFieldConfig,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 128,
        n_levels: int = 5,
        log2_hashmap_size: int = 17,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.config = config
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }
        if self.config.use_elastic:
            self.mlp_base = ElasticMLPWithInputEncoding(
                input_dim=num_dim,
                output_dim=1,
                encoding_config=encoding_config,
                elastic_mlp=config.base,
            )
        else:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=num_dim,
                n_output_dims=1,
                encoding_config=encoding_config,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )

    def forward(self, positions: torch.Tensor, active_neurons: Optional[int] = None):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons

        if self.unbounded:
            positions = contract_to_unisphere(positions, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        density_before_activation = (
            self.mlp_base(positions.view(-1, self.num_dim), **kwargs)
            .view(list(positions.shape[:-1]) + [1])
            .to(positions)
        )
        density = (
            self.density_activation(density_before_activation) * selector[..., None]
        )
        return density
