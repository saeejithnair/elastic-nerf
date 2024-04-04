"""
Copyright (c) 2023 Saeejith Nair, University of Waterloo.
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, Literal, Optional, Type, Union

from cycler import V
import torch
import torch.nn as nn
import torch.nn.functional as F
from gonas.configs.base_config import FlexibleInstantiateConfig, InstantiateConfig
from typing_extensions import NotRequired, TypedDict
import tinycudann as tcnn
from elastic_nerf.modules.elastic_mlp import GranularNorm, GranularNormConfig
from mup import MuReadout, set_base_shapes
import mup


def get_field_granular_state_dict(
    radiance_field: "VanillaNeRFRadianceField", active_neurons
):
    """Returns a state dict of the active neurons in the field."""
    field_state_dict = radiance_field.state_dict()
    elastic_mlp_state_dict = radiance_field.mlp.base.state_dict(active_neurons)
    for key in elastic_mlp_state_dict:
        field_state_dict[f"mlp.base.{key}"] = elastic_mlp_state_dict[key]
    return field_state_dict


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: Optional[int] = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: Optional[int] = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


def slice_weights_and_biases_from_linear_layer(
    layer,
    active_neurons: int,
    input_dim: int,
    first_layer: bool = False,
    skip_layer: bool = False,
    output_layer: bool = False,
):
    weight = layer.weight
    bias = layer.bias
    active_neuron_indices = torch.arange(active_neurons, device=weight.device)
    if first_layer:
        W = weight[active_neuron_indices, :]
        b = bias[active_neuron_indices] if bias is not None else None
    elif skip_layer:
        W = weight[active_neuron_indices, : input_dim + active_neurons]
        b = bias[active_neuron_indices] if bias is not None else None
    elif output_layer:
        W = weight[:, active_neuron_indices]
        b = bias if bias is not None else None
    else:
        W = weight[:active_neurons, active_neuron_indices]
        b = bias[active_neuron_indices] if bias is not None else None
    return W, b


def linear_layer_muladd_act(x, W, b, activation):
    x = torch.matmul(x, W.t())
    if b is not None:
        x += b

    if activation is not None:
        x = activation(x)

    return x


def linear_layer_muladd_granularnorm_act(x, W, b, granular_norm, activation):
    x = torch.matmul(x, W.t())
    if b is not None:
        x += b
    x = granular_norm(x)

    if activation is not None:
        x = activation(x)

    return x


NerfMLPHiddenLayers = TypedDict(
    "NerfMLPHiddenLayers",
    {
        "hidden_layers.0": NotRequired[int],
        "hidden_layers.1": NotRequired[int],
        "hidden_layers.2": NotRequired[int],
        "hidden_layers.3": NotRequired[int],
        "hidden_layers.4": NotRequired[int],
        "hidden_layers.5": NotRequired[int],
        "hidden_layers.6": NotRequired[int],
        "hidden_layers.7": NotRequired[int],
    },
    total=False,
)


@dataclass
class ElasticMLPConfig(FlexibleInstantiateConfig):
    _target: Type = field(default_factory=lambda: ElasticMLP)

    hidden_init: Callable = nn.init.xavier_uniform_
    """Initialization function to apply to hidden layers."""
    hidden_activation: Callable = nn.ReLU()
    """Activation to apply after each hidden layer."""
    output_init: Optional[Callable] = nn.init.xavier_uniform_
    """Initialization function to apply to output layer."""
    output_activation: Optional[Callable] = nn.Identity()
    """Activation to apply after output layer."""
    bias_enabled: bool = True
    """Whether to include bias in linear layers."""
    bias_init: Callable = nn.init.zeros_
    """Initialization function to apply to bias."""
    # granular_norm: Optional[GranularNormConfig] = field(
    #     default_factory=lambda: GranularNormConfig()
    # )
    # """Configuration for granular normalization."""
    # widths: NerfMLPHiddenLayers = field(default_factory=lambda: NerfMLPHiddenLayers())
    # """The number of active neurons per layer. If empty, defaults to creating an
    # MLP with `net_depth` layers` and `net_width` neurons per layer.
    # """

    @staticmethod
    def from_dict(data: dict) -> "ElasticMLPConfig":
        cls = ElasticMLPConfig()
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


class ElasticMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: Optional[int] = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
        granular_norm: Optional[GranularNormConfig] = None,
        elastic_widths: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init
        self.use_granular_norm = True if granular_norm else False
        self.granular_norm = granular_norm
        if self.use_granular_norm:
            print(f"Using granular norms")
        self.elastic_widths = elastic_widths if elastic_widths is not None else {}

        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if self.granular_norm else None

        in_features = self.input_dim

        for i in range(self.net_depth):
            layer_name = f"hidden_layers.{i}"
            layer_width = (
                self.elastic_widths[layer_name]
                if layer_name in self.elastic_widths
                else self.net_width
            )
            if layer_width == 0:
                raise NotImplementedError("Layer width of 0 is not supported.")

            self.hidden_layers.append(
                nn.Linear(in_features, layer_width, bias=bias_enabled)
            )

            if self.granular_norm:
                self.norm_layers.append(
                    self.granular_norm.setup(num_features=layer_width)
                )

            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                in_features = layer_width + self.input_dim
            else:
                in_features = layer_width
        if self.output_enabled:
            self.output_layer = MuReadout(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            # Note that while the assignment RHS is labelled as `in_features`,
            # this was last set to `layer_width` in the loop above.
            self.output_dim = in_features
            raise ValueError("Output layer must be enabled.")

        # self.initialize()

    def initialize(self):
        raise ValueError("Initialize with mup manually please")

    # def init_func_hidden(m):
    #     if isinstance(m, nn.Linear):
    #         if self.hidden_init is not None:
    #             self.hidden_init(m.weight)
    #         if self.bias_enabled and self.bias_init is not None:
    #             self.bias_init(m.bias)

    # self.hidden_layers.apply(init_func_hidden)
    # if self.output_enabled:

    #     def init_func_output(m):
    #         if isinstance(m, nn.Linear):
    #             if self.output_init is not None:
    #                 self.output_init(m.weight)
    #             if self.bias_enabled and self.bias_init is not None:
    #                 self.bias_init(m.bias)

    # self.output_layer.apply(init_func_output)

    def forward_hidden(self, x, active_neurons, input_dim, use_granular_norm=False):
        inputs = x

        first_layer = True
        skip_layer = False
        pseudo_output_layer = False
        if not self.output_enabled:
            pseudo_output_layer_idx = len(self.hidden_layers) - 2

        for i, layer in enumerate(self.hidden_layers):
            W, b = slice_weights_and_biases_from_linear_layer(
                layer,
                active_neurons,
                input_dim,
                first_layer=first_layer,
                skip_layer=skip_layer,
                output_layer=pseudo_output_layer,
            )
            first_layer = False

            if use_granular_norm:
                norm = self.norm_layers[i]
                x = linear_layer_muladd_granularnorm_act(
                    x, W, b, norm, self.hidden_activation
                )
            else:
                x = linear_layer_muladd_act(x, W, b, self.hidden_activation)

            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                x = torch.cat([x, inputs], dim=-1)
                skip_layer = True
            elif (not self.output_enabled) and (i == pseudo_output_layer_idx):
                pseudo_output_layer = True
            else:
                skip_layer = False

        return x

    def forward(self, x, active_neurons: Optional[int] = None):
        if active_neurons is None:
            active_neurons = self.net_width

        x = self.forward_hidden(
            x, active_neurons, self.input_dim, self.use_granular_norm
        )

        if self.output_enabled:
            W_out, b = slice_weights_and_biases_from_linear_layer(
                self.output_layer,
                active_neurons,
                self.input_dim,
                first_layer=False,
                skip_layer=False,
                output_layer=True,
            )

            x = linear_layer_muladd_act(x, W_out, b, self.output_activation)

        return x

    def get_hidden_layers_at_granularity(self, active_neurons):
        first_layer = True
        skip_layer = False

        granular_hidden_layers_state_dict = self.hidden_layers.state_dict()
        for i, layer in enumerate(self.hidden_layers):
            # layer_weight = hidden_layers_state_dict[f"{i}.weight"]
            W, b = slice_weights_and_biases_from_linear_layer(
                layer,
                active_neurons,
                self.input_dim,
                first_layer,
                skip_layer,
            )
            first_layer = False

            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                skip_layer = True

            granular_hidden_layers_state_dict[f"{i}.weight"] = W
            if b is not None:
                granular_hidden_layers_state_dict[f"{i}.bias"] = b

        return granular_hidden_layers_state_dict

    def get_spectral_loss(self, active_neurons):
        elastic_state_dict = self.state_dict(active_neurons=active_neurons)
        loss = 0
        for name, param in elastic_state_dict.items():
            fanout, fanin = param.shape
            variance = torch.var(param)
            desired_variance = (1 / float(fanin)) * min(1, fanout / fanin)
            loss += (variance - desired_variance) ** 2

        return loss

    def state_dict(
        self, active_neurons=None, destination=None, prefix="", keep_vars=False
    ):
        if active_neurons is None:
            # If no active_neurons specified, return the standard state dict
            return super().state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        # Custom state dict for specific granularity
        granular_hidden_layers_state_dict = self.get_hidden_layers_at_granularity(
            active_neurons
        )
        custom_state_dict = {}
        for name, param in (
            super()
            .state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
            .items()
        ):
            if "hidden_layers" in name:
                hidden_layer_name = name.split("hidden_layers.")[1]
                if hidden_layer_name in granular_hidden_layers_state_dict:
                    custom_state_dict[name] = granular_hidden_layers_state_dict[
                        hidden_layer_name
                    ]
            elif "output_layer" in name:
                if "weight" in name:
                    custom_state_dict[name] = param[:, :active_neurons]
                elif "bias" in name:
                    custom_state_dict[name] = param
            elif "norm_layers" in name:
                custom_state_dict[name] = param[:active_neurons]
            else:
                custom_state_dict[name] = param

        return custom_state_dict

    def get_sliced_net(self, new_net_width: int):
        """
        Generates a new ElasticMLP instance with the specified net_width size,
        using the sliced weights and biases from the original model's elastic state dict.

        Args:
        new_net_width (int): The width for the new MLP layers.

        Returns:
        ElasticMLP: A new ElasticMLP instance with adjusted layer sizes.
        """
        # Create a new ElasticMLP instance with the same configuration but the new width
        new_model = ElasticMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            net_depth=self.net_depth,
            net_width=new_net_width,
            skip_layer=self.skip_layer,
            hidden_init=self.hidden_init,
            hidden_activation=self.hidden_activation,
            output_enabled=self.output_enabled,
            output_init=self.output_init,
            output_activation=self.output_activation,
            bias_enabled=self.bias_enabled,
            bias_init=self.bias_init,
            granular_norm=self.granular_norm,
        )

        # Use the existing state_dict method to get the elastic state dict for the new_net_width
        elastic_state_dict = self.state_dict(active_neurons=new_net_width)

        # Load the elastic state dict into the new model
        new_model.load_state_dict(elastic_state_dict)

        return new_model

    def pad_input_to_16(self, tensor, value=0, block_size=16):
        output_dim = tensor.shape[1]
        pad_size = (block_size - (output_dim % block_size)) % block_size
        return nn.functional.pad(tensor, pad=(0, pad_size), value=value)

    def pad_output_to_16(self, tensor, value=0, block_size=16):
        output_dim = tensor.shape[0]
        pad_size = (block_size - (output_dim % block_size)) % block_size
        return nn.functional.pad(tensor, pad=(0, 0, 0, pad_size), value=value)

    def get_packed_weights(self, block_size=16):
        layers = []
        input_layer_padded = self.pad_input_to_16(
            self.hidden_layers[0].weight.data, value=0, block_size=block_size
        )
        layers.append(input_layer_padded.flatten())

        for i in range(1, self.net_depth):
            layer_padded = self.pad_output_to_16(
                self.hidden_layers[i].weight.data, value=0, block_size=block_size
            )
            layers.append(layer_padded.flatten())

        output_layer_padded = self.pad_output_to_16(
            self.output_layer.weight.data, value=0, block_size=16
        )
        layers.append(output_layer_padded.flatten())
        params = torch.cat(layers)
        return params

    def make_fused(self, width, params: Optional[torch.Tensor] = None):
        ff_supported_widths = [16, 32, 64, 128]
        otype = "FullyFusedMLP" if width in ff_supported_widths else "CutlassMLP"
        fused_model = tcnn.Network(
            n_input_dims=self.input_dim,
            n_output_dims=self.output_dim,
            network_config={
                "otype": otype,
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": width,
                "n_hidden_layers": self.net_depth,
            },
        )
        if params is not None:
            fused_model.params.data[...] = params

        return fused_model

    def load_fused(self, elastic_width):
        block_size = 8 if elastic_width == 8 else 16
        params = self.get_packed_weights(block_size=block_size)
        fused_model = self.make_fused(elastic_width, params)
        return fused_model


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


@dataclass
class NerfMLPConfig(FlexibleInstantiateConfig):
    _target: Type = field(default_factory=lambda: NerfMLP)

    net_depth: int = 8
    """The depth of the MLP."""
    net_width: int = 256
    """The width of the MLP."""
    skip_layer: int = 4
    """The layer to add skip layers to."""
    net_depth_condition: int = 1
    """The depth of the second part of MLP."""
    net_width_condition: int = 128
    """The width of the second part of MLP."""
    output_enabled: bool = False
    """Whether to enable the output layer."""
    use_elastic: bool = True
    """Whether to use an elastic MLP."""
    base: ElasticMLPConfig = field(
        default_factory=lambda: ElasticMLPConfig(output_activation=None)
    )
    """Configuration if using an elastic MLP."""


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        output_enabled: bool = False,  # Whether to enable the output layer.
        use_elastic: bool = False,  # Whether to use an elastic MLP.
        base: Optional[ElasticMLPConfig] = None,
    ):
        super().__init__()
        self.use_elastic = use_elastic

        if use_elastic:
            assert base is not None
            self.base: ElasticMLP = base.setup(
                input_dim=input_dim,
                net_depth=net_depth,
                net_width=net_width,
                skip_layer=skip_layer,
                output_enabled=output_enabled,
            )
        else:
            self.base = MLP(
                input_dim=input_dim,
                net_depth=net_depth,
                net_width=net_width,
                skip_layer=skip_layer,
                output_enabled=output_enabled,
            )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x, active_neurons=None):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons
        x = self.base(x, **kwargs)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None, active_neurons=None):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons
        x = self.base(x, **kwargs)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (int(self.use_identity) + (self.max_deg - self.min_deg) * 2) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


@dataclass
class VanillaNeRFRadianceFieldConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VanillaNeRFRadianceField)

    mlp: NerfMLPConfig = field(default_factory=lambda: NerfMLPConfig())


class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        config: VanillaNeRFRadianceFieldConfig,
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp: NerfMLP = config.mlp.setup(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
        )

    def query_opacity(self, x, step_size, active_neurons=None):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons
        density = self.query_density(x, **kwargs)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x, active_neurons=None):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons

        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x, **kwargs)
        return F.relu(sigma)

    def forward(self, x, condition=None, active_neurons=None):
        kwargs = {}
        if active_neurons is not None:
            kwargs["active_neurons"] = active_neurons
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition, **kwargs)
        return torch.sigmoid(rgb), F.relu(sigma)

    def load_elastic_width(
        self,
        elastic_width: int,
    ):
        """Replaces base size modules with their smaller width version."""
        self.mlp.base = self.mlp.base.get_sliced_net(elastic_width)

    def load_width(self, elastic_width: int, load_fused: bool):
        if not self.mlp.use_elastic:
            return

        if load_fused:
            raise ValueError("Fused loading is not supported for Vanilla NeRF MLPs.")

        self.load_elastic_width(elastic_width)


# class TNeRFRadianceField(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
#         self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
#         self.warp = MLP(
#             input_dim=self.posi_encoder.latent_dim + self.time_encoder.latent_dim,
#             output_dim=3,
#             net_depth=4,
#             net_width=64,
#             skip_layer=2,
#             output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
#         )
#         self.nerf = VanillaNeRFRadianceField()

#     def query_opacity(self, x, timestamps, step_size):
#         idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
#         t = timestamps[idxs]
#         density = self.query_density(x, t)
#         # if the density is small enough those two are the same.
#         # opacity = 1.0 - torch.exp(-density * step_size)
#         opacity = density * step_size
#         return opacity

#     def query_density(self, x, t):
#         x = x + self.warp(
#             torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1)
#         )
#         return self.nerf.query_density(x)

#     def forward(self, x, t, condition=None):
#         x = x + self.warp(
#             torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1)
#         )
#         return self.nerf(x, condition=condition)
