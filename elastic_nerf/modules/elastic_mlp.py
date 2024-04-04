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
Multi Layer Perceptron
"""
import unittest
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Tuple, Type, Union

import torch
from gonas.configs.base_config import FlexibleInstantiateConfig
from jaxtyping import Float
from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.external import TCNN_EXISTS, tcnn
from nerfstudio.utils.printing import print_tcnn_speed_warning
from nerfstudio.utils.rich_utils import CONSOLE
from torch import Tensor, nn


def activation_to_tcnn_string(activation: Union[nn.Module, None]) -> str:
    """Converts a torch.nn activation function to a string that can be used to
    initialize a TCNN activation function.

    Args:
        activation: torch.nn activation function
    Returns:
        str: TCNN activation function string
    """

    if isinstance(activation, nn.ReLU):
        return "ReLU"
    if isinstance(activation, nn.LeakyReLU):
        return "Leaky ReLU"
    if isinstance(activation, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(activation, nn.Softplus):
        return "Softplus"
    if isinstance(activation, nn.Tanh):
        return "Tanh"
    if isinstance(activation, type(None)):
        return "None"
    tcnn_documentation_url = "https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions"
    raise ValueError(
        f"TCNN activation {activation} not supported for now.\nSee {tcnn_documentation_url} for TCNN documentation."
    )


class MRL_Linear_Layer(nn.Module):
    def __init__(
        self, nesting_list: List, in_dim: int, out_dim: int, efficient=False, **kwargs
    ):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.in_dim = in_dim  # Input dimension
        self.efficient = efficient
        if self.efficient:
            setattr(
                self,
                f"nesting_classifier_{0}",
                nn.Linear(nesting_list[-1], self.num_classes, **kwargs),
            )
        else:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(
                    self,
                    f"nesting_classifier_{i}",
                    nn.Linear(num_feat, self.num_classes, **kwargs),
                )

    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()  # type: ignore
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()

    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:  # type: ignore
                    nesting_logits += (
                        torch.matmul(
                            x[:, :num_feat],
                            (self.nesting_classifier_0.weight[:, :num_feat]).t(),  # type: ignore
                        ),
                    )
                else:
                    nesting_logits += (
                        torch.matmul(
                            x[:, :num_feat],
                            (self.nesting_classifier_0.weight[:, :num_feat]).t(),  # type: ignore
                        )
                        + self.nesting_classifier_0.bias,  # type: ignore
                    )
            else:
                nesting_logits += (
                    getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),
                )

        return nesting_logits


@dataclass
class GranularNormConfig(FlexibleInstantiateConfig):
    _target: Type = field(default_factory=lambda: GranularNorm)

    eps: float = 1e-5
    """A small value added for numerical stability in normalization calculations."""
    normalization_method: Literal["var", "std"] = "var"
    """Normalization method to use."""
    enabled: bool = False
    """Whether to use GranularNorm for normalization."""


class GranularNorm(nn.Module):
    """
    A normalization layer tailored for Elastic MLP, GranularNorm adjusts the activations
    of each sample in a batch across its features. It applies normalization per sample and
    allows for learned scaling and shifting of each feature through gamma and beta parameters.

    Attributes:
        num_features (int): Maximum number of features (F) across all widths.
        eps (float): A small value added for numerical stability in normalization calculations.
        gamma (torch.nn.Parameter): Learnable scaling parameters for each feature.
        beta (torch.nn.Parameter): Learnable shifting parameters for each feature.

    Args:
        num_features (int): Maximum number of features (F).
        eps (float): Small value for numerical stability in normalization. Default: 1e-5.

    Input:
        x (torch.Tensor): A tensor of shape [B, active_neurons], where B is the batch size
                          and active_neurons is the number of active neurons for the layer.
        active_neurons (int): Number of active neurons in the current forward pass.

    Output:
        torch.Tensor: Normalized and adjusted tensor of shape [B, active_neurons]. Each active
                      feature of each sample in the batch is normalized, scaled, and shifted.

    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        normalization_method: Literal["var", "std"] = "std",
        enabled: bool = False,
    ):
        assert enabled is True, "GranularNorm is not enabled."
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.normalization_method = normalization_method
        if self.normalization_method == "std":
            self.normalization_fn = self.normalize_std
        elif self.normalization_method == "var":
            self.normalization_fn = self.normalize_var
        else:
            raise ValueError(
                f"Normalization method {self.normalization_method} not supported."
            )
        # Learnable parameters for scale and shift, initialized for each feature
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def normalize_var(self, x):
        # Compute the mean and variance for each sample in the batch across its features
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the data using the variance
        # Note: The square root of the variance is the standard deviation
        # so why doesn't this give identical results as normalize_std lol?
        # If we move self.eps outside the sqrt, the result is closer to normalize_std.
        # However, putting the self.eps inside the sqrt is how all other PyTorch
        # normalization layers do it, since it's supposed to be faster. I haven't
        # seen any significant difference in performance between the two methods though.
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return x_normalized

    def normalize_std(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + self.eps)
        return x_normalized

    def reset_parameters(self):
        """Reset the parameters to their default initialization"""
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        active_neurons = x.shape[-1]
        x_normalized = self.normalization_fn(x)

        # Apply a subset of learnable parameters (gamma and beta) based on active_neurons
        x_scaled = (
            self.gamma[:active_neurons] * x_normalized + self.beta[:active_neurons]
        )
        return x_scaled


class ElasticMLP(nn.Module):
    """Elastic Multilayer Perceptron

    Args:
        in_dim: Input layer dimension.
        hidden_dim: Hidden layer dimension, which is the max size used in the elastic MLP.
        num_hidden_layers: Number of hidden layers.
        out_dim: Output layer dimension.
        activation: Activation function for the hidden layer.
        out_activation: Output activation function.
        use_granular_norm: Whether to use GranularNorm for normalization.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        out_dim: int,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        use_granular_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.activation = activation
        self.out_activation = out_activation
        self.use_granular_norm = use_granular_norm
        # Create the hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(1, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        if self.use_granular_norm:
            # Create the normalization layers.
            self.norm_layers = nn.ModuleList(
                [GranularNorm(hidden_dim) for _ in range(num_hidden_layers)]
            )
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, x: torch.Tensor, active_neurons: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass for the ElasticMLP.

        Args:
            x: Input tensor.
            active_neurons: Number of active neurons to use in the hidden layers.
                If None, uses full hidden dim for each layer.

        Returns:
            torch.Tensor: Output tensor.
        """
        if active_neurons is None:
            active_neurons = self.hidden_dim

        # Validate that active_neurons is within the allowed range
        if not (0 < active_neurons <= self.hidden_dim):
            raise ValueError(
                "active_neurons must be within the range of hidden layer size."
            )

        # Forward pass through hidden layers
        first_layer = True
        for i, layer in enumerate(self.hidden_layers):
            if first_layer:
                # For the first layer, use the complete weight matrix and bias
                W = layer.weight[:active_neurons, :]
                b = layer.bias[:active_neurons]
                first_layer = False
            else:
                W = layer.weight[:active_neurons, :active_neurons]
                b = layer.bias[:active_neurons]

            x = torch.matmul(x, W.t()) + b
            if self.use_granular_norm:
                # Apply GranularNorm
                x = self.norm_layers[i](x)
            x = self.activation(x) if self.activation is not None else x

        # Forward pass through output layer
        W_out = self.output_layer.weight[:, :active_neurons]
        x = torch.matmul(x, W_out.t()) + self.output_layer.bias
        x = self.out_activation(x) if self.out_activation is not None else x

        return x

    def compute_active_weight_norm(self, active_neurons: Optional[int] = None) -> dict:
        if active_neurons is None:
            active_neurons = self.hidden_dim

        if not (0 < active_neurons <= self.hidden_dim):
            raise ValueError(
                "active_neurons must be within the range of hidden layer size."
            )

        norms = {}
        for i, layer in enumerate(self.hidden_layers):
            W = layer.weight[:active_neurons, :]
            b = layer.bias[:active_neurons]
            W_norm = torch.norm(W)
            b_norm = torch.norm(b)
            combined_norm = torch.sqrt(W_norm**2 + b_norm**2)
            layer_name = f"Layer{i+1}"
            norms.update(
                {
                    f"{layer_name}/weight": W_norm.item(),
                    f"{layer_name}/bias": b_norm.item(),
                    f"{layer_name}/combined": combined_norm.item(),
                }
            )

        # Norms for the output layer
        W_out = self.output_layer.weight[:, :active_neurons]
        b_out = self.output_layer.bias
        W_out_norm = torch.norm(W_out)
        b_out_norm = torch.norm(b_out)
        out_combined_norm = torch.sqrt(W_out_norm**2 + b_out_norm**2)
        norms.update(
            {
                "Output/weight": W_out_norm.item(),
                "Output/bias": b_out_norm.item(),
                "Output/combined": out_combined_norm.item(),
            }
        )

        return norms
