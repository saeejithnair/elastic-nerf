# %%
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from elastic_nerf.nerfacc.radiance_fields.mlp import (
    ElasticMLP,
    slice_weights_and_biases_from_linear_layer,
)
import math
from torch.nn.init import (
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
    _no_grad_fill_,
    _no_grad_normal_,
    _no_grad_uniform_,
    calculate_gain,
)

max_width = 256


def _inf_fan_adjust_xavier(scale, tensor):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    # following are needed to accomodate SP models where all infshapes are finite so base_dims are Nones
    fan_out_base_dim = max_width
    fan_in_base_dim = 32
    scale *= math.sqrt((fan_out + fan_in) / (fan_out_base_dim + fan_in_base_dim))

    fanin_width_mult = fan_in / fan_in_base_dim
    scale /= math.sqrt(fanin_width_mult)

    return scale


def xavier_uniform_(tensor, gain=1.0):
    """Drop-in replacement of `torch.nn.init.xavier_uniform_`.
    Note:
        -  if using this function, ensure `gain` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if gain = 1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    std = _inf_fan_adjust_xavier(std, tensor)
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)


mlp = ElasticMLP(
    input_dim=32, output_dim=16, net_depth=1, net_width=max_width, bias_enabled=False
)
# mlp.initialize()
gain = math.sqrt(2)
# gain = math.sqrt(1/2)
# xavier_uniform_(mlp.hidden_layers[0].weight, gain=gain)

# %%
weights = {}
means = {}
stds = {}
spectral_maxes = {}
chunks = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]
weight, _ = slice_weights_and_biases_from_linear_layer(
    mlp.hidden_layers[0], active_neurons=max_width, input_dim=32, first_layer=True
)
for i, (start, end) in enumerate(chunks):
    xavier_uniform_(weight[start:end, :])

for i in list(range(1, max_width)):
    params = mlp.state_dict(active_neurons=i)
    for param_name, param in params.items():
        if param_name not in weights:
            weights[param_name] = []
            means[param_name] = []
            stds[param_name] = []
            spectral_maxes[param_name] = []

        param, _ = slice_weights_and_biases_from_linear_layer(
            mlp.hidden_layers[0],
            active_neurons=i,
            input_dim=32,
            first_layer=True,
        )
        # torch.nn.init.xavier_normal_(param)
        # xavier_uniform_(param)
        param = param.detach().cpu()
        spectral_max = torch.linalg.matrix_norm(param, ord=2).item()
        param = param.numpy()
        weights[param_name].append(param)
        means[param_name].append(param.mean())
        stds[param_name].append(param.std())
        spectral_maxes[param_name].append(spectral_max)


# %%
# For each param in weights, plot the mean and std of the weights
# as a function of the active width
fig, ax = plt.subplots(2, 3, figsize=(10, 10))
for i, (param_name, param) in enumerate(weights.items()):
    ax[i, 0].plot(means[param_name], label="mean")
    ax[i, 1].plot(stds[param_name], label="std")
    ax[i, 2].plot(spectral_maxes[param_name], label="spectral_max")
    ax[i, 2].plot(1 / np.array(spectral_maxes[param_name]), label="1/spectral_max")
    ax[i, 2].plot(
        np.sqrt(np.array([float(i / 32) for i in range(max_width)])),
        label="sqrt(n_l / n_(l-1))",
    )
    ax[i, 2].plot(
        np.sqrt(1 / np.array([float(i / 32) for i in range(max_width)])),
        label="sqrt(n_(l-1) / n_l)",
    )
    # ax[i, 2].plot(
    #     np.sqrt(np.array([float(i) for i in range(max_width)])),
    #     label="sqrt(n)",
    # )
    ax[i, 0].set_title(f"{param_name}: mean")
    ax[i, 1].set_title(f"{param_name}: std")
    # ax[i, 2].set_title(f"{param_name}: spectral_max")
    ax[i, 0].legend()
    ax[i, 1].legend()
    ax[i, 2].legend()
plt.show()
# %%

import numpy as np


# def create_array(a, b, c, d):
#     array = np.zeros(64)

#     # Generate the first 8 elements
#     array[:8] = np.random.normal(loc=0, scale=a, size=8)

#     # Generate the next 8 elements
#     scale = np.sqrt(16 * b**2 - 8 * a**2) / np.sqrt(8)
#     array[8:16] = np.random.normal(loc=0, scale=scale, size=8)

#     # Generate the next 16 elements
#     scale = np.sqrt(32 * c**2 - 16 * b**2) / np.sqrt(16)
#     array[16:32] = np.random.normal(loc=0, scale=scale, size=16)

#     # Generate the last 32 elements
#     scale = np.sqrt(64 * d**2 - 32 * c**2) / np.sqrt(32)
#     array[32:] = np.random.normal(loc=0, scale=scale, size=32)

#     return array

import numpy as np


def covariance_function(i, j, scale):
    l = scale / 2  # Adjust the length scale based on the scale
    return np.exp(-((i - j) ** 2) / (2 * l**2))


def generate_array(a, b, c, d):
    n = 64
    covariance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i < 8 and j < 8:
                covariance_matrix[i, j] = covariance_function(i, j, 8)
            elif i < 16 and j < 16:
                covariance_matrix[i, j] = covariance_function(i, j, 16)
            elif i < 32 and j < 32:
                covariance_matrix[i, j] = covariance_function(i, j, 32)
            else:
                covariance_matrix[i, j] = covariance_function(i, j, 64)

    # Add a small positive value to the diagonal to ensure positive definiteness
    covariance_matrix += 1e-6 * np.eye(n)

    L = np.linalg.cholesky(covariance_matrix)
    z = np.random.normal(0, 1, n)
    x = np.dot(L, z)

    x[:8] *= a / np.sqrt(8)
    x[:16] *= b / np.sqrt(16)
    x[:32] *= c / np.sqrt(32)
    x *= d / np.sqrt(64)

    return x


# Example usage
array = generate_array(a=1.0, b=0.8, c=0.6, d=0.4)
print(array)


# Run the test
a = 1.0
b = 0.5
c = 0.25
d = 0.125

# Create the array
# array = create_array(a, b, c, d)
# %%
means = []
stds = []
for i in range(1, 65):
    means.append(np.mean(array[:i]))
    stds.append(np.std(array[:i]))
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(means, label="mean")
ax[1].plot(stds, label="std")
ax[0].set_title(f"mean")
ax[1].set_title(f"std")
ax[0].legend()
ax[1].legend()
plt.show()
# %%
len(array)
# %%
len(means)
# %%
