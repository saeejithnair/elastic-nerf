# %%
from collections import defaultdict
from altair import value
from mup import (
    MuReadout,
    make_base_shapes,
    set_base_shapes,
    MuSGD,
    MuAdam,
)
from mup.shape import load_base_shapes, _extract_shapes, get_infshapes
import mup
from sympy import evaluate
import torch
import torch.nn as nn
import math

# %%


class MyModel(nn.Module):
    def __init__(self, width, d_out=16):
        super().__init__()
        ### In model definition, replace output layer with MuReadout
        # readout = nn.Linear(width, d_out)
        self.layer1 = nn.Linear(3, width)
        self.layer2 = nn.Linear(width, width)
        self.readout = MuReadout(width, d_out)
        ### If tying weights with an input nn.Embedding layer, do
        # readout = MuSharedReadout(input_layer.weight)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.readout(x)


# ### Instantiate a base model
# base_model = MyModel(width=1)
# ### Optionally, use `torchdistx.deferred_init.deferred_init` to avoid instantiating the parameters
# ### Simply install `torchdistx` and use
# # base_model = torchdistx.deferred_init.deferred_init(MyModel, width=1)
# ### Instantiate a "delta" model that differs from the base model
# ###   in all dimensions ("widths") that one wishes to scale.
# ### Here it's simple, but e.g., in a Transformer, you may want to scale
# ###   both nhead and dhead, so the delta model should differ in both.
# delta_model = MyModel(width=2)  # Optionally use `torchdistx` to avoid instantiating

# ### Instantiate the target model (the model you actually want to train).
# ### This should be the same as the base model except
# ###   the widths could be potentially different.
# ### In particular, base_model and model should have the same depth.
# model = MyModel(width=100)

# ### Set base shapes
# ### When `model` has same parameter shapes as `base_model`,
# ###   `model` behaves exactly the same as `base_model`
# ###   (which is in PyTorch's default parametrization).
# ###   This provides backward compatibility at this particular model size.
# ###   Otherwise, `model`'s init and LR are scaled by μP.
# ### IMPORTANT: this should be called as soon as possible,
# ###   before re-initialization and optimizer definition.
# set_base_shapes(model, base_model, delta=delta_model)

# ### Alternatively, one can save the base model shapes in a file
# # make_base_shapes(base_model, delta_model, filename)
# ### and later set base shapes directly from the filename
# # set_base_shapes(model, filename)
# ### This is useful when one cannot fit both
# ###   base_model and model in memory at the same time

# ### Replace your custom init, if any
# for param in model.parameters():
#     ### If initializing manually with fixed std or bounds,
#     ### then replace with same function from mup.init
#     # torch.nn.init.uniform_(param, -0.1, 0.1)
#     mup.init.uniform_(param, -0.1, 0.1)
#     ### Likewise, if using
#     ###   `xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_`
#     ### from `torch.nn.init`, replace with the same functions from `mup.init`

# ### Use the optimizers from `mup.optim` instead of `torch.optim`
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# optimizer = MuSGD(model.parameters(), lr=0.1)

### Then just train normally
# %%

import matplotlib.pyplot as plt

base_model = MyModel(width=1)
delta_model = MyModel(width=2)
print(base_model)
print(_extract_shapes(base_model))
print(delta_model)
print(_extract_shapes(delta_model))


# %%
def compute_expected_scaling_ratios(param, lr, base_init_var):
    if "weight" in name:
        fanin, fanout = param.infshape.fanin_fanout()
    elif "bias" in name:
        fanin = 1
        fanout = param.infshape[0]
    else:
        raise NotImplementedError(f"Parameter type not supported: {name}")
    print(param.infshape, param.infshape.ninf())
    if param.infshape.ninf() == 2:
        # Hidden weight
        print(f"Hidden weight; fanin: {fanin}, fanout: {fanout}")
        return {
            "init_var": base_init_var * 1 / fanout.dim,
            "multiplier": 1,
            "lr_adam": lr * 1 / fanout.dim,
        }
    elif param.infshape.ninf() == 1:
        if fanout.isinf():
            # Input weight
            scale = fanin if isinstance(fanin, int) else fanin.dim
            print(f"Input weight; fanin: {fanin}, fanout: {fanout}")
            return {
                "init_var": base_init_var * 1 / fanin.dim,
                "multiplier": fanout.dim,
                "lr_adam": lr,
            }
        elif fanin.isinf():
            # Output weight
            print(f"Output weight; fanin: {fanin}")
            return {
                "init_var": base_init_var * 1 / fanin.dim,
                "multiplier": 1 / fanin.dim,
                "lr_adam": lr,
            }
        else:
            raise ValueError(f"Unexpected infshape {param.infshape}")
    elif param.infshape.ninf() == 0:
        # Output bias
        scale = fanin if isinstance(fanin, int) else fanin.dim
        print(f"Unknown weight; fanin: {fanin}, scale: {scale}")
        return {
            "init_var": base_init_var * 1 / scale,
            "multiplier": 1 / scale,
            "lr_adam": lr * 1 / scale,
        }
    else:
        raise NotImplementedError(f"Unexpected infshape {param.infshape}")


def compute_actual_scaling_ratios(models, optimizers, name):
    ratios = {}
    for stat_name in ["init_var", "multiplier", "lr_adam"]:
        ratios[stat_name] = []

    for model, optimizer in zip(models, optimizers):
        param = get_nested_attr(model, name)
        if "weight" in name:
            ratios["init_var"].append(param.var().item())
            ratios["multiplier"].append(param.infshape.width_mult())
            ratios["lr_adam"].append(
                next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if any(id(p) == id(param) for p in group["params"])
                    ),
                    None,
                )
            )
        elif "bias" in name:
            ratios["init_var"].append(param.var().item())
            ratios["multiplier"].append(param.infshape.width_mult())
            ratios["lr_adam"].append(
                next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if any(id(p) == id(param) for p in group["params"])
                    ),
                    None,
                )
            )
        else:
            raise NotImplementedError(f"Parameter type not supported: {name}")

    scaling_ratios = {}
    for stat_name, values in ratios.items():
        if values[0] == 0:
            scaling_ratios[stat_name] = [0 for value in values]
        else:
            scaling_ratios[stat_name] = [value / values[0] for value in values]

    return scaling_ratios, ratios


# %%
def plot_scaling_ratios(expected_ratios, actual_ratios, widths, name, mode):
    fig, axs = plt.subplots(1, len(expected_ratios), figsize=(20, 5))
    fig.suptitle(f"{mode} for {name}", fontsize=16)

    for i, (stat_name, expected_ratio) in enumerate(expected_ratios.items()):
        axs[i].plot(
            widths,
            expected_ratio,
            label="Expected",
            linestyle="--",
        )
        axs[i].plot(widths, actual_ratios[stat_name], label="Actual")
        axs[i].set_xlabel("Width")
        axs[i].set_ylabel(mode)
        axs[i].set_title(stat_name)
        axs[i].legend()

    plt.tight_layout()
    plt.show()


# %%
widths = list(range(3, 1024))
models = []
optimizers = []
lr = 0.1
expected = defaultdict(list)
for width in widths:
    base_model = MyModel(width=1)
    delta_model = MyModel(width=2)
    model = MyModel(width=width)
    set_base_shapes(model, base_model, delta=delta_model)
    metadata = []
    for name, param in model.named_parameters():
        if "weight" in name:
            initialized_param, debug = mup.init.xavier_normal_(param)
            actual_std = initialized_param.std().item()
            actual_var = initialized_param.var().item()
            details = f"{width};{name};{debug};{actual_std};{actual_var}"
            metadata.append((param, details))
        elif "bias" in name:
            mup.init.uniform_(param, 0, 0)
        else:
            raise NotImplementedError(f"Parameter type not supported: {name}")
    optimizer = MuAdam(model.parameters(), lr=lr)
    models.append(model)
    optimizers.append(optimizer)
    for i, (param, details) in enumerate(metadata):
        for group in optimizer.param_groups:
            if any(id(p) == id(param) for p in group["params"]):
                opt_details = f"{lr};{group['lr']};{group['betas']};{group['eps']}"
                print(f"{details};{opt_details}")

# %%
actual_scaling = {}
actual_ratios = {}
init_std = 1.0
for name, param in models[0].named_parameters():
    expected = defaultdict(list)
    for model in models:
        print(f"\nParameter: {name}")
        expected_ratios = compute_expected_scaling_ratios(param, lr, init_std)
        for stat_name in expected_ratios:
            expected[stat_name].append(expected_ratios[stat_name])

    actual_scaling_ratios, ratios = compute_actual_scaling_ratios(
        models, optimizers, name
    )
    plot_scaling_ratios(expected, ratios, widths, name, "Hyperparameters")


# %%
