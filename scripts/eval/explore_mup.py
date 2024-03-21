# %%
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
def compute_expected_scaling_ratios(model, name, param):
    if "weight" in name:
        fanin, fanout = param.infshape[-2], param.infshape[-1]
        if isinstance(getattr(model, name.split(".")[0]), nn.Linear):
            return {
                "mup_init_var": fanin.width_mult(),
                "mup_multiplier": fanin.width_mult() if "readout" in name else 1,
                "mup_lr_sgd": 1,
                "mup_lr_adam": fanin.width_mult(),
            }
        else:
            raise NotImplementedError(
                f"Layer type not supported: {type(getattr(model, name.split('.')[0]))}"
            )
    elif "bias" in name:
        fanin = param.infshape[-1]
        return {
            "mup_init_var": fanin.width_mult(),
            "mup_multiplier": 1,
            "mup_lr_sgd": 1 / fanin.width_mult(),
            "mup_lr_adam": 1,
        }
    else:
        raise NotImplementedError(f"Parameter type not supported: {name}")


def get_nested_attr(obj, attr_path):
    attributes = attr_path.split(".")
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def compute_actual_scaling_ratios(models, optimizers, name):
    ratios = {}
    for stat_name in ["mup_init_var", "mup_multiplier", "mup_lr_sgd", "mup_lr_adam"]:
        ratios[stat_name] = []

    for model, optimizer in zip(models, optimizers):
        param = get_nested_attr(model, name)
        if "weight" in name:
            ratios["mup_init_var"].append(param.var().item())
            ratios["mup_multiplier"].append(param.infshape.width_mult())
            ratios["mup_lr_sgd"].append(
                next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if any(id(p) == id(param) for p in group["params"])
                    ),
                    None,
                )
            )
            ratios["mup_lr_adam"].append(
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
            ratios["mup_init_var"].append(param.var().item())
            ratios["mup_multiplier"].append(param.infshape.width_mult())
            ratios["mup_lr_sgd"].append(
                next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if any(id(p) == id(param) for p in group["params"])
                    ),
                    None,
                )
            )
            ratios["mup_lr_adam"].append(
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
        # axs[i].plot(
        #     widths, [expected_ratio] * len(widths), label="Expected", linestyle="--"
        # )
        axs[i].plot(widths, actual_ratios[stat_name], label="Actual")
        axs[i].set_xlabel("Width")
        axs[i].set_ylabel("Scaling Ratio")
        axs[i].set_title(stat_name)
        axs[i].legend()

    plt.tight_layout()
    plt.show()


# %%
widths = list(range(3, 4096+3, 32))
models = []
optimizers = []

for width in widths:
    base_model = MyModel(width=1)
    delta_model = MyModel(width=2)
    model = MyModel(width=width)
    set_base_shapes(model, base_model, delta=delta_model)
    for name, param in model.named_parameters():
        if "weight" in name:
            mup.init.xavier_normal_(param)
        elif "bias" in name:
            mup.init.uniform_(param, 0, 0)
        else:
            raise NotImplementedError(f"Parameter type not supported: {name}")
    optimizer = MuAdam(model.parameters(), lr=0.1)
    models.append(model)
    optimizers.append(optimizer)

expected = {}
actual_scaling = {}
actual_ratios = {}
for name, param in models[0].named_parameters():
    print(f"\nParameter: {name}")
    expected_ratios = compute_expected_scaling_ratios(models[0], name, param)
    actual_scaling_ratios, ratios = compute_actual_scaling_ratios(
        models, optimizers, name
    )
    plot_scaling_ratios(
        expected_ratios, actual_scaling_ratios, widths, name, "Scaling Ratios"
    )
    plot_scaling_ratios(expected_ratios, ratios, widths, name, "Hyperparameters")

    expected[name] = expected_ratios
    actual_scaling[name] = actual_scaling_ratios
    actual_ratios[name] = ratios

# %%
