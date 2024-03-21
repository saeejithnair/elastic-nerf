# %%
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
base_model = MyModel(width=1)
delta_model = MyModel(width=2)
print(base_model)
print(_extract_shapes(base_model))
print(delta_model)
print(_extract_shapes(delta_model))


# %%
def compute_expected_mup_stats(model, name, param):
    if "weight" in name:
        fan_in, fan_out = param.shape[-2], param.shape[-1]
        if isinstance(getattr(model, name.split(".")[0]), nn.Linear):
            return {
                "mup_init_var": (
                    1 / fan_in if "readout" not in name else 1 / (fan_in**2)
                ),
                "mup_multiplier": 1,
                "mup_lr_sgd": fan_out if "readout" not in name else 1,
                "mup_lr_adam": 1 if "readout" not in name else 1 / fan_in,
            }
        else:
            raise NotImplementedError(
                f"Layer type not supported: {type(getattr(model, name.split('.')[0]))}"
            )
    elif "bias" in name:
        fan_in = param.shape[-1]
        return {
            "mup_init_var": 1 / fan_in,
            "mup_multiplier": fan_in if "readout" not in name else 1,
            "mup_lr_sgd": fan_in,
            "mup_lr_adam": 1,
        }
    else:
        raise NotImplementedError(
            f"Layer type not supported: {type(getattr(model, name.split('.')[0]))}"
        )


def compute_actual_mup_stats(model, optimizer, name, param):
    if "weight" in name:
        fan_in, fan_out = param.shape[-2], param.shape[-1]
        if isinstance(getattr(model, name.split(".")[0]), nn.Linear):
            return {
                "mup_init_var": param.var().item(),
                "mup_multiplier": 1,
                "mup_lr_sgd": optimizer.param_groups[0]["lr"]
                * (fan_out if "readout" not in name else 1),
                "mup_lr_adam": optimizer.param_groups[0]["lr"]
                * (1 if "readout" not in name else 1 / fan_in),
            }
        else:
            raise NotImplementedError(
                f"Layer type not supported: {type(getattr(model, name.split('.')[0]))}"
            )
    elif "bias" in name:
        fan_in = param.shape[-1]
        return {
            "mup_init_var": param.var().item(),
            "mup_multiplier": fan_in if "readout" not in name else 1,
            "mup_lr_sgd": optimizer.param_groups[0]["lr"] * fan_in,
            "mup_lr_adam": optimizer.param_groups[0]["lr"],
        }
    else:
        raise NotImplementedError(
            f"Layer type not supported: {type(getattr(model, name.split('.')[0]))}"
        )


# %%
def print_stats(layer_stats):
    for stat_name, stat_value in layer_stats.items():
        print(f"    {stat_name}: {stat_value:.4f}")


def evaluate_mup(model, optimizer):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}")
        expected_stats = compute_expected_mup_stats(model, name, param)
        actual_stats = compute_actual_mup_stats(model, optimizer, name, param)
        print("  Expected:")
        print_stats(expected_stats)
        print("  Actual:")
        print_stats(actual_stats)


# %%
widths = [3, 10, 30, 100]

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
            raise NotImplementedError(
                f"Layer type not supported: {type(getattr(model, name.split('.')[0]))}"
            )
    optimizer = MuSGD(model.parameters(), lr=0.1)
    print(f"\nModel width: {width}")
    evaluate_mup(model, optimizer)

# %%
