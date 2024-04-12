import math
from torch.optim import Adam
from mup.optim import process_param_groups
from collections import defaultdict
from torch.nn.init import _calculate_fan_in_and_fan_out

# class ElasticMuAdam(Adam):
#     def __init__(self, global_lr, params, **kwargs):
#         self.global_lr = global_lr
#         super().__init__(params, **kwargs)

#     def step(self, closure=None):
#         """Prior to stepping the optimizer, creates new parameter groups based on
#         scaling the learning rate based on the spectral norm of the gradients.
#         """
#         new_param_groups = []
#         for param_group in self.param_groups:
#             # For every existing param group, we split into several new groups
#             def new_group():
#                 new_g = {k: v for k, v in param_group.items() if k != "params"}
#                 new_g["params"] = []
#                 return new_g

#             # The matrix-like weights might need multiple groups since weights
#             # might have different width multipliers
#             spectral_scaling_groups = defaultdict(new_group)
#             for p in param_group["params"]:
#                 assert hasattr(p, "infshape"), (
#                     f"A parameter with shape {p.shape} does not have `infshape` attribute. "
#                     "Did you forget to call `mup.set_base_shapes` on the model?"
#                 )
#                 if p.ndim == 1:
#                     # mup paper says for biases, fanin is 1.
#                     # Assume a tensor of shape `torch.Size[n]`. To get fanin of 1,
#                     # we need to add a dimension to make it `torch.Size[n, 1]`.
#                     fanin, fanout = _calculate_fan_in_and_fan_out(p.unsqueeze(-1))
#                 else:
#                     fanin, fanout = _calculate_fan_in_and_fan_out(p)

#                 if p.infshape.ninf() == 0:
#                     scaling = 1
#                 else:
#                     scaling = math.sqrt(fanout / fanin)
#                 spectral_scaling_groups[scaling]["params"].append(p)


def ElasticMuAdam(params, impl=Adam, **kwargs):
    """Adam with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
    """
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        spectral_scaling_groups = defaultdict(new_group)  # key is scaling factor
        for p in param_group["params"]:
            assert hasattr(p, "infshape"), (
                f"A parameter with shape {p.shape} does not have `infshape` attribute. "
                "Did you forget to call `mup.set_base_shapes` on the model?"
            )
            if p.ndim == 1:
                # mup paper says for biases, fanin is 1.
                # Assume a tensor of shape `torch.Size[n]`. To get fanin of 1,
                # we need to add a dimension to make it `torch.Size[n, 1]`.
                fanin, fanout = _calculate_fan_in_and_fan_out(p.unsqueeze(-1))
            else:
                fanin, fanout = _calculate_fan_in_and_fan_out(p)

            if p.infshape.ninf() == 0:
                scaling = 1
            else:
                fan_out, fan_in = p.weight.infshape[:2]
                # following are needed to accomodate SP models where all infshapes are finite so base_dims are Nones
                fan_out_base_dim = fan_out.base_dim or fan_out.dim
                fan_in_base_dim = fan_in.base_dim or fan_in.dim
                scaling = (fanout / fanin) / (fan_out_base_dim / fan_in_base_dim)
            spectral_scaling_groups[scaling]["params"].append(p)

        for scaling, group in spectral_scaling_groups.items():
            # Scale learning rate and weight decay accordingly
            group["lr"] /= scaling
            group["weight_decay"] *= scaling
        new_param_groups.extend(list(spectral_scaling_groups.values()))
    return impl(new_param_groups, **kwargs)
