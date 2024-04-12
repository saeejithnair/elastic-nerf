# %%
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence, Dict
from elastic_nerf.nerfacc.trainers.ngp_prop import NGPPropTrainer, NGPPropTrainerConfig
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP
from elastic_nerf.nerfacc.radiance_fields.ngp import ElasticMLPWithInputEncoding
import math

# from elastic_nerf.nerfacc.trainers.ngp_occ import NGPOccTrainer, NGPOccTrainerConfig
import torch
from pprint import pprint as pp


# Function to attach hooks to models
def attach_hooks(model, layer_records, width, prefix=""):
    for name, module in model.named_children():
        # Skip certain layers as needed
        layer_name = prefix + ("." if prefix else "") + name
        if isinstance(module, ElasticMLP) or isinstance(
            module, ElasticMLPWithInputEncoding
        ):
            print(f"Attaching hook to {layer_name}")
            module.register_forward_hook(get_hook_fn(layer_name, layer_records, width))
        attach_hooks(module, layer_records, width, layer_name)


# Hook function to record pre-activations
def get_hook_fn(layer_name, layer_records, width):
    def hook(module, input, output):
        # Record the norm of the output before activation
        for layer_result in module.layer_records:
            result = module.layer_records[layer_result]
            layer_result_name = f"{layer_name}.{layer_result}"
            if layer_result_name not in layer_records:
                layer_records[layer_result_name] = {}

            if width not in layer_records[layer_result_name]:
                layer_records[layer_result_name][width] = []

            layer_records[layer_result_name][width].append(result)

        module.layer_records = {}

    return hook


# Function to initialize training and record pre-activations
def train_and_record(trainer: NGPPropTrainer, layer_records, width):
    models_to_watch = trainer.models_to_watch
    for model_name, model in models_to_watch.items():
        print(f"Attaching hooks to {model_name}")
        attach_hooks(model, layer_records, width, model_name)

    trainer.train()

    return layer_records


def plot_layer_records(
    layer_records,
    metric="spectral",
    num_steps: int = 3,
    exclude="/input/",
    scale: bool = False,
):
    num_steps = min(
        num_steps,
        len(
            next(iter(layer_records.values()))[
                next(iter(next(iter(layer_records.values())).keys()))
            ]
        ),
    )
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 5))
    fig.suptitle(f"Coordinate Checking with Metric: {metric}")

    for step in range(num_steps):
        ax = axes[step] if num_steps > 1 else axes
        ax.set_title(f"Training Step {step+1}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Norm Value")

        for layer_name, width_data in layer_records.items():
            try:
                layer_name_base = layer_name.split("/")[0]
                if not metric in layer_name:
                    continue
                if exclude in layer_name:
                    continue
                widths = list(width_data.keys())
                if scale:
                    if "output/frobenius" in layer_name:
                        norms = [width_data[w][step] / math.sqrt(w) for w in widths]
                    else:
                        norms = []
                        for w in widths:
                            fanin = layer_records[f"{layer_name_base}/fanin"][w][step]
                            fanout = layer_records[f"{layer_name_base}/fanout"][w][step]
                            norms.append(
                                width_data[w][step] / (math.sqrt(fanout / fanin))
                            )
                else:
                    norms = [width_data[w][step] for w in widths]
                ax.plot(widths, norms, label=layer_name)
            except:
                continue

        if step == 2 or num_steps == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_yscale("log", base=2)
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.show()


def create_trainer(width: int, num_steps: int = 3, optimizer_lr: float = 0.1):
    ngp_prop_config = NGPPropTrainerConfig(
        enable_logging=False,
        fused_eval=False,
        device="cuda:0",
        scene="counter",
        dataset_name="mipnerf360",
        optimizer_lr=optimizer_lr,
        seed=42,
        num_train_widths=1,
        num_eval_elastic_widths=1,
        hidden_dim=width,
        max_steps=num_steps,
        save_checkpoints=False,
        save_weights_grads=False,
        enable_eval=False,
    )
    ngp_prop_config.radiance_field.use_elastic = True
    ngp_prop_config.radiance_field.use_elastic_head = True
    ngp_prop_config.density_field.use_elastic = True
    ngp_prop_config.radiance_field.head_depth = 1
    # ngp_prop_config.radiance_field.base.granular_norm.enabled = True
    # ngp_prop_config.radiance_field.head.granular_norm.enabled = True
    # ngp_prop_config.density_field.base.granular_norm.enabled = True
    ngp_prop_trainer = ngp_prop_config.setup()
    return ngp_prop_trainer


# %%
widths = [2**i for i in range(3, 10)]
layer_records = {}
for width in widths:
    print(f"Training with width: {width}")
    trainer = create_trainer(width, num_steps=5, optimizer_lr=0.1)
    train_and_record(trainer, layer_records, width)
    torch.cuda.empty_cache()
# %%
plot_layer_records(layer_records, "param/spectral", scale=True, exclude="None")
plot_layer_records(layer_records, "output/frobenius", scale=True, exclude="None")

# %%
plot_layer_records(layer_records, "grad/spectral", 5, scale=True, exclude="None")

# %%
# plot_layer_records(layer_records, "spectral", exclude="/output/")
# plot_layer_records(layer_records, "frobenius", exclude="/output/")
# plot_layer_records(layer_records, "l1", exclude="/output/")
# plot_layer_records(layer_records, "std", exclude="/output/")
# %%
# plot_layer_records(layer_records, "output_layer/std")

# %%
plot_layer_records(layer_records, "param/spectral", scale=True)

# %%
plot_layer_records(layer_records, "param/spectral")
# %%
# What do I expect?
# I expect the spectral norm at initialization to scale as prefactor * sqrt(fanout / fanin)
# I expect the frobenius (L2) norm to scale as sqrt(d)
# # %%
# plot_layer_records(layer_records, "param/std", scale=True)

# # %%
# plot_layer_records(layer_records, "param/spectral", scale=False)

# # %%
# plot_layer_records(layer_records, "param/spectral", scale=True)

# %%
plot_layer_records(layer_records, "output/l1", scale=False)

# %%
plot_layer_records(layer_records, "output/l1", scale=True)

# %%
plot_layer_records(layer_records, "param/l1", scale=False)

# %%
plot_layer_records(layer_records, "output/frobenius", scale=False)

# %%
plot_layer_records(layer_records, "output/frobenius", scale=True)

# %%
plot_layer_records(layer_records, "input/frobenius", scale=False, exclude="proposal")

# %%
plot_layer_records(
    layer_records, "input/frobenius", scale=False, exclude="proposal_net"
)

# %%
plot_layer_records(layer_records, "output/frobenius", scale=True, exclude="None")

# %%
