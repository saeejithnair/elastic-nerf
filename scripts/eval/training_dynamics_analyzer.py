# %%
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Literal, Optional, Union
import torch.nn as nn
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
from matplotlib import colormaps as colormaps
from matplotlib.colors import ListedColormap

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.train.train_elastic_ngp_occ import NGPOccTrainer, NGPOccTrainerConfig


class TrainingDynamicsAnalyzer:
    def __init__(
        self,
        max_width: int,
        model: ElasticMLP,
        run_id: str,
        config: NGPOccTrainerConfig,
        width_step: int = 1,
    ):
        self.max_width = max_width
        self.training_steps = []
        self.norm_types = {
            "L1": 1,
            "L2": 2,
            "Frobenius": "fro",
            "Spectral (max)": "spectral_max",
            "Spectral (min)": "spectral_min",
        }
        self.run_id = run_id
        self.config = config
        self.output_dir = Path(
            "/home/user/workspace/elastic-nerf/generated/training_dynamics"
        )
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weight_norms = {
            param_name: {
                norm_type: {"1D": [], "2D": []} for norm_type in self.norm_types
            }
            for param_name, p in model.named_parameters()
            if "hidden_layers" in param_name
        }
        self.grad_norms = {
            param_name: {
                norm_type: {"1D": [], "2D": []} for norm_type in self.norm_types
            }
            for param_name, p in model.named_parameters()
            if "hidden_layers" in param_name
        }
        self.width_step = width_step
        self.widths = list(range(1, self.max_width + 1, self.width_step))
        set3_cmap = plt.get_cmap("Set2")
        colors = [set3_cmap(i) for i in range(7)]
        self.cmap = ListedColormap(colors)

    def compute_norm(self, param, norm_type):
        if self.norm_types[norm_type] == "spectral_max":
            return torch.linalg.matrix_norm(param, ord=2).item()
        elif self.norm_types[norm_type] == "spectral_min":
            return torch.linalg.matrix_norm(param, ord=-2).item()
        else:
            return torch.norm(param, p=self.norm_types[norm_type]).item()

    def get_color_for_width(self, width_idx, buckets, cmap_name="Set2"):
        # Find the nearest bucket
        bucket_idx = np.searchsorted(buckets, width_idx, side="right") - 1
        # Get the color from the colormap
        cmap = colormaps[cmap_name]
        return cmap(bucket_idx / len(buckets))

    def compute_gradients(self, model: ElasticMLP, sliced_state_dict):
        gradients = {}
        for name, p in model.named_parameters():
            if name in sliced_state_dict:
                grad = p.grad.clone().detach().cpu()
                weight = sliced_state_dict[name]
                if grad.ndim == 1:
                    grad = grad.unsqueeze(-1)

                if weight.ndim == 1:
                    weight = weight.unsqueeze(-1)
                shape = weight.shape
                grad = grad[: shape[0], : shape[1]]
                gradients[name] = grad

        return gradients

    def compute_norms_from_state_dict(self, sliced_state_dicts, norms_dict):
        for param_name in norms_dict:
            for norm_type in norms_dict[param_name]:
                norms_1D = []
                for i, state_dict in enumerate(sliced_state_dicts):
                    # Compute 1D norms for all widths
                    param = state_dict.get(param_name)
                    if param is None:
                        raise ValueError(
                            f"Parameter {param_name} not found in state dict"
                        )
                    if param.ndim == 1:
                        param = param.unsqueeze(-1)
                    norm_value = self.compute_norm(param, norm_type)
                    norms_1D.append(norm_value)

                norms_dict[param_name][norm_type]["1D"].append(norms_1D)

    def compute_and_store_norms(self, model: ElasticMLP, step: int):
        self.training_steps.append(step)

        # Slicing the state dict only once per step, as it is common for all norm types
        sliced_weight_state_dicts = [
            model.state_dict(active_neurons=width) for width in self.widths
        ]
        sliced_grad_state_dicts = [
            self.compute_gradients(model, state_dict)
            for state_dict in sliced_weight_state_dicts
        ]

        self.compute_norms_from_state_dict(sliced_weight_state_dicts, self.weight_norms)
        self.compute_norms_from_state_dict(sliced_grad_state_dicts, self.grad_norms)

    def plot_norms(
        self, figsize_scales=(8, 5), param_type: Literal["weights", "grads"] = "weights"
    ):
        norms_dict = self.weight_norms if param_type == "weights" else self.grad_norms
        nrows = len(self.norm_types)
        ncols = len(norms_dict)
        buckets = np.array([0, 1, 2, 4, 8, 16, 32, 64])
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_scales[0] * ncols, figsize_scales[1] * nrows),
        )

        for j, (param_name, norm_types) in enumerate(norms_dict.items()):
            for i, (norm_type, norm_data) in enumerate(norm_types.items()):
                norms_1D = np.array(
                    norm_data["1D"]
                )  # Convert list of lists to a 2D array

                # Plot 1D norms with a color gradient
                # norm = Normalize(vmin=1, vmax=self.max_width)
                norm = BoundaryNorm(buckets, len(buckets))
                # Iterate through widths, apply the color map
                for width_idx in range(norms_1D.shape[1]):
                    # val = 2 ** int(np.log2(width_idx + 1))
                    color = self.cmap(norm(width_idx))
                    # color = self.get_color_for_width(
                    #     self.widths[width_idx], buckets, cmap_name
                    # )
                    axes[i][j].plot(
                        self.training_steps, norms_1D[:, width_idx], color=color
                    )

                # Create a colorbar with the correct scale
                sm = cm.ScalarMappable(cmap=self.cmap, norm=norm)
                sm.set_array([])
                cb = plt.colorbar(
                    sm,
                    ticks=buckets,
                    format="%.0f",
                    ax=axes[i][j],
                )
                cb.ax.minorticks_off()

                axes[i][j].set_title(
                    f"{self.norm_types[norm_type]} Norm vs. Steps for {param_type.capitalize()}: {param_name}"
                )
                axes[i][j].set_xlabel("Training Step")
                axes[i][j].set_ylabel(f"{self.norm_types[norm_type]} Norm")

            # Which config parameters do we care about? Scene, # Samples, Sampling method
            fig.suptitle(
                f"Training Dynamics for Run {self.run_id} | Scene: {self.config.scene.capitalize()} | # Samples: {self.config.num_widths_to_sample} | Sampling Strategy: {self.config.sampling_strategy.capitalize()}"
            )
            fig.tight_layout()
            fig.savefig(
                self.output_dir / f"{self.run_id}_{param_type}_norms.jpg", dpi=300
            )


# %%
def get_checkpoints(run_id: str, results_dir: Path):
    ckpt_dir = results_dir / run_id / "checkpoints"
    ckpt_files = list(ckpt_dir.glob("*.pt"))
    # Parse checkpoint files based on the step number.
    # E.g. "0tlyjl6x_ship_10000.pt" -> 10000
    ckpt_steps = [int(f.stem.split("_")[-1]) for f in ckpt_files]
    # Create a dictionary of step -> checkpoint file
    ckpt_dict = dict(zip(ckpt_steps, ckpt_files))

    return ckpt_dict


def get_weights_grads(run_id: str, results_dir: Path):
    weights_grads_dir = results_dir / run_id / "weights_grads"
    weights_grads_files = list(weights_grads_dir.glob("*.pt"))
    # Parse weights_grads files based on the step number and model.
    # E.g. "radiance_field_step_10500.pt" -> ('radiance_field', 10500)
    weights_grads_info = []
    for f in weights_grads_files:
        model, step = f.stem.split("_step_")
        step = int(step)
        weights_grads_info.append((model, step))

    # Create a dictionary of {model: {step -> weights_grads file}}
    weights_grads_dict = defaultdict(dict)
    for i, (model, step) in enumerate(weights_grads_info):
        weights_grads_dict[model][step] = weights_grads_files[i]

    return weights_grads_dict


def get_config(run_id: str, results_dir: Path):
    config_file = results_dir / run_id / "config.yaml"
    # Load yaml file from string
    with open(config_file, "r") as file:
        yaml_string = file.read()
    return yaml_string


# %%
# run_id = "z0bejqau"
# run_id = "gnatcn6w"
# run_id = "qx56molz"
run_id = "v7kztsvf"
log_dir = Path("/home/user/shared/results/elastic-nerf") / run_id
wandb_dir = Path("/home/user/shared/wandb_cache/elastic-nerf") / run_id
results_cache_dir = Path("/home/user/shared/results/elastic-nerf")

run_results = {}
run_results["checkpoints"] = get_checkpoints(run_id, results_cache_dir)
run_results["config"] = get_config(run_id, results_cache_dir)
run_results["weights_grads"] = get_weights_grads(run_id, results_cache_dir)

results_dict = {}
config = run_results["config"]
trainer = NGPOccTrainer.load_trainer(
    config,
    log_dir=log_dir,
    wandb_dir=wandb_dir,
    ckpt_path=run_results["checkpoints"][20000],
)
# %%
tda = TrainingDynamicsAnalyzer(
    max_width=64,
    model=trainer.radiance_field.mlp_base.elastic_mlp,
    run_id=run_id,
    config=trainer.config,
)
module_name = "radiance_field"
steps = sorted(list(run_results["weights_grads"][module_name]))
for step in tqdm(steps):
    ckpt = run_results["weights_grads"][module_name][step]
    trainer.load_weights_grads(ckpt, module_name)
    tda.compute_and_store_norms(trainer.radiance_field.mlp_base.elastic_mlp, step)
# %%
tda.plot_norms(figsize_scales=(10, 5), param_type="weights")

# %%
tda.plot_norms(figsize_scales=(10, 5), param_type="grads")

# %%
