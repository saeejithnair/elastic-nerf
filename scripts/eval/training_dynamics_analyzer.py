# %%
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union
import torch.nn as nn
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.cm as cm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.train.train_elastic_ngp_occ import NGPOccTrainer


class TrainingDynamicsAnalyzer:
    def __init__(self, max_width: int, model: ElasticMLP, width_step: int = 1):
        self.max_width = max_width
        self.training_steps = []
        self.layer_norms = {
            param_name: {norm_type: {"1D": [], "2D": []} for norm_type in [1, 2, "fro"]}
            for param_name, p in model.named_parameters()
            if "hidden_layers" in param_name
        }
        self.width_step = width_step
        self.widths = list(range(1, self.max_width + 1, self.width_step))

    def compute_and_store_norms(self, model: ElasticMLP, step: int):
        self.training_steps.append(step)

        # Slicing the state dict only once per step, as it is common for all norm types
        sliced_state_dicts = [
            model.state_dict(active_neurons=width) for width in self.widths
        ]

        for param_name in self.layer_norms:
            for norm_type in self.layer_norms[param_name]:
                norms_1D = []
                for i, state_dict in enumerate(sliced_state_dicts):
                    # Compute 1D norms for all widths
                    param = state_dict.get(param_name)
                    if param is None:
                        raise ValueError(
                            f"Parameter {param_name} not found in state dict"
                        )
                    norm_value = torch.norm(param, p=norm_type).item()
                    norms_1D.append(norm_value)

                    if self.widths[i] == self.max_width:
                        # Additionally compute 2D norm at max_width
                        if param.ndim == 2:
                            norm_value = (
                                torch.norm(param, p=norm_type, dim=-1)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        elif param.ndim == 1:
                            norm_value = param.detach().cpu().numpy()
                        else:
                            raise ValueError(
                                f"Parameter {param_name} has unexpected shape {param.shape}"
                            )
                        self.layer_norms[param_name][norm_type]["2D"].append(norm_value)

                self.layer_norms[param_name][norm_type]["1D"].append(norms_1D)

    def plot_norms(self, figsize_scales=(8, 5)):
        for param_name, norm_types in self.layer_norms.items():
            nrows = len(norm_types)
            ncols = 2  # 1D and 2D norms
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(figsize_scales[0] * ncols, figsize_scales[1] * nrows),
            )
            for i, (norm_type, norm_data) in enumerate(norm_types.items()):
                norms_1D = np.array(
                    norm_data["1D"]
                )  # Convert list of lists to a 2D array

                # Plot 1D norms with a color gradient
                cmap = plt.cm.viridis
                norm = mcolors.Normalize(vmin=1, vmax=self.max_width)

                # Iterate through widths, apply the color map
                for width_idx in range(norms_1D.shape[1]):
                    color = cmap(norm(width_idx + 1))
                    axes[i][0].plot(
                        self.training_steps, norms_1D[:, width_idx], color=color
                    )

                # Create a colorbar with the correct scale
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(
                    sm, ticks=np.linspace(1, self.max_width, num=16), ax=axes[i][0]
                )

                axes[i][0].set_title(
                    f"Training Steps vs 1D p={norm_type} Norms for {param_name}"
                )
                axes[i][0].set_xlabel("Training Step")
                axes[i][0].set_ylabel(f"p={norm_type} Norm")

                # Plot 2D norms
                norms_2D = np.array(
                    norm_data["2D"]
                )  # Convert list of arrays to 2D array
                for neuron_idx in range(norms_2D.shape[1]):
                    color = cmap(norm(neuron_idx + 1))
                    axes[i][1].plot(
                        self.training_steps,
                        norms_2D[:, neuron_idx],
                        color=color,
                    )
                plt.colorbar(
                    sm, ticks=np.linspace(1, self.max_width, num=16), ax=axes[i][1]
                )
                axes[i][1].set_title(
                    f"Training Steps vs 2D p={norm_type} Norms for {param_name}"
                )
                axes[i][1].set_xlabel("Training Step")
                axes[i][1].set_ylabel(f"p={norm_type} Norm")

            fig.show()
            fig.tight_layout()


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
run_id = "z0bejqau"
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
    max_width=64, model=trainer.radiance_field.mlp_base.elastic_mlp
)
module_name = "radiance_field"
steps = sorted(list(run_results["weights_grads"][module_name]))
for step in tqdm(steps):
    ckpt = run_results["weights_grads"][module_name][step]
    trainer.load_weights_grads(ckpt, module_name)
    tda.compute_and_store_norms(trainer.radiance_field.mlp_base.elastic_mlp, step)
# %%
tda.plot_norms(figsize_scales=(10, 5))

# %%
