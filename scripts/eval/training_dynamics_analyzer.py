# %%
from collections import defaultdict
from re import sub
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Literal, Optional, Union
import torch.nn as nn
from vine import transform
from elastic_nerf.nerfacc.radiance_fields.mlp import ElasticMLP
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
from matplotlib import colormaps as colormaps
from matplotlib.colors import ListedColormap
from elastic_nerf.utils import wandb_utils as wu
import tyro

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.train.train_elastic_ngp_occ import NGPOccTrainer, NGPOccTrainerConfig


class TrainingDynamicsAnalyzer:
    def __init__(
        self,
        max_width: int,
        model: ElasticMLP,
        run_id: str,
        config: NGPOccTrainerConfig,
        sweep_id: str,
        width_step: int = 1,
    ):
        self.max_width = max_width
        self.training_steps = []
        self.norm_types = {
            "L1": 1,
            "Frobenius": "fro",
            "Spectral (max)": "spectral_max",
            "Spectral (min)": "spectral_min",
        }
        self.run_id = run_id
        self.sweep_id = sweep_id
        self.config = config
        self.output_dir = (
            Path("/home/user/workspace/elastic-nerf/generated/training_dynamics")
            / sweep_id
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
                if p.grad is None:
                    continue
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
                        continue
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
        self.compute_norms_from_state_dict(sliced_weight_state_dicts, self.weight_norms)

        if step > 0:
            # Gradient is not available for step 0 (before training has started).
            sliced_grad_state_dicts = [
                self.compute_gradients(model, state_dict)
                for state_dict in sliced_weight_state_dicts
            ]

            self.compute_norms_from_state_dict(sliced_grad_state_dicts, self.grad_norms)

    def plot_norms(
        self, figsize_scales=(8, 5), param_type: Literal["weights", "grads"] = "weights"
    ):
        norms_dict = self.weight_norms if param_type == "weights" else self.grad_norms
        nrows = len(self.norm_types)
        ncols = len(norms_dict)
        buckets = np.array([0, 1, 2, 4, 8, 16, 32, 64])
        # Gradient is not available for step 0 (before training has started)
        training_steps = (
            self.training_steps[1:] if param_type == "grads" else self.training_steps
        )
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_scales[0] * ncols, figsize_scales[1] * nrows),
            layout="constrained",
        )

        for j, (param_name, norm_types) in enumerate(norms_dict.items()):
            for i, (norm_type, norm_data) in enumerate(norm_types.items()):
                ax = axes[i] if ncols == 1 else axes[i][j]
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
                    ax.plot(training_steps, norms_1D[:, width_idx], color=color)

                # Create a colorbar with the correct scale
                sm = cm.ScalarMappable(cmap=self.cmap, norm=norm)
                sm.set_array([])
                cb = plt.colorbar(
                    sm,
                    ticks=buckets,
                    format="%.0f",
                    ax=ax,
                )
                cb.ax.minorticks_off()

                ax.set_title(
                    f"{norm_type} Norm vs. Steps for {param_type.capitalize()}: {param_name}"
                )
                ax.set_xlabel("Training Step")
                ax.set_ylabel(f"{norm_type} Norm")

            # Which config parameters do we care about? Scene, # Samples, Sampling method
            fig.suptitle(
                f"Training Dynamics for Run {self.run_id} | Scene: {self.config.scene.capitalize()} | # Samples: {self.config.num_widths_to_sample} | Sampling Strategy: {self.config.sampling_strategy.capitalize()}"
            )
            fig.savefig(
                self.output_dir / f"{self.run_id}_{param_type}_norms.jpg", dpi=300
            )


# %%
class SweepDynamicsPlotter:
    def __init__(self, sweep_id, log_dir, wandb_dir, results_cache_dir):
        self.sweep_id = sweep_id
        self.sweep = wu.fetch_sweep(sweep_id)
        self.log_dir = log_dir
        self.wandb_dir = wandb_dir
        self.results_cache_dir = results_cache_dir
        self.run_ids = []
        self.sweep_results = {}
        self.training_steps = []
        set3_cmap = plt.get_cmap("Set2")
        colors = [set3_cmap(i) for i in range(7)]
        self.cmap = ListedColormap(colors)
        self.buckets = np.array([0, 1, 2, 4, 8, 16, 32, 64])
        self.norm = BoundaryNorm(self.buckets, len(self.buckets))
        self.configs = []
        self.run_summaries = []
        self.output_dir = (
            Path("/home/user/workspace/elastic-nerf/generated/training_dynamics")
            / sweep_id
        )
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        for run in self.sweep.runs:
            self.run_summaries.append(wu.RunResult.download_summary(run))
            self.compute_training_dynamics(run.id)
            print(f"Computed training dynamics for run {run.id}")

        # Compute indices of the sorted run IDs based on scene name
        scene_names = [config.scene for config in self.configs]
        self.sorted_run_indices = np.argsort(scene_names)

    def get_checkpoints(self, run_id: str, results_dir: Path):
        ckpt_dir = results_dir / run_id / "checkpoints"
        ckpt_files = list(ckpt_dir.glob("*.pt"))
        # Parse checkpoint files based on the step number.
        # E.g. "0tlyjl6x_ship_10000.pt" -> 10000
        ckpt_steps = [int(f.stem.split("_")[-1]) for f in ckpt_files]
        # Create a dictionary of step -> checkpoint file
        ckpt_dict = dict(zip(ckpt_steps, ckpt_files))

        return ckpt_dict

    def get_weights_grads(self, run_id: str, results_dir: Path):
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

    def get_config(self, run_id: str, results_dir: Path):
        config_file = results_dir / run_id / "config.yaml"
        # Load yaml file from string
        with open(config_file, "r") as file:
            yaml_string = file.read()
        return yaml_string

    def create_trainer(self, run_id):
        logged_results = {}
        logged_results["checkpoints"] = self.get_checkpoints(
            run_id, self.results_cache_dir
        )
        logged_results["config"] = self.get_config(run_id, self.results_cache_dir)
        logged_results["weights_grads"] = self.get_weights_grads(
            run_id, self.results_cache_dir
        )

        config = logged_results["config"]
        trainer = NGPOccTrainer.load_trainer(
            config,
            log_dir=self.log_dir,
            wandb_dir=self.wandb_dir,
            ckpt_path=logged_results["checkpoints"][20000],
        )
        return trainer, logged_results

    def compute_training_dynamics(self, run_id):
        trainer, run_results = self.create_trainer(run_id)
        tda = TrainingDynamicsAnalyzer(
            max_width=64,
            model=trainer.radiance_field.mlp_base.elastic_mlp,
            run_id=run_id,
            config=trainer.config,
            sweep_id=self.sweep_id,
        )
        module_name = "radiance_field"
        steps = sorted(list(run_results["weights_grads"][module_name]))
        for step in tqdm(steps):
            ckpt = run_results["weights_grads"][module_name][step]
            trainer.load_weights_grads(ckpt, module_name)
            tda.compute_and_store_norms(
                trainer.radiance_field.mlp_base.elastic_mlp, step
            )

        self.run_ids.append(run_id)
        weight_norms = tda.weight_norms
        grad_norms = tda.grad_norms
        self.training_steps.append(tda.training_steps)
        self.configs.append(tda.config)
        for param_name in weight_norms:
            if param_name not in self.sweep_results:
                self.sweep_results[param_name] = {"weights": {}, "grads": {}}
            for norm_type in weight_norms[param_name]:
                if norm_type not in self.sweep_results[param_name]["weights"]:
                    self.sweep_results[param_name]["weights"][norm_type] = []
                self.sweep_results[param_name]["weights"][norm_type].append(
                    weight_norms[param_name][norm_type]["1D"]
                )
        for param_name in grad_norms:
            for norm_type in grad_norms[param_name]:
                if norm_type not in self.sweep_results[param_name]["grads"]:
                    self.sweep_results[param_name]["grads"][norm_type] = []
                self.sweep_results[param_name]["grads"][norm_type].append(
                    grad_norms[param_name][norm_type]["1D"]
                )

    def plot_sweep_dynamics(
        self,
        figsize_scales=(8, 5),
        param_type="weights",
        offset=0.05,
        num_runs_to_plot: Optional[int] = None,
    ):
        nrows = (
            min(len(self.run_ids), num_runs_to_plot)
            if num_runs_to_plot
            else len(self.run_ids)
        )
        for param_name in self.sweep_results:
            ncols = len(self.sweep_results[param_name][param_type])
            # Increase figure height to account for row titles
            fig_height = figsize_scales[1] * nrows + offset * nrows
            fig = plt.figure(
                constrained_layout=True,
                figsize=(figsize_scales[0] * ncols, fig_height),
            )
            fig.suptitle(
                f"Training Dynamics for Sweep {self.sweep_id}",
                fontsize="large",
                weight="bold",
                color="blue",
            )
            subfigs = fig.subfigures(nrows=nrows, ncols=1)
            for row, subfig in enumerate(subfigs):
                run_idx = self.sorted_run_indices[row]
                run_id = self.run_ids[run_idx]
                summary = self.run_summaries[run_idx]
                widths = [64, 32, 16, 8]
                keys = [f"Eval Results Summary/psnr_avg/elastic_{i}" for i in widths]
                metrics = [summary.get(key) for key in keys]
                filtered_metrics = [
                    (w, m) for w, m in zip(widths, metrics) if m is not None
                ]
                row_title = f"Run {run_id} | Scene: {self.configs[run_idx].scene.capitalize()} | # Samples: {self.configs[run_idx].num_widths_to_sample} | Sampling Strategy: {self.configs[run_idx].sampling_strategy.capitalize()}"
                row_title += (
                    " | Results: ("
                    + ", ".join([f"$\\mathbf{{{m:.3f}}}$" for _, m in filtered_metrics])
                    + ") PSNR for Widths ("
                    + ", ".join([str(w) for w, _ in filtered_metrics])
                    + ")"
                )
                subfig.suptitle(row_title, color="blue")

                axs = subfig.subplots(nrows=1, ncols=ncols)
                norm_types = list(self.sweep_results[param_name][param_type].keys())
                for col, ax in enumerate(axs):
                    norm_type = norm_types[col]
                    norms = self.sweep_results[param_name][param_type][norm_type]
                    norms_1D = np.array(norms[run_idx])
                    training_steps = (
                        self.training_steps[run_idx][1:]
                        if param_type == "grads"
                        else self.training_steps[run_idx]
                    )
                    for width_idx in range(norms_1D.shape[1]):
                        color = self.cmap(self.norm(width_idx))
                        ax.plot(training_steps, norms_1D[:, width_idx], color=color)

                    cb = plt.colorbar(
                        cm.ScalarMappable(cmap=self.cmap, norm=self.norm),
                        ticks=self.buckets,
                        format="%.0f",
                        ax=ax,
                    )
                    cb.ax.minorticks_off()
                    ax.set_title(
                        f"{norm_type} Norm vs. Steps for {param_name}: {param_type.capitalize()}"
                    )
                    ax.set_xlabel("Training Step")
                    ax.set_ylabel(f"{norm_type} Norm")
            fig.savefig(
                self.output_dir
                / f"{self.sweep_id}_{param_type}_{param_name}_norms.pdf",
                dpi=300,
            )


# %%
def main(sweep_id: str):
    log_dir = Path("/home/user/shared/results/elastic-nerf")
    wandb_dir = Path("/home/user/shared/wandb_cache/elastic-nerf")
    results_cache_dir = Path("/home/user/shared/results/elastic-nerf")
    sdp = SweepDynamicsPlotter(sweep_id, log_dir, wandb_dir, results_cache_dir)
    sdp.plot_sweep_dynamics(figsize_scales=(8, 5), param_type="weights")
    sdp.plot_sweep_dynamics(figsize_scales=(8, 5), param_type="grads")


# %%
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)

# %%
