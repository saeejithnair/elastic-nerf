# %%
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Sequence, Dict
from elastic_nerf.nerfacc.trainers.ngp_prop import NGPPropTrainer, NGPPropTrainerConfig
from elastic_nerf.nerfacc.trainers.ngp_occ import NGPOccTrainer, NGPOccTrainerConfig


# %%
def count_params(model, modules_to_skip=["encoding", "estimator"]) -> int:
    sum = 0
    for n, p in model.named_parameters():
        skip_module = False
        if not p.requires_grad:
            continue
        for m in modules_to_skip:
            if m in n:
                skip_module = True
                break
        if not skip_module:
            print(n, p.numel())
            sum += p.numel()

    return sum


def extract_params(models_info):
    params_dict = {}
    for width, model_info in models_info.items():
        for module_name, module in model_info.items():
            print(module_name)
            if isinstance(module, Sequence):
                for i, m in enumerate(module):
                    param_count = count_params(m)
                    if param_count == 0:
                        continue
                    if module_name not in params_dict:
                        params_dict[module_name] = {}
                    params_dict[module_name][f"{width}_{i}"] = param_count
            else:
                param_count = count_params(module)
                if param_count == 0:
                    continue
                if module_name not in params_dict:
                    params_dict[module_name] = {}
                params_dict[module_name][width] = param_count
    return params_dict


def plot_module_params(params_dict):
    for module_name, width_params in params_dict.items():
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Absolute parameter counts
        widths = list(width_params.keys())
        param_counts = list(width_params.values())
        color = "tab:blue"
        ax1.set_xlabel("Width")
        ax1.set_ylabel("Parameter Count", color=color)
        ax1.bar(
            widths,
            param_counts,
            color=color,
            alpha=0.6,
            label="Absolute Parameter Count",
        )
        ax1.tick_params(axis="y", labelcolor=color)

        # Relative parameter counts (percentage)
        ax2 = ax1.twinx()
        max_count = max(param_counts)
        relative_counts = [(count / max_count) * 100 for count in param_counts]
        color = "tab:red"
        ax2.set_ylabel("Relative Parameter Count (%)", color=color)
        ax2.plot(
            widths,
            relative_counts,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            label="Relative Parameter Count",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        # Combined legend for both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")

        # Final plot adjustments
        plt.title(f"Module: {module_name}")
        plt.xticks(widths)
        plt.show()


def generate_param_tables(params_dict):
    for module_name, width_params in params_dict.items():
        print(f"Module: {module_name}")
        df = pd.DataFrame(
            list(width_params.items()), columns=["Width", "Parameter Count"]
        )
        print(df, "\n")


# %%
# Evaluate NGP-Prop and NGP-Occ models across specified widths
ngp_prop_config = NGPPropTrainerConfig(
    enable_logging=False, fused_eval=True, device="cpu"
)
ngp_prop_config.radiance_field.use_elastic = True
ngp_prop_config.radiance_field.use_elastic_head = True
ngp_prop_config.density_field.use_elastic = True
ngp_prop_trainer = ngp_prop_config.setup()

ngp_occ_config = NGPOccTrainerConfig(
    enable_logging=False, fused_eval=True, device="cpu"
)
ngp_occ_config.radiance_field.use_elastic = True
ngp_occ_trainer = ngp_occ_config.setup()

# %%
modules = ngp_prop_trainer.get_modules_for_eval(8)

# %%
widths = range(1, 65, 1)
ngp_occ_info = {width: ngp_occ_trainer.get_modules_for_eval(width) for width in widths}

ngp_prop_info = {
    width: ngp_prop_trainer.get_modules_for_eval(width) for width in widths
}

# Extract and organize parameters
ngp_occ_params = extract_params(ngp_occ_info)
ngp_prop_params = extract_params(ngp_prop_info)

# Plot parameters
plot_module_params(ngp_occ_params)
plot_module_params(ngp_prop_params)

# Generate tables
generate_param_tables(ngp_occ_params)
generate_param_tables(ngp_prop_params)
# %%
